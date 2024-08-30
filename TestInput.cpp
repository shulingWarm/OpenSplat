#include<iostream>
#include <filesystem>
#include"input_data.hpp"
#include<vector>
#include<string>
#include"utils.hpp"
#include"model.hpp"
#include"tensor_math.hpp"
#include"Debug.hpp"

#define DEBUG_LOG(exp) std::cout<<exp<<std::endl;

static const int DATA_SIZE = 100;
static const int POINT_DIM = 3;

//与相机内参相关的参数，它们可以被视为连续的数组
static const size_t INTRINSIC_SIZE = 11;
//外参的数据量大小
static const size_t EXTERNAL_SIZE = 16;
//每个camera数据里面的数据量
static const size_t CAMERA_DATA_SIZE = (INTRINSIC_SIZE + EXTERNAL_SIZE);

//打印某一个具体相机的内参
static void printCameraIntrinsic(Camera& camera)
{
	std::cout << "camera: " <<
		camera.fx << " " << camera.fy << " " <<
		camera.cx << " " << camera.cy << std::endl;
}

//打印单个image的外参
static void printSingleExternal(Camera& camera)
{
	std::cout << "camera external: " << std::endl <<
		camera.camToWorld << std::endl;
}

//打印input数据里面的外参
static void printExternalParams(InputData& inputData)
{
	auto& images = inputData.cameras;
	//打印前两个image
	for (unsigned idImage = 0; idImage < 2; ++idImage)
	{
		auto& image = images[idImage];
		printSingleExternal(image);
	}
}

//打印input_data里面的内参
static void printIntrinsic(InputData& inputData)
{
	auto& images = inputData.cameras;
	//打印输入data的个数
	std::cout << "image num: " << images.size() << std::endl;
	//遍历两个相机
	for (int i = 0; i < 2; ++i)
	{
		//打印某一个相机的内参
		printCameraIntrinsic(images[i]);
	}
}

//打印input data里面的点坐标
static void printPoints(InputData& inputData)
{
	//遍历前10个点
	auto dataPtr = static_cast<uint8_t*>(inputData.points.rgb.data_ptr());
	for (int i = 0; i < 10; ++i)
	{
		std::cout << (int) dataPtr[i] << " ";
	}
	std::cout << std::endl;
}

//载入一个单个的相机
static void loadSingleCamera(
	Camera& targetCamera,
	float* cameraData
)
{
	//直接依次处理里面的每个数
	auto widthHead = (void*)(&targetCamera.width);
	memcpy(widthHead, cameraData, sizeof(float) * INTRINSIC_SIZE);
	//外参的头指针
	auto externalHead = cameraData + INTRINSIC_SIZE;
	//把数据复制到camera里面
	targetCamera.camToWorld = torch::from_blob(externalHead, { 4,4 }, torch::kFloat32);
}

//输入相机的信息，然后转换成camera的列表
static void loadCameraData(
	std::vector<Camera>& cameraList,
	float* cameraData, //相机数据
	size_t cameraNum //相机信息的个数
)
{
	cameraList.resize(cameraNum);
	//遍历每个可能被处理的camera
	for (size_t idCamera = 0; idCamera < cameraNum; ++idCamera)
	{
		//当前位置的头指针
		auto headPointer = cameraData + idCamera * CAMERA_DATA_SIZE;
		//当前位置的相机
		auto& currCamera = cameraList[idCamera];
		//用当前的头指针构造一个相机数据
		loadSingleCamera(currCamera, headPointer);
	}
}

//load所有的输入数据
static void loadAllInput(
	InputData& inputData,
	//相机的个数
	const size_t cameraNum,
	float* cameraData,
	//点的个数
	const size_t pointNum,
	float* pointData,
	float* colorData,
	std::vector<std::string>& imgPathList
)
{
	//需要确定imgPathList和camera的数量是一样的
	if (cameraNum != imgPathList.size()) throw - 2;
	//载入相机数据
	loadCameraData(inputData.cameras, cameraData, cameraNum);
	//初始化scale，就直接设置成1就行
	inputData.scale = 1;
	//初始化translation
	inputData.translation = torch::ones({ 3 });
	//初始化点数据和颜色数据
	inputData.points.xyz = torch::from_blob(pointData, 
		{ static_cast<unsigned int>(pointNum),3}, torch::kFloat32);
	inputData.points.rgb = torch::from_blob(colorData,
		{ static_cast<unsigned int>(pointNum),3 }, torch::kFloat32);
	//循环记录每个camera的图片路径 
	for (size_t idCamera = 0; idCamera < imgPathList.size(); ++idCamera)
	{
		auto& camera = inputData.cameras[idCamera];
		camera.filePath = imgPathList[idCamera];
	}
}

//执行3D gaissoam splatting的主流程
static void splatCompute(InputData inputData)
{
	parallel_for(inputData.cameras.begin(), inputData.cameras.end(), [](Camera& cam) {
		cam.loadImage(1);
	});

	// Withhold a validation camera if necessary
	auto t = inputData.getCameras(false, "random");
	std::vector<Camera> cams = std::get<0>(t);
	Camera* valCam = std::get<1>(t);

	//总的迭代次数
	const int numIters = 300;
	//主要使用的device
	auto device = torch::kCUDA;
	//结构相似度的权重占比
	float structureSimiliarWeight = 0.2f;

	Model model(inputData,
		cams.size(),
		2,//默认会将图片缩小一倍 后面可以酌情关闭
		3000, //这属于是每迭代3000次就把图片的分辨率翻倍一次
		3, //球谐函数的最大度数，目前也不知道这是干啥用的
		1000, //这是每迭代1000次就增大一次球谐函数的最大度数
		100, //这可能才是正经比较重要的参数,每迭代100次就更新一下3DGS的大小
		500,//前500次迭代完全不考虑分割3DGS
		30,//每30次高斯优先就会把所有的3DGS的透明度重置一次
		4000,//位置梯度的阈值，目前不知道这是做什么用的
		0.0002,//3DGS的切割阈值 refine的时候如果这个东西太大就切割一下
		4000,//超过这个迭代次数就不再考虑切割3DGS了
		0.05, //另一个衡量切割3DGS的指标，如果这个3DGS弄出来超过了0.05倍的屏幕大小，那必须切了它
		numIters, //迭代的次数，可能是最需要频繁调整的一个东西了
		false, //这可能指的是保持原本的相对大小，似乎不是很重要
		device //使用cuda来做初始化
	);

	std::vector< size_t > camIndices(cams.size());
	std::iota(camIndices.begin(), camIndices.end(), 0);
	InfiniteRandomIterator<size_t> camsIter(camIndices);

	//这东西是迭代过程中使用到的局部就是，不知道是干嘛用的
	//总之就是初始化成10，可能是表示展示中间结果的周期
	int displayStep = 10;
	//输出路径，其实以后不需要这个东西，放在这里也仅仅是为了测试
	std::string outputScene = "E:/temp/splat.ply";

	int imageSize = -1;
	for (size_t step = 1; step <= numIters; step++) {
		Camera& cam = cams[camsIter.next()];

		model.optimizersZeroGrad();

		torch::Tensor rgb = model.forward(cam, step);
		torch::Tensor gt = cam.getImage(model.getDownscaleFactor(step));
		gt = gt.to(device);

		torch::Tensor mainLoss = model.mainLoss(rgb, gt, structureSimiliarWeight);

		mainLoss.backward();

		if (step % displayStep == 0) std::cout << "Step " << step << ": " << mainLoss.item<float>() << std::endl;

		model.optimizersStep();
		model.schedulersStep(step);
		model.afterTrain(step);
	}

	model.save(outputScene);
}

//读取自制的colmap输入
static void loadPointInput(std::string filePath,
	InputData& inputData
)
{
	std::fstream fileHandle(filePath,std::ios::in|std::ios::binary);
	//读取点的个数
	unsigned pointNum = 0;
	fileHandle.read((char*)&pointNum,sizeof(unsigned));
	DEBUG_LOG("pointNum: " << pointNum);
	//遍历读取每个点
	float* pointData = new float[pointNum * 3];
	uint8_t* colorData = new uint8_t[pointNum * 3];
	//遍历读取每个位置的颜色
	for (unsigned idPoint = 0; idPoint < pointNum; ++idPoint)
	{
		auto xyzHead = pointData + idPoint * 3;
		auto colorHead = colorData + idPoint * 3;
		//读取点坐标
		fileHandle.read((char*)xyzHead, sizeof(float) * 3);
		fileHandle.read((char*)colorHead, sizeof(uint8_t) * 3);
	}
	fileHandle.close();
	//用点数据和颜色数据生成blob
	inputData.points.xyz = torch::from_blob(pointData, { pointNum,3 }, torch::kFloat32).clone();
	inputData.points.rgb = torch::from_blob(colorData, { pointNum,3 }, torch::kU8).clone();
}

//读取input的内参部分
static void readIntrinsicParams(
	std::string filePath,
	InputData& inputData
)
{
	std::fstream fileHandle;
	fileHandle.open(filePath,std::ios::in|std::ios::binary);
	//读取相机的个数
	unsigned cameraNum;
	fileHandle.read((char*)&cameraNum, sizeof(unsigned));
	//确保内参只有一个
	MY_ASSERT(cameraNum == 1);
	//读取图片的shape
	unsigned imgShape[2];
	fileHandle.read((char*)imgShape, sizeof(unsigned) * 2);
	//读取焦距和主点信息
	float intrinsicParams[4];
	fileHandle.read((char*)intrinsicParams, sizeof(float) * 4);
	//遍历读取每个相机的信息
	for (auto& eachCamera : inputData.cameras)
	{
		//读取宽高
		eachCamera.width = imgShape[0];
		eachCamera.height = imgShape[1];
		//记录主点和焦距
		eachCamera.fx = intrinsicParams[0];
		eachCamera.fy = intrinsicParams[1];
		eachCamera.cx = intrinsicParams[2];
		eachCamera.cy = intrinsicParams[3];
	}
	fileHandle.close();
}

//从文件流里面读取图片的路径
static std::string readString(std::fstream& fileHandle)
{
	//读取字符串的长度
	unsigned strLength;
	fileHandle.read((char*)&strLength,sizeof(unsigned));
	//打印字符串的长度
	DEBUG_LOG("string length: " << strLength);
	//读取字符串的内容
	std::string str(strLength, '\0');
	fileHandle.read(str.data(), strLength);
	return str;
}

//读取input data里面的外参
static void convertExternalParams(
	std::fstream& fileHandle,
	std::string imgRootPath, //图片的根目录
	InputData& inputData
)
{
	unsigned numImages;
	fileHandle.read((char*)&numImages, sizeof(unsigned));
	//打印image的个数
	DEBUG_LOG("numImages: "<<numImages);
	//初始化图片的个数
	inputData.cameras.resize(numImages);
	torch::Tensor unorientedPoses = torch::zeros({ static_cast<long int>(numImages), 4, 4 }, torch::kFloat32);

	//准备开始遍历

	for (size_t i = 0; i < numImages; i++) {
		DEBUG_LOG(i);
		//读取图片的id
		unsigned idImage;
		fileHandle.read((char*)&idImage,sizeof(unsigned));
		//当前位置的camera
		auto& currCamera = inputData.cameras.at(i);
		//记录图片的id
		currCamera.id = idImage;
		//读取四元数
		float q[4];
		fileHandle.read((char*)q, sizeof(float) * 4);


		torch::Tensor qVec = torch::tensor({
			q[0],q[1],q[2],q[3]
			}, torch::kFloat32);
		torch::Tensor R = quatToRotMat(qVec);


		//从colmap里面读取平移量
		float t[3];
		fileHandle.read((char*)t, sizeof(float) * 3);
		//然后再读取图片路径
		currCamera.filePath = imgRootPath + readString(fileHandle);
		torch::Tensor T = torch::tensor({
			{ t[0]},
			{ t[1]},
			{ t[2]}
		}, torch::kFloat32);

		torch::Tensor Rinv = R.transpose(0, 1);
		torch::Tensor Tinv = torch::matmul(-Rinv, T);


		unorientedPoses[i].index_put_({ Slice(None, 3), Slice(None, 3) }, Rinv);
		unorientedPoses[i].index_put_({ Slice(None, 3), Slice(3, 4) }, Tinv);
		unorientedPoses[i][3][3] = 1.0f;

		// Convert COLMAP's camera CRS (OpenCV) to OpenGL
		unorientedPoses[i].index_put_({ Slice(0, 3), Slice(1,3) }, unorientedPoses[i].index({ Slice(0, 3), Slice(1,3) }) * -1.0f);
	}

	DEBUG_LOG("end read all images");

	auto r = autoScaleAndCenterPoses(unorientedPoses);
	torch::Tensor poses = std::get<0>(r);
	inputData.translation = std::get<1>(r);
	inputData.scale = std::get<2>(r);

	for (size_t i = 0; i < inputData.cameras.size(); i++) {
		inputData.cameras[i].camToWorld = poses[i];
	}
}

//读取文件流里面的外参
static void readExternalParams(
	std::string filePath,
	std::string imgRootPath, //图片的根目录
	InputData& inputData
)
{
	//打开文件输入流
	std::fstream fileHandle;
	fileHandle.open(filePath, std::ios::in | std::ios::binary);
	//直接调用外参数据的转换过程
	convertExternalParams(fileHandle,imgRootPath, inputData);
	fileHandle.close();
}

//从文件中读取colmap的数据
static void makeInputDataFromFile(InputData& inputData)
{
	std::string pointPath = "E:/temp/point.bin";
	std::string imagePath = "E:/temp/image.bin";
	std::string cameraPath = "E:/temp/camera.bin";
	//图片的根目录
	std::string imgRootPath = "E:/temp/video/";
	//先读取inut里面的点数据
	DEBUG_LOG("loading points");
	loadPointInput(pointPath, inputData);
	//读取外参
	DEBUG_LOG("readExternalParams");
	readExternalParams(imagePath, imgRootPath, inputData);
	DEBUG_LOG("readIntrinsicParams");
	readIntrinsicParams(cameraPath, inputData);
	//给点做一下后处理，应用translation和缩放
	auto xyzPoint = inputData.points.xyz;
	inputData.points.xyz = (xyzPoint - inputData.translation) * inputData.scale;
}

//运行splat的函数
void runSplat()
{
	InputData inputData;
	//载入input数据
	makeInputDataFromFile(inputData);
	printPoints(inputData);
	//调用opensplat的实质性计算
	splatCompute(inputData);
}

int main()
{
	runSplat();
}
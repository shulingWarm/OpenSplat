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

//������ڲ���صĲ��������ǿ��Ա���Ϊ����������
static const size_t INTRINSIC_SIZE = 11;
//��ε���������С
static const size_t EXTERNAL_SIZE = 16;
//ÿ��camera���������������
static const size_t CAMERA_DATA_SIZE = (INTRINSIC_SIZE + EXTERNAL_SIZE);

//��ӡĳһ������������ڲ�
static void printCameraIntrinsic(Camera& camera)
{
	std::cout << "camera: " <<
		camera.fx << " " << camera.fy << " " <<
		camera.cx << " " << camera.cy << std::endl;
}

//��ӡ����image�����
static void printSingleExternal(Camera& camera)
{
	std::cout << "camera external: " << std::endl <<
		camera.camToWorld << std::endl;
}

//��ӡinput������������
static void printExternalParams(InputData& inputData)
{
	auto& images = inputData.cameras;
	//��ӡǰ����image
	for (unsigned idImage = 0; idImage < 2; ++idImage)
	{
		auto& image = images[idImage];
		printSingleExternal(image);
	}
}

//��ӡinput_data������ڲ�
static void printIntrinsic(InputData& inputData)
{
	auto& images = inputData.cameras;
	//��ӡ����data�ĸ���
	std::cout << "image num: " << images.size() << std::endl;
	//�����������
	for (int i = 0; i < 2; ++i)
	{
		//��ӡĳһ��������ڲ�
		printCameraIntrinsic(images[i]);
	}
}

//��ӡinput data����ĵ�����
static void printPoints(InputData& inputData)
{
	//����ǰ10����
	auto dataPtr = static_cast<uint8_t*>(inputData.points.rgb.data_ptr());
	for (int i = 0; i < 10; ++i)
	{
		std::cout << (int) dataPtr[i] << " ";
	}
	std::cout << std::endl;
}

//����һ�����������
static void loadSingleCamera(
	Camera& targetCamera,
	float* cameraData
)
{
	//ֱ�����δ��������ÿ����
	auto widthHead = (void*)(&targetCamera.width);
	memcpy(widthHead, cameraData, sizeof(float) * INTRINSIC_SIZE);
	//��ε�ͷָ��
	auto externalHead = cameraData + INTRINSIC_SIZE;
	//�����ݸ��Ƶ�camera����
	targetCamera.camToWorld = torch::from_blob(externalHead, { 4,4 }, torch::kFloat32);
}

//�����������Ϣ��Ȼ��ת����camera���б�
static void loadCameraData(
	std::vector<Camera>& cameraList,
	float* cameraData, //�������
	size_t cameraNum //�����Ϣ�ĸ���
)
{
	cameraList.resize(cameraNum);
	//����ÿ�����ܱ������camera
	for (size_t idCamera = 0; idCamera < cameraNum; ++idCamera)
	{
		//��ǰλ�õ�ͷָ��
		auto headPointer = cameraData + idCamera * CAMERA_DATA_SIZE;
		//��ǰλ�õ����
		auto& currCamera = cameraList[idCamera];
		//�õ�ǰ��ͷָ�빹��һ���������
		loadSingleCamera(currCamera, headPointer);
	}
}

//load���е���������
static void loadAllInput(
	InputData& inputData,
	//����ĸ���
	const size_t cameraNum,
	float* cameraData,
	//��ĸ���
	const size_t pointNum,
	float* pointData,
	float* colorData,
	std::vector<std::string>& imgPathList
)
{
	//��Ҫȷ��imgPathList��camera��������һ����
	if (cameraNum != imgPathList.size()) throw - 2;
	//�����������
	loadCameraData(inputData.cameras, cameraData, cameraNum);
	//��ʼ��scale����ֱ�����ó�1����
	inputData.scale = 1;
	//��ʼ��translation
	inputData.translation = torch::ones({ 3 });
	//��ʼ�������ݺ���ɫ����
	inputData.points.xyz = torch::from_blob(pointData, 
		{ static_cast<unsigned int>(pointNum),3}, torch::kFloat32);
	inputData.points.rgb = torch::from_blob(colorData,
		{ static_cast<unsigned int>(pointNum),3 }, torch::kFloat32);
	//ѭ����¼ÿ��camera��ͼƬ·�� 
	for (size_t idCamera = 0; idCamera < imgPathList.size(); ++idCamera)
	{
		auto& camera = inputData.cameras[idCamera];
		camera.filePath = imgPathList[idCamera];
	}
}

//ִ��3D gaissoam splatting��������
static void splatCompute(InputData inputData)
{
	parallel_for(inputData.cameras.begin(), inputData.cameras.end(), [](Camera& cam) {
		cam.loadImage(1);
	});

	// Withhold a validation camera if necessary
	auto t = inputData.getCameras(false, "random");
	std::vector<Camera> cams = std::get<0>(t);
	Camera* valCam = std::get<1>(t);

	//�ܵĵ�������
	const int numIters = 300;
	//��Ҫʹ�õ�device
	auto device = torch::kCUDA;
	//�ṹ���ƶȵ�Ȩ��ռ��
	float structureSimiliarWeight = 0.2f;

	Model model(inputData,
		cams.size(),
		2,//Ĭ�ϻὫͼƬ��Сһ�� �����������ر�
		3000, //��������ÿ����3000�ξͰ�ͼƬ�ķֱ��ʷ���һ��
		3, //��г��������������ĿǰҲ��֪�����Ǹ�ɶ�õ�
		1000, //����ÿ����1000�ξ�����һ����г������������
		100, //����ܲ��������Ƚ���Ҫ�Ĳ���,ÿ����100�ξ͸���һ��3DGS�Ĵ�С
		500,//ǰ500�ε�����ȫ�����Ƿָ�3DGS
		30,//ÿ30�θ�˹���Ⱦͻ�����е�3DGS��͸��������һ��
		4000,//λ���ݶȵ���ֵ��Ŀǰ��֪��������ʲô�õ�
		0.0002,//3DGS���и���ֵ refine��ʱ������������̫����и�һ��
		4000,//����������������Ͳ��ٿ����и�3DGS��
		0.05, //��һ�������и�3DGS��ָ�꣬������3DGSŪ����������0.05������Ļ��С���Ǳ���������
		numIters, //�����Ĵ���������������ҪƵ��������һ��������
		false, //�����ָ���Ǳ���ԭ������Դ�С���ƺ����Ǻ���Ҫ
		device //ʹ��cuda������ʼ��
	);

	std::vector< size_t > camIndices(cams.size());
	std::iota(camIndices.begin(), camIndices.end(), 0);
	InfiniteRandomIterator<size_t> camsIter(camIndices);

	//�ⶫ���ǵ���������ʹ�õ��ľֲ����ǣ���֪���Ǹ����õ�
	//��֮���ǳ�ʼ����10�������Ǳ�ʾչʾ�м���������
	int displayStep = 10;
	//���·������ʵ�Ժ���Ҫ�����������������Ҳ������Ϊ�˲���
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

//��ȡ���Ƶ�colmap����
static void loadPointInput(std::string filePath,
	InputData& inputData
)
{
	std::fstream fileHandle(filePath,std::ios::in|std::ios::binary);
	//��ȡ��ĸ���
	unsigned pointNum = 0;
	fileHandle.read((char*)&pointNum,sizeof(unsigned));
	DEBUG_LOG("pointNum: " << pointNum);
	//������ȡÿ����
	float* pointData = new float[pointNum * 3];
	uint8_t* colorData = new uint8_t[pointNum * 3];
	//������ȡÿ��λ�õ���ɫ
	for (unsigned idPoint = 0; idPoint < pointNum; ++idPoint)
	{
		auto xyzHead = pointData + idPoint * 3;
		auto colorHead = colorData + idPoint * 3;
		//��ȡ������
		fileHandle.read((char*)xyzHead, sizeof(float) * 3);
		fileHandle.read((char*)colorHead, sizeof(uint8_t) * 3);
	}
	fileHandle.close();
	//�õ����ݺ���ɫ��������blob
	inputData.points.xyz = torch::from_blob(pointData, { pointNum,3 }, torch::kFloat32).clone();
	inputData.points.rgb = torch::from_blob(colorData, { pointNum,3 }, torch::kU8).clone();
}

//��ȡinput���ڲβ���
static void readIntrinsicParams(
	std::string filePath,
	InputData& inputData
)
{
	std::fstream fileHandle;
	fileHandle.open(filePath,std::ios::in|std::ios::binary);
	//��ȡ����ĸ���
	unsigned cameraNum;
	fileHandle.read((char*)&cameraNum, sizeof(unsigned));
	//ȷ���ڲ�ֻ��һ��
	MY_ASSERT(cameraNum == 1);
	//��ȡͼƬ��shape
	unsigned imgShape[2];
	fileHandle.read((char*)imgShape, sizeof(unsigned) * 2);
	//��ȡ�����������Ϣ
	float intrinsicParams[4];
	fileHandle.read((char*)intrinsicParams, sizeof(float) * 4);
	//������ȡÿ���������Ϣ
	for (auto& eachCamera : inputData.cameras)
	{
		//��ȡ���
		eachCamera.width = imgShape[0];
		eachCamera.height = imgShape[1];
		//��¼����ͽ���
		eachCamera.fx = intrinsicParams[0];
		eachCamera.fy = intrinsicParams[1];
		eachCamera.cx = intrinsicParams[2];
		eachCamera.cy = intrinsicParams[3];
	}
	fileHandle.close();
}

//���ļ��������ȡͼƬ��·��
static std::string readString(std::fstream& fileHandle)
{
	//��ȡ�ַ����ĳ���
	unsigned strLength;
	fileHandle.read((char*)&strLength,sizeof(unsigned));
	//��ӡ�ַ����ĳ���
	DEBUG_LOG("string length: " << strLength);
	//��ȡ�ַ���������
	std::string str(strLength, '\0');
	fileHandle.read(str.data(), strLength);
	return str;
}

//��ȡinput data��������
static void convertExternalParams(
	std::fstream& fileHandle,
	std::string imgRootPath, //ͼƬ�ĸ�Ŀ¼
	InputData& inputData
)
{
	unsigned numImages;
	fileHandle.read((char*)&numImages, sizeof(unsigned));
	//��ӡimage�ĸ���
	DEBUG_LOG("numImages: "<<numImages);
	//��ʼ��ͼƬ�ĸ���
	inputData.cameras.resize(numImages);
	torch::Tensor unorientedPoses = torch::zeros({ static_cast<long int>(numImages), 4, 4 }, torch::kFloat32);

	//׼����ʼ����

	for (size_t i = 0; i < numImages; i++) {
		DEBUG_LOG(i);
		//��ȡͼƬ��id
		unsigned idImage;
		fileHandle.read((char*)&idImage,sizeof(unsigned));
		//��ǰλ�õ�camera
		auto& currCamera = inputData.cameras.at(i);
		//��¼ͼƬ��id
		currCamera.id = idImage;
		//��ȡ��Ԫ��
		float q[4];
		fileHandle.read((char*)q, sizeof(float) * 4);


		torch::Tensor qVec = torch::tensor({
			q[0],q[1],q[2],q[3]
			}, torch::kFloat32);
		torch::Tensor R = quatToRotMat(qVec);


		//��colmap�����ȡƽ����
		float t[3];
		fileHandle.read((char*)t, sizeof(float) * 3);
		//Ȼ���ٶ�ȡͼƬ·��
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

//��ȡ�ļ�����������
static void readExternalParams(
	std::string filePath,
	std::string imgRootPath, //ͼƬ�ĸ�Ŀ¼
	InputData& inputData
)
{
	//���ļ�������
	std::fstream fileHandle;
	fileHandle.open(filePath, std::ios::in | std::ios::binary);
	//ֱ�ӵ���������ݵ�ת������
	convertExternalParams(fileHandle,imgRootPath, inputData);
	fileHandle.close();
}

//���ļ��ж�ȡcolmap������
static void makeInputDataFromFile(InputData& inputData)
{
	std::string pointPath = "E:/temp/point.bin";
	std::string imagePath = "E:/temp/image.bin";
	std::string cameraPath = "E:/temp/camera.bin";
	//ͼƬ�ĸ�Ŀ¼
	std::string imgRootPath = "E:/temp/video/";
	//�ȶ�ȡinut����ĵ�����
	DEBUG_LOG("loading points");
	loadPointInput(pointPath, inputData);
	//��ȡ���
	DEBUG_LOG("readExternalParams");
	readExternalParams(imagePath, imgRootPath, inputData);
	DEBUG_LOG("readIntrinsicParams");
	readIntrinsicParams(cameraPath, inputData);
	//������һ�º���Ӧ��translation������
	auto xyzPoint = inputData.points.xyz;
	inputData.points.xyz = (xyzPoint - inputData.translation) * inputData.scale;
}

//����splat�ĺ���
void runSplat()
{
	InputData inputData;
	//����input����
	makeInputDataFromFile(inputData);
	printPoints(inputData);
	//����opensplat��ʵ���Լ���
	splatCompute(inputData);
}

int main()
{
	runSplat();
}
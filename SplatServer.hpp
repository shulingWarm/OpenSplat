#pragma once
#include"ColmapServer/SparseScene.hpp"
#include<string>
#include<iostream>
#include <filesystem>
#include"input_data.hpp"
#include<vector>
#include<string>
#include"utils.hpp"
#include"model.hpp"
#include"tensor_math.hpp"
#include"Debug.hpp"
#include"ModelFunction.hpp"

void convertToSplatInput(SparseScene& sparseScene,
	InputData& inputData
)
{
	//点的数量
	uint32_t pointNum = sparseScene.xyz.size() / 3;
	std::cout << "point num: " << pointNum << std::endl;
	//载入点数据
	inputData.points.xyz = torch::from_blob(sparseScene.xyz.data(), { pointNum,3 }, torch::kFloat32).clone();
	//载入颜色数据
	inputData.points.rgb = torch::from_blob(sparseScene.rgb.data(), { pointNum,3 }, torch::kU8).clone();
	std::cout << "inputData.cameras.resize(sparseScene.cameraList.size());" << std::endl;
	//初始化camera的数量
	inputData.cameras.resize(sparseScene.cameraList.size());
	//opengl形式的外参表示方法，虽然不知道它具体是怎样做的
	torch::Tensor unorientedPoses = torch::zeros({ static_cast<long int>(
		sparseScene.cameraList.size()), 4, 4 }, torch::kFloat32);
	//遍历每个camera
	for (size_t idCamera = 0; idCamera < sparseScene.cameraList.size(); ++idCamera)
	{
		std::cout << idCamera << std::endl;
		auto& dstCamera = inputData.cameras[idCamera];
		auto& srcCamera = sparseScene.cameraList[idCamera];
		//记录图片路径
		dstCamera.filePath = srcCamera.imgPath;
		//把四元数变成torch tensor
		torch::Tensor qVec = torch::tensor({
			srcCamera.rotation[0],
			srcCamera.rotation[1],
			srcCamera.rotation[2],
			srcCamera.rotation[3]
			}, torch::kFloat32);
		//把四元数转换成旋转矩阵
		torch::Tensor R = quatToRotMat(qVec);
		//把平移量转换成tensor
		torch::Tensor T = torch::tensor({
			{srcCamera.translation[0]},
			{srcCamera.translation[1]},
			{srcCamera.translation[2]}
			}, torch::kFloat32);
		//旋转矩阵的逆矩阵
		torch::Tensor Rinv = R.transpose(0, 1);
		torch::Tensor Tinv = torch::matmul(-Rinv, T);
		

		unorientedPoses[idCamera].index_put_({ Slice(None, 3), Slice(None, 3) }, Rinv);
		unorientedPoses[idCamera].index_put_({ Slice(None, 3), Slice(3, 4) }, Tinv);
		unorientedPoses[idCamera][3][3] = 1.0f;

		// Convert COLMAP's camera CRS (OpenCV) to OpenGL
		unorientedPoses[idCamera].index_put_({ Slice(0, 3), Slice(1,3) }, unorientedPoses[idCamera].index({ Slice(0, 3), Slice(1,3) }) * -1.0f);
	}

	//记录外参数据
	auto r = autoScaleAndCenterPoses(unorientedPoses);
	torch::Tensor poses = std::get<0>(r);
	inputData.translation = std::get<1>(r);
	inputData.scale = std::get<2>(r);

	for (size_t i = 0; i < inputData.cameras.size(); i++) {
		inputData.cameras[i].camToWorld = poses[i];
	}

	//遍历每个相机
	for (auto& eachCamera : inputData.cameras)
	{
		//记录宽高
		eachCamera.width = sparseScene.imgShape[0];
		eachCamera.height = sparseScene.imgShape[1];
		//记录主点和焦距
		eachCamera.fx = sparseScene.intrinsic[0];
		eachCamera.fy = sparseScene.intrinsic[1];
		eachCamera.cx = sparseScene.intrinsic[2];
		eachCamera.cy = sparseScene.intrinsic[3];
	}

	//给点做一下后处理，应用translation和缩放
	auto xyzPoint = inputData.points.xyz;
	inputData.points.xyz = (xyzPoint - inputData.translation) * inputData.scale;
}

//执行3D gaissoam splatting的主流程
static void splatCompute(InputData inputData, std::string outputPath,Model*& retModel)
{
	parallel_for(inputData.cameras.begin(), inputData.cameras.end(), [](Camera& cam) {
		cam.loadImage(1);
		});

	// Withhold a validation camera if necessary
	auto t = inputData.getCameras(false, "random");
	std::vector<Camera> cams = std::get<0>(t);
	Camera* valCam = std::get<1>(t);

	//总的迭代次数
	const int numIters = 500;
	//主要使用的device
	auto device = torch::kCUDA;
	//结构相似度的权重占比
	float structureSimiliarWeight = 0.2f;

	//用于返回结果的model
	retModel = new Model(inputData,
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

	auto& model = *retModel;

	std::vector< size_t > camIndices(cams.size());
	std::iota(camIndices.begin(), camIndices.end(), 0);
	InfiniteRandomIterator<size_t> camsIter(camIndices);

	//这东西是迭代过程中使用到的局部就是，不知道是干嘛用的
	//总之就是初始化成10，可能是表示展示中间结果的周期
	int displayStep = 10;

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

	model.save(outputPath);
}

//取出InputData里面的光心坐标
static void getCameraCenterFromInputData(
	InputData& inputData,
	float* camCenter
)
{
	//遍历input data里面的每个camera
	for (int idCamera = 0; idCamera < inputData.cameras.size(); ++idCamera)
	{

	}
}

//测试torch slice的效果
void testSlice()
{
	//先随便新建一个torch tensor
	torch::Tensor testTensor = torch::zeros({ 4,4 });
	//再新建一个切片用于表示旋转
	auto rotTensor = torch::randn({ 3,3 });
	auto transTensor = torch::randn({ 3,1 });
	std::cout << "Rotation" << std::endl;
	std::cout << rotTensor << std::endl;
	std::cout << "Translation" << std::endl;
	std::cout << transTensor << std::endl;
	//打印原始的tensor
	std::cout << "original tensor" << std::endl;
	std::cout << testTensor << std::endl;
	testTensor.index_put_({ Slice(None, 3), Slice(None, 3) }, rotTensor);
	std::cout << "Apply rot tensor" << std::endl;
	std::cout << testTensor << std::endl;
	testTensor.index_put_({ Slice(None, 3), Slice(3, 4) }, transTensor);
	std::cout << "Apply transpose tensor" << std::endl;
	std::cout << testTensor << std::endl;
	//然后执行最后一步的oprngl形式的处理
	testTensor.index_put_({ Slice(0, 3), Slice(1,3) }, testTensor.index({ Slice(0, 3), Slice(1,3) }) * (-1.0f));
	std::cout << "Final index put" << std::endl;
	std::cout << testTensor << std::endl;
}

//运行splat server的过程，弄完之后直接把结果保存成ply就可以了
void splatServer(SparseScene& sparseScene,
	std::string dstFile,Model*& retModel,
	float** camCenter = nullptr
) {
	std::cout << "running splatServer" << std::endl;
	//把sparse scene转换成colmap
	InputData inputData;
	convertToSplatInput(sparseScene, inputData);
	std::cout << "begin splatCompute" << std::endl;
	//调用splat过程的核心计算
	splatCompute(inputData, dstFile,retModel);
}
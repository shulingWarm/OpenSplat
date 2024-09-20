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
	//�������
	uint32_t pointNum = sparseScene.xyz.size() / 3;
	std::cout << "point num: " << pointNum << std::endl;
	//���������
	inputData.points.xyz = torch::from_blob(sparseScene.xyz.data(), { pointNum,3 }, torch::kFloat32).clone();
	//������ɫ����
	inputData.points.rgb = torch::from_blob(sparseScene.rgb.data(), { pointNum,3 }, torch::kU8).clone();
	std::cout << "inputData.cameras.resize(sparseScene.cameraList.size());" << std::endl;
	//��ʼ��camera������
	inputData.cameras.resize(sparseScene.cameraList.size());
	//opengl��ʽ����α�ʾ��������Ȼ��֪������������������
	torch::Tensor unorientedPoses = torch::zeros({ static_cast<long int>(
		sparseScene.cameraList.size()), 4, 4 }, torch::kFloat32);
	//����ÿ��camera
	for (size_t idCamera = 0; idCamera < sparseScene.cameraList.size(); ++idCamera)
	{
		std::cout << idCamera << std::endl;
		auto& dstCamera = inputData.cameras[idCamera];
		auto& srcCamera = sparseScene.cameraList[idCamera];
		//��¼ͼƬ·��
		dstCamera.filePath = srcCamera.imgPath;
		//����Ԫ�����torch tensor
		torch::Tensor qVec = torch::tensor({
			srcCamera.rotation[0],
			srcCamera.rotation[1],
			srcCamera.rotation[2],
			srcCamera.rotation[3]
			}, torch::kFloat32);
		//����Ԫ��ת������ת����
		torch::Tensor R = quatToRotMat(qVec);
		//��ƽ����ת����tensor
		torch::Tensor T = torch::tensor({
			{srcCamera.translation[0]},
			{srcCamera.translation[1]},
			{srcCamera.translation[2]}
			}, torch::kFloat32);
		//��ת����������
		torch::Tensor Rinv = R.transpose(0, 1);
		torch::Tensor Tinv = torch::matmul(-Rinv, T);
		

		unorientedPoses[idCamera].index_put_({ Slice(None, 3), Slice(None, 3) }, Rinv);
		unorientedPoses[idCamera].index_put_({ Slice(None, 3), Slice(3, 4) }, Tinv);
		unorientedPoses[idCamera][3][3] = 1.0f;

		// Convert COLMAP's camera CRS (OpenCV) to OpenGL
		unorientedPoses[idCamera].index_put_({ Slice(0, 3), Slice(1,3) }, unorientedPoses[idCamera].index({ Slice(0, 3), Slice(1,3) }) * -1.0f);
	}

	//��¼�������
	auto r = autoScaleAndCenterPoses(unorientedPoses);
	torch::Tensor poses = std::get<0>(r);
	inputData.translation = std::get<1>(r);
	inputData.scale = std::get<2>(r);

	for (size_t i = 0; i < inputData.cameras.size(); i++) {
		inputData.cameras[i].camToWorld = poses[i];
	}

	//����ÿ�����
	for (auto& eachCamera : inputData.cameras)
	{
		//��¼���
		eachCamera.width = sparseScene.imgShape[0];
		eachCamera.height = sparseScene.imgShape[1];
		//��¼����ͽ���
		eachCamera.fx = sparseScene.intrinsic[0];
		eachCamera.fy = sparseScene.intrinsic[1];
		eachCamera.cx = sparseScene.intrinsic[2];
		eachCamera.cy = sparseScene.intrinsic[3];
	}

	//������һ�º���Ӧ��translation������
	auto xyzPoint = inputData.points.xyz;
	inputData.points.xyz = (xyzPoint - inputData.translation) * inputData.scale;
}

//ִ��3D gaissoam splatting��������
static void splatCompute(InputData inputData, std::string outputPath,Model*& retModel)
{
	parallel_for(inputData.cameras.begin(), inputData.cameras.end(), [](Camera& cam) {
		cam.loadImage(1);
		});

	// Withhold a validation camera if necessary
	auto t = inputData.getCameras(false, "random");
	std::vector<Camera> cams = std::get<0>(t);
	Camera* valCam = std::get<1>(t);

	//�ܵĵ�������
	const int numIters = 500;
	//��Ҫʹ�õ�device
	auto device = torch::kCUDA;
	//�ṹ���ƶȵ�Ȩ��ռ��
	float structureSimiliarWeight = 0.2f;

	//���ڷ��ؽ����model
	retModel = new Model(inputData,
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

	auto& model = *retModel;

	std::vector< size_t > camIndices(cams.size());
	std::iota(camIndices.begin(), camIndices.end(), 0);
	InfiniteRandomIterator<size_t> camsIter(camIndices);

	//�ⶫ���ǵ���������ʹ�õ��ľֲ����ǣ���֪���Ǹ����õ�
	//��֮���ǳ�ʼ����10�������Ǳ�ʾչʾ�м���������
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

//ȡ��InputData����Ĺ�������
static void getCameraCenterFromInputData(
	InputData& inputData,
	float* camCenter
)
{
	//����input data�����ÿ��camera
	for (int idCamera = 0; idCamera < inputData.cameras.size(); ++idCamera)
	{

	}
}

//����torch slice��Ч��
void testSlice()
{
	//������½�һ��torch tensor
	torch::Tensor testTensor = torch::zeros({ 4,4 });
	//���½�һ����Ƭ���ڱ�ʾ��ת
	auto rotTensor = torch::randn({ 3,3 });
	auto transTensor = torch::randn({ 3,1 });
	std::cout << "Rotation" << std::endl;
	std::cout << rotTensor << std::endl;
	std::cout << "Translation" << std::endl;
	std::cout << transTensor << std::endl;
	//��ӡԭʼ��tensor
	std::cout << "original tensor" << std::endl;
	std::cout << testTensor << std::endl;
	testTensor.index_put_({ Slice(None, 3), Slice(None, 3) }, rotTensor);
	std::cout << "Apply rot tensor" << std::endl;
	std::cout << testTensor << std::endl;
	testTensor.index_put_({ Slice(None, 3), Slice(3, 4) }, transTensor);
	std::cout << "Apply transpose tensor" << std::endl;
	std::cout << testTensor << std::endl;
	//Ȼ��ִ�����һ����oprngl��ʽ�Ĵ���
	testTensor.index_put_({ Slice(0, 3), Slice(1,3) }, testTensor.index({ Slice(0, 3), Slice(1,3) }) * (-1.0f));
	std::cout << "Final index put" << std::endl;
	std::cout << testTensor << std::endl;
}

//����splat server�Ĺ��̣�Ū��֮��ֱ�Ӱѽ�������ply�Ϳ�����
void splatServer(SparseScene& sparseScene,
	std::string dstFile,Model*& retModel,
	float** camCenter = nullptr
) {
	std::cout << "running splatServer" << std::endl;
	//��sparse sceneת����colmap
	InputData inputData;
	convertToSplatInput(sparseScene, inputData);
	std::cout << "begin splatCompute" << std::endl;
	//����splat���̵ĺ��ļ���
	splatCompute(inputData, dstFile,retModel);
}
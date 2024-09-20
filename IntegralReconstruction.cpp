#include<string>
#include<iostream>
#include"ColmapServer/ColmapServer.hpp"
#include"SplatServer.hpp"
#include"IntegralReconstruction.hpp"
#include"ModelFunction.hpp"
#include"GaussianViewAngle/GaussianViewAngle.h"

//把颜色信息保存到SplatScene里面
static void loadColorToSplatScene(Model& model, SplatScene& splatScene)
{
	//把颜色数据转换到cpu
	auto cpuColorTensor = model.featuresDc.cpu();
	//给颜色数据开辟空间
	splatScene.color = new float[3 * splatScene.pointNum];
	//遍历所有的点 记录它们的颜色
	for (uint32_t idPoint = 0; idPoint < splatScene.pointNum; ++idPoint)
	{
		auto dstHead = splatScene.color + idPoint * 3;
		auto srcHead = reinterpret_cast<float*>(cpuColorTensor[idPoint].data_ptr());
		memcpy(dstHead, srcHead, sizeof(float) * 3);
	}
}

//把SplatScene里面的scale存储进去
static void loadScaleToSplatScene(Model& model, SplatScene& splatScene)
{
	//把scale数据转换到cpu
	auto cpuScale = model.scales.cpu();
	//给scale开辟空间
	splatScene.scale = new float[3 * splatScene.pointNum];
	//遍历所有的点，记录它们的scale
	for (uint32_t idPoint = 0; idPoint < splatScene.pointNum; ++idPoint)
	{
		auto dstHead = splatScene.scale + idPoint * 3;
		auto srcHead = reinterpret_cast<float*>(cpuScale[idPoint].data_ptr());
		memcpy(dstHead, srcHead, sizeof(float) * 3);
	}
}

//存储SplatScene里面的旋转信息
static void loadRotationToSplatScene(Model& model, SplatScene& splatScene)
{
	//把旋转数据转换到cpu
	auto cpuRotation = model.quats.cpu();
	//给旋转数据开空间
	splatScene.rotation = new float[splatScene.pointNum * 4];
	//遍历所有的点，记录旋转信息
	for (uint32_t idPoint = 0; idPoint < splatScene.pointNum; ++idPoint)
	{
		auto dstHead = splatScene.rotation + idPoint * 4;
		auto srcHead = reinterpret_cast<float*>(cpuRotation[idPoint].data_ptr());
		memcpy(dstHead, srcHead, sizeof(float) * 4);
	}
}

//存储透明度信息
static void loadOpacityToSplatScene(Model& model, SplatScene& splatScene)
{
	//把透明度转换到cpu
	auto cpuOpacity = model.opacities.cpu();
	//开辟透明度的内容
	splatScene.opacity = new float[splatScene.pointNum];
	//遍历每个点
	for (uint32_t idPoint = 0; idPoint < splatScene.pointNum; ++idPoint)
	{
		auto dstHead = splatScene.opacity + idPoint;
		auto srcHead = reinterpret_cast<float*>(cpuOpacity[idPoint].data_ptr());
		memcpy(dstHead, srcHead, sizeof(float));
	}
}

//把model里面的数据存储成C格式
static void convertToSplatScene(Model& model,SplatScene& splatScene)
{
	//获取点坐标的tensor序列
	auto pointTensor = getCpuPointList(model);
	//记录里面的点个数
	splatScene.pointNum = getPointNum(model);
	//给点坐标开辟数据空间
	splatScene.pointList = new float[splatScene.pointNum * 3];
	//依次记录每个点的坐标
	for (int idPoint = 0; idPoint < splatScene.pointNum; ++idPoint)
	{
		auto dstHead = splatScene.pointList + idPoint * 3;
		auto srcHead = reinterpret_cast<float*>(pointTensor[idPoint].data_ptr());
		memcpy(dstHead, srcHead, sizeof(float) * 3);
	}
	//把颜色数据保存到SplatScene里面
	loadColorToSplatScene(model, splatScene);
	//存储SplatScene里面的scale
	loadScaleToSplatScene(model, splatScene);
	//存储SplatScene里面的旋转信息
	loadRotationToSplatScene(model, splatScene);
	//存储透明度信息model
	loadOpacityToSplatScene(model, splatScene);
}

//ply格式下的属性
class PlyProperty
{
public:
	std::string type;
	std::string name;
};

//ply的文件格式信息
class PlyHeader
{
public:
	std::string format;
	//节点的个数
	int verticeNum = 0;
	//面片的个数
	int faceNum = 0;
	//所有的属性列表，包括节点的和面片的
	std::vector<PlyProperty> propertyList;
};

//一个节点的数据
class Vertex
{
public:
	//位置
	float position[3];
	//法向量
	float normal[3];
	//球谐函数的参数
	float shs[48];
	//透明度
	float opacity;
	//三个方向的scale
	float scale[3];
	//由四元数构成的旋转
	float rotation[4];
};


//载入点云的数据头
static void loadPlyHeader(std::ifstream& fileHandle,
	PlyHeader& header
)
{
	//中间读取到的每一行的结果
	std::string line;
	while (std::getline(fileHandle, line))
	{
		//新建输入流
		std::istringstream tempStream(line);
		//读取一个单词
		std::string token;
		tempStream >> token;
		//判断是不是格式信息
		if (token == "format")
		{
			tempStream >> header.format;
		}
		//判断读取到的是不是element信息
		else if (token == "element")
		{
			//再读取element的类型
			tempStream >> token;
			//判断是节点个数还是面片的个数
			if (token == "vertex")
			{
				tempStream >> header.verticeNum;
			}
			else if (token == "face")
			{
				tempStream >> header.faceNum;
			}
			else
			{
				throw std::runtime_error("Unknown element type");
			}
		}
		//再判断是否读取到了属性信息
		else if (token == "property")
		{
			//新建一个临时的属性
			PlyProperty tempProperty;
			//记录它的type和名字
			tempStream >> tempProperty.type >> tempProperty.name;
			//把属性放到列表里面
			header.propertyList.push_back(tempProperty);
		}
		//header部分的结束符
		else if (token == "end_header")
		{
			break;
		}

	}
}

//直接强行载入splat scene
static void loadSplatScene(SplatScene& splatScene)
{
	//点云所在的文件
	std::string filePath = "E:/temp/test.ply";
	//打开文件的输入流
	std::ifstream fileHandle(filePath, std::ios::binary);
	//新建ply的头部描述符
	PlyHeader header;
	loadPlyHeader(fileHandle, header);
	//用于读取缓冲区的数据大小，目前暂时不考虑扩展性
	const int BUFFER_SIZE = 62;
	//用于读取数据的缓冲区
	float pointBuffer[BUFFER_SIZE];
	//给splat scene里面的内容开辟好空间
	splatScene.pointList = new float[header.verticeNum * 3];
	splatScene.color = new float[header.verticeNum * 3];
	splatScene.scale = new float[header.verticeNum * 3];
	splatScene.rotation = new float[header.verticeNum * 4];
	splatScene.opacity = new float[header.verticeNum];
	splatScene.pointNum = header.verticeNum;
	//遍历读取每个点
	for (int idPoint = 0; idPoint < header.verticeNum; ++idPoint)
	{
		//当前要操作的每个数据的数据头
		auto posHead = splatScene.pointList + idPoint * 3;
		auto colorHead = splatScene.color + idPoint * 3;
		auto sizeHead = splatScene.scale + idPoint * 3;
		auto rotHead = splatScene.rotation + idPoint * 4;
		auto opacityHead = splatScene.opacity + idPoint;
		//读取当前点的所有数据信息
		fileHandle.read((char*)pointBuffer, sizeof(float) * BUFFER_SIZE);
		//读取点数据
		memcpy(posHead, pointBuffer, 3 * sizeof(float));
		memcpy(colorHead, pointBuffer + 6, 3 * sizeof(float));
		//读取透明度信息
		opacityHead[0] = pointBuffer[54];
		//读取scale信息
		memcpy(sizeHead, pointBuffer + 55, 3 * sizeof(float));
		//读取旋转数据
		memcpy(rotHead, pointBuffer + 58, 4 * sizeof(float));
	}
	fileHandle.close();
}

//从model里面获取gaussian点的列表
static float* getGaussianList(Model& model)
{
	return reinterpret_cast<float*>(model.means.data_ptr());
}

//获取一个四元数的逆旋转
static void invertQuanternions(float* quanternions,float* dstInvertQuant)
{
	//记录w
	dstInvertQuant[0] = quanternions[0];
	for (int i = 1; i < 4; ++i)
		dstInvertQuant[i] = -quanternions[i];
}

//把四元数转换成旋转矩阵
static void convertQuanternionsToRotMat(float* q, float* rotMat)
{
	rotMat[0] = 1 - 2 * (q[2] * q[2] + q[3] * q[3]);
	rotMat[1] = 2 * (q[1] * q[2] - q[0] * q[3]);
	rotMat[2] = 2 * (q[1] * q[3] + q[0] * q[2]);
	rotMat[3] = 2 * (q[1] * q[2] + q[0] * q[3]);
	rotMat[4] = 1 - 2 * (q[1] * q[1] + q[3] * q[3]);
	rotMat[5] = 2 * (q[2] * q[3] - q[0] * q[1]);
	rotMat[6] = 2 * (q[1] * q[3] - q[0] * q[2]);
	rotMat[7] = 2 * (q[2] * q[3] + q[0] * q[1]);
	rotMat[8] = 1 - 2 * (q[1] * q[1] + q[2] * q[2]);
}

//对点执行旋转操作
static void applyRotation(float* rotation,float* point,float* dstPoint)
{
	for (int idRow = 0; idRow < 3; ++idRow)
	{
		dstPoint[idRow] = 0;
		for (int i = 0; i < 3; ++i)
		{
			dstPoint[idRow] += rotation[idRow * 3 + i] * point[i];
		}
	}
}

//从input里面获取camera光心的列表
static void getCamCenterList(std::vector<float>& dstCenterList,
	SparseScene& scene
)
{
	//scene里面的相机列表
	auto& cameraList = scene.cameraList;
	//遍历读取每个相机 注意这里后面需要用多线程加速一下
	dstCenterList.resize(cameraList.size() * 3);
	for (uint32_t i = 0; i < cameraList.size(); ++i)
	{
		auto& tempCamera = cameraList[i];
		//获取这个四元数的逆旋转
		float invertQuant[4];
		invertQuanternions(tempCamera.rotation, invertQuant);
		//把四元数转换成旋转矩阵
		float rotMat[9];
		convertQuanternionsToRotMat(invertQuant, rotMat);
		//对一个3D点做旋转
		applyRotation(rotMat, tempCamera.translation, &dstCenterList[i * 3]);
	}
}

//把3D坐标保存成ply文件，这里仅仅是做最简单的点保存
static void savePlyFile(float* pointData,uint32_t pointNum,std::string filePath)
{
	std::fstream fileHandle(filePath, std::ios::out | std::ios::binary);
	if (!fileHandle.is_open())
	{
		std::cerr << "Cannot open " << filePath << std::endl;
		return;
	}
	//写入ply的数据头
	fileHandle << "ply\n";
	fileHandle << "format binary_little_endian 1.0\n"; // 注意这里指定了字节序
	fileHandle << "element vertex " << pointNum << "\n";
	fileHandle << "property float x\n";
	fileHandle << "property float y\n";
	fileHandle << "property float z\n";
	fileHandle << "end_header\n";
	//写入点的二进制数据
	fileHandle.write((char*)pointData, sizeof(float) * 3 * pointNum);
	fileHandle.close();
}

//交换一个点列表的y,z
static void exchangeYZ(float* pointData,uint32_t pointNum)
{
	//遍历所有的点数据
	for (int idPoint = 0; idPoint < pointNum; ++idPoint)
	{
		auto pointHead = pointData + idPoint * 3;
		auto y = pointHead[1];
		auto z = pointHead[2];
		pointHead[1] = -z;
		pointHead[2] = -y;
	}
}

//保存3D高斯的视角范围
static void saveGaussianViewRange(
	float* viewRange,
	uint32_t gaussianNum,
	std::string filePath
)
{
	std::fstream fileHandle(filePath, std::ios::out | std::ios::binary);
	//写入gaussian的个数
	fileHandle.write((char*)&gaussianNum, sizeof(uint32_t));
	//保存这里面真正的点个数
	fileHandle.write((char*)viewRange, sizeof(float) * gaussianNum * 4);
	fileHandle.close();
}

extern "C" __declspec(dllexport) void reconstruct(const char* imgPathStr,
	const char* workspaceStr,void* dstScene)
{
	testSlice();
	return;
	//把数据内容转换成SplatScene
	auto splatScene = reinterpret_cast<SplatScene*>(dstScene);
	//直接从文件里面读取点云
	//loadSplatScene(*splatScene);
	//return;
	//初始化sparse scene，用于获取colmap的结果
	SparseScene sparseScene;
	auto imgPath = std::string(imgPathStr);
	auto workspace = std::string(workspaceStr);
	callColmap(imgPath, workspace, sparseScene);

	std::cout << "point num: "
		<< sparseScene.xyz.size() << std::endl;

	//遍历每个colmap的图片路径，把里面的图片路径变成绝对路径
	for (auto& eachCamera : sparseScene.cameraList)
	{
		eachCamera.imgPath = imgPath + "/" +
			eachCamera.imgPath;
	}

	Model* retModel;
	//调用splat的计算
	splatServer(sparseScene, "E:/temp/splat.ply", retModel);
	//获取center的列表
	std::vector<float> camCenterList;
	getCamCenterList(camCenterList, sparseScene);
	//把相机光心的列表保存成点云
	savePlyFile(camCenterList.data(), camCenterList.size() / 3, "E:/temp/camera.ply");
	//置换相机光心里面的y,z坐标，这是为了和后面的gaussian的预处理阶段对齐
	exchangeYZ(camCenterList.data(), camCenterList.size() / 3);
	auto gaussianNum = getPointNum(*retModel);
	std::vector<float> viewAngle(gaussianNum * 4);
	getGaussianViewAngle(getGaussianList(*retModel), camCenterList.data(),
		viewAngle.data(), camCenterList.size() / 3,
		gaussianNum);
	//保存每个 3D gaussian的视角范围
	saveGaussianViewRange(viewAngle.data(), gaussianNum,
		"E:/temp/viewRange.bin");
	//调用模型的转换
	convertToSplatScene(*retModel, *splatScene);
}
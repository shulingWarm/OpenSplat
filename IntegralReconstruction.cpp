#include<string>
#include<iostream>
#include"ColmapServer/ColmapServer.hpp"
#include"SplatServer.hpp"
#include"IntegralReconstruction.hpp"
#include"ModelFunction.hpp"

//����ɫ��Ϣ���浽SplatScene����
static void loadColorToSplatScene(Model& model, SplatScene& splatScene)
{
	//����ɫ����ת����cpu
	auto cpuColorTensor = model.featuresDc.cpu();
	//����ɫ���ݿ��ٿռ�
	splatScene.color = new float[3 * splatScene.pointNum];
	//�������еĵ� ��¼���ǵ���ɫ
	for (uint32_t idPoint = 0; idPoint < splatScene.pointNum; ++idPoint)
	{
		auto dstHead = splatScene.color + idPoint * 3;
		auto srcHead = reinterpret_cast<float*>(cpuColorTensor[idPoint].data_ptr());
		memcpy(dstHead, srcHead, sizeof(float) * 3);
	}
}

//��SplatScene�����scale�洢��ȥ
static void loadScaleToSplatScene(Model& model, SplatScene& splatScene)
{
	//��scale����ת����cpu
	auto cpuScale = model.scales.cpu();
	//��scale���ٿռ�
	splatScene.scale = new float[3 * splatScene.pointNum];
	//�������еĵ㣬��¼���ǵ�scale
	for (uint32_t idPoint = 0; idPoint < splatScene.pointNum; ++idPoint)
	{
		auto dstHead = splatScene.scale + idPoint * 3;
		auto srcHead = reinterpret_cast<float*>(cpuScale[idPoint].data_ptr());
		memcpy(dstHead, srcHead, sizeof(float) * 3);
	}
}

//�洢SplatScene�������ת��Ϣ
static void loadRotationToSplatScene(Model& model, SplatScene& splatScene)
{
	//����ת����ת����cpu
	auto cpuRotation = model.quats.cpu();
	//����ת���ݿ��ռ�
	splatScene.rotation = new float[splatScene.pointNum * 4];
	//�������еĵ㣬��¼��ת��Ϣ
	for (uint32_t idPoint = 0; idPoint < splatScene.pointNum; ++idPoint)
	{
		auto dstHead = splatScene.rotation + idPoint * 4;
		auto srcHead = reinterpret_cast<float*>(cpuRotation[idPoint].data_ptr());
		memcpy(dstHead, srcHead, sizeof(float) * 4);
	}
}

//�洢͸������Ϣ
static void loadOpacityToSplatScene(Model& model, SplatScene& splatScene)
{
	//��͸����ת����cpu
	auto cpuOpacity = model.opacities.cpu();
	//����͸���ȵ�����
	splatScene.opacity = new float[splatScene.pointNum];
	//����ÿ����
	for (uint32_t idPoint = 0; idPoint < splatScene.pointNum; ++idPoint)
	{
		auto dstHead = splatScene.opacity + idPoint;
		auto srcHead = reinterpret_cast<float*>(cpuOpacity[idPoint].data_ptr());
		memcpy(dstHead, srcHead, sizeof(float));
	}
}

//��model��������ݴ洢��C��ʽ
static void convertToSplatScene(Model& model,SplatScene& splatScene)
{
	//��ȡ�������tensor����
	auto pointTensor = getCpuPointList(model);
	//��¼����ĵ����
	splatScene.pointNum = getPointNum(model);
	//�������꿪�����ݿռ�
	splatScene.pointList = new float[splatScene.pointNum * 3];
	//���μ�¼ÿ���������
	for (int idPoint = 0; idPoint < splatScene.pointNum; ++idPoint)
	{
		auto dstHead = splatScene.pointList + idPoint * 3;
		auto srcHead = reinterpret_cast<float*>(pointTensor[idPoint].data_ptr());
		memcpy(dstHead, srcHead, sizeof(float) * 3);
	}
	//����ɫ���ݱ��浽SplatScene����
	loadColorToSplatScene(model, splatScene);
	//�洢SplatScene�����scale
	loadScaleToSplatScene(model, splatScene);
	//�洢SplatScene�������ת��Ϣ
	loadRotationToSplatScene(model, splatScene);
	//�洢͸������Ϣmodel
	loadOpacityToSplatScene(model, splatScene);
}

//ply��ʽ�µ�����
class PlyProperty
{
public:
	std::string type;
	std::string name;
};

//ply���ļ���ʽ��Ϣ
class PlyHeader
{
public:
	std::string format;
	//�ڵ�ĸ���
	int verticeNum = 0;
	//��Ƭ�ĸ���
	int faceNum = 0;
	//���е������б������ڵ�ĺ���Ƭ��
	std::vector<PlyProperty> propertyList;
};

//һ���ڵ������
class Vertex
{
public:
	//λ��
	float position[3];
	//������
	float normal[3];
	//��г�����Ĳ���
	float shs[48];
	//͸����
	float opacity;
	//���������scale
	float scale[3];
	//����Ԫ�����ɵ���ת
	float rotation[4];
};


//������Ƶ�����ͷ
static void loadPlyHeader(std::ifstream& fileHandle,
	PlyHeader& header
)
{
	//�м��ȡ����ÿһ�еĽ��
	std::string line;
	while (std::getline(fileHandle, line))
	{
		//�½�������
		std::istringstream tempStream(line);
		//��ȡһ������
		std::string token;
		tempStream >> token;
		//�ж��ǲ��Ǹ�ʽ��Ϣ
		if (token == "format")
		{
			tempStream >> header.format;
		}
		//�ж϶�ȡ�����ǲ���element��Ϣ
		else if (token == "element")
		{
			//�ٶ�ȡelement������
			tempStream >> token;
			//�ж��ǽڵ����������Ƭ�ĸ���
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
		//���ж��Ƿ��ȡ����������Ϣ
		else if (token == "property")
		{
			//�½�һ����ʱ������
			PlyProperty tempProperty;
			//��¼����type������
			tempStream >> tempProperty.type >> tempProperty.name;
			//�����Էŵ��б�����
			header.propertyList.push_back(tempProperty);
		}
		//header���ֵĽ�����
		else if (token == "end_header")
		{
			break;
		}

	}
}

//ֱ��ǿ������splat scene
static void loadSplatScene(SplatScene& splatScene)
{
	//�������ڵ��ļ�
	std::string filePath = "E:/temp/test.ply";
	//���ļ���������
	std::ifstream fileHandle(filePath, std::ios::binary);
	//�½�ply��ͷ��������
	PlyHeader header;
	loadPlyHeader(fileHandle, header);
	//���ڶ�ȡ�����������ݴ�С��Ŀǰ��ʱ��������չ��
	const int BUFFER_SIZE = 62;
	//���ڶ�ȡ���ݵĻ�����
	float pointBuffer[BUFFER_SIZE];
	//��splat scene��������ݿ��ٺÿռ�
	splatScene.pointList = new float[header.verticeNum * 3];
	splatScene.color = new float[header.verticeNum * 3];
	splatScene.scale = new float[header.verticeNum * 3];
	splatScene.rotation = new float[header.verticeNum * 4];
	splatScene.opacity = new float[header.verticeNum];
	splatScene.pointNum = header.verticeNum;
	//������ȡÿ����
	for (int idPoint = 0; idPoint < header.verticeNum; ++idPoint)
	{
		//��ǰҪ������ÿ�����ݵ�����ͷ
		auto posHead = splatScene.pointList + idPoint * 3;
		auto colorHead = splatScene.color + idPoint * 3;
		auto sizeHead = splatScene.scale + idPoint * 3;
		auto rotHead = splatScene.rotation + idPoint * 4;
		auto opacityHead = splatScene.opacity + idPoint;
		//��ȡ��ǰ�������������Ϣ
		fileHandle.read((char*)pointBuffer, sizeof(float) * BUFFER_SIZE);
		//��ȡ������
		memcpy(posHead, pointBuffer, 3 * sizeof(float));
		memcpy(colorHead, pointBuffer + 6, 3 * sizeof(float));
		//��ȡ͸������Ϣ
		opacityHead[0] = pointBuffer[54];
		//��ȡscale��Ϣ
		memcpy(sizeHead, pointBuffer + 55, 3 * sizeof(float));
		//��ȡ��ת����
		memcpy(rotHead, pointBuffer + 58, 4 * sizeof(float));
	}
	fileHandle.close();
}

extern "C" __declspec(dllexport) void reconstruct(const char* imgPathStr,
	const char* workspaceStr,void* dstScene)
{
	//����������ת����SplatScene
	auto splatScene = reinterpret_cast<SplatScene*>(dstScene);
	//ֱ�Ӵ��ļ������ȡ����
	//loadSplatScene(*splatScene);
	//return;
	//��ʼ��sparse scene�����ڻ�ȡcolmap�Ľ��
	SparseScene sparseScene;
	auto imgPath = std::string(imgPathStr);
	auto workspace = std::string(workspaceStr);
	callColmap(imgPath, workspace, sparseScene);

	std::cout << "point num: "
		<< sparseScene.xyz.size() << std::endl;

	//����ÿ��colmap��ͼƬ·�����������ͼƬ·����ɾ���·��
	for (auto& eachCamera : sparseScene.cameraList)
	{
		eachCamera.imgPath = imgPath + "/" +
			eachCamera.imgPath;
	}

	Model* retModel;
	//����splat�ļ���
	splatServer(sparseScene, "E:/temp/splat.ply", retModel);
	//����ģ�͵�ת��
	convertToSplatScene(*retModel, *splatScene);
}
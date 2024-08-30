#pragma once

//���ڴ���colmap����������������
class ColmapCamera
{
public:
	//��ת������Ԫ����ʾ�� xyzw
	float rotation[4];
	//ƽ�� xyz
	float translation[3];
	//ͼƬ��·��
	std::string imgPath;
};

//���ڴ洢���ؽ���Ľӿ�
class SparseScene
{
public:
	//xyz�ĵ�����
	std::vector<float> xyz;
	std::vector<uint8_t> rgb;

	//�ڲ���Ϣ �����ǽ��������
	std::vector<float> intrinsic;
	//ͼƬ��shape����Ҳ���ڲε�һ����
	unsigned imgShape[2];

	//ÿ����������
	std::vector<ColmapCamera> cameraList;
};
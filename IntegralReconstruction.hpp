#pragma once

//�ؽ��õĽ��
struct SplatScene
{
	//�����
	uint32_t pointNum;
	//�ؽ��õ�3D��
	float* pointList;
	//��ɫ��Ϣ
	float* color;
	//scale��Ϣ
	float* scale;
	//��ת��Ϣ
	float* rotation;
	//͸����
	float* opacity;
};
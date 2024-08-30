#pragma once

//重建好的结果
struct SplatScene
{
	//点个数
	uint32_t pointNum;
	//重建好的3D点
	float* pointList;
	//颜色信息
	float* color;
	//scale信息
	float* scale;
	//旋转信息
	float* rotation;
	//透明度
	float* opacity;
};
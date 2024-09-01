#pragma once

//调用计算每个3D gaussian的合法观察角度
void getGaussianViewAngle(float* gaussianList, float* camCenterList, 
	float* dstViewAngle,unsigned cameraNum, unsigned gaussianNum);
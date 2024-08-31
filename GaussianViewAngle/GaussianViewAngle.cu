#include"GaussianViewAngle/GaussianViewAngle.h"

__global__ void func() {

}

//调用计算每个3D gaussian的合法观察角度
//camCenterList 这个是cpu内存，gaussianList是gpu内存
void getGaussianViewAngle(float* gaussianList, float* camCenterList, float* dstViewAngle)
{

}
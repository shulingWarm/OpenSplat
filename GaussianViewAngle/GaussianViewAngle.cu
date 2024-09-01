#include"GaussianViewAngle/GaussianViewAngle.h"
#include <cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<iostream>
#include <thrust/extrema.h>

cudaError_t cudaHandleError(cudaError_t error_code)
{
    if (error_code != cudaSuccess)
    {
        printf("error_code: %d, error_name: %s, error_description: %s\n",
            error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code));
    }
    return error_code;
}

//简单的3D点数据结构
struct Point3D
{
    float x;
    float y;
    float z;
};

//计算两个相机的夹角
template<bool X_ANGLE>
__device__ float computeAngle(float* pointDiff);

//计算两个相机的夹角 这里是X方向的夹角 或者也可以理解成经度的夹角
template<>
__device__ float computeAngle<true>(float* pointDiff) //3D gaussian的坐标
{
    return atan2(pointDiff[0], pointDiff[1]);
}

//计算两个相机的夹角
template<>
__device__ float computeAngle<false>(float* pointDiff) //3D gaussian的坐标
{
    return atan2(pointDiff[2],
        norm3df(pointDiff[0],pointDiff[1], 0));
}

//一个最佳角度的比较周期
template<bool FIRST_LOOP> //判断是否为第一次循环
__device__ float compareAngleOneLoop(Point3D* sharedHead,
    Point3D* gaussianHead,
    float* bestAngle
)
{
    //计算差值向量
    float pointDiff[3] = { sharedHead->x - gaussianHead->x,
        sharedHead->y - gaussianHead->y,
        sharedHead->z - gaussianHead->z };
    //计算x方向的夹角
    float xAngle = computeAngle<true>(pointDiff);
    float yAngle = computeAngle<false>(pointDiff);
    //判断是不是第一次记录
    if constexpr (FIRST_LOOP)
    {
        bestAngle[0] = xAngle;
        bestAngle[1] = xAngle;
        bestAngle[2] = yAngle;
        bestAngle[3] = yAngle;
    }
    else
    {
        bestAngle[0] = fmin(xAngle, bestAngle[0]);
        bestAngle[1] = fmax(xAngle, bestAngle[1]);
        bestAngle[2] = fmin(yAngle, bestAngle[2]);
        bestAngle[3] = fmax(yAngle, bestAngle[3]);
    }
}

//一个计算周期
template<bool FIRST_LOOP, //是否要作为第一步来处理
    uint32_t CAM_LOAD_NUM,
    uint32_t THREAD_NUM //kernel里面的线程数
>
__device__ void bestAngleOneLoop(Point3D* sharedHead, //共享内存的头指针，会用于往这里面存储数据
    Point3D* camCenterList, //相机光心列表，这是global memory
    Point3D* gaussianPoint, //3D高斯点, 这里面仅仅是一个3D坐标 属于是寄存器上面的指针
    float* bestAngle, //目前的最佳角度 它可能是未初始化过的，这会由FIRST_STEP这个模板参数来决定
    unsigned cameraNum //剩余需要读取的相机个数
)
{
    //当前迭代需要读取的相机总个数
    auto loadingCamNum = CAM_LOAD_NUM;
    if (cameraNum < loadingCamNum)
        loadingCamNum = cameraNum;
    for (uint32_t readId = threadIdx.x; readId < loadingCamNum; readId += THREAD_NUM)
    {
        //从global memory里面取值
        auto localCamCenter = camCenterList[readId];
        //将计算结果写入到shared memory里面
        sharedHead[readId] = localCamCenter;
    }
    __syncthreads();
    //先进行第一轮考虑先手的比较逻辑
    compareAngleOneLoop<FIRST_LOOP>(sharedHead, gaussianPoint, bestAngle);
    //循环比较后续的角度
    for (uint32_t idCam = 1; idCam < loadingCamNum; ++idCam)
    {
        //进行后续的角度比较
        compareAngleOneLoop<false>(sharedHead + idCam, gaussianPoint, bestAngle);
    }
}

template<uint32_t CAM_LOAD_NUM>
__global__ void computeGaussianViewAngle(
    Point3D* gaussianList, Point3D* camCenterList,
    float* dstViewAngle, unsigned cameraNum, unsigned gaussianNum
) {
    //用于存储相机光心坐标的共享内存
    __shared__ Point3D sharedCamCenter[CAM_LOAD_NUM];
    //当前线程的warp id
    uint32_t idPoint = threadIdx.x / 32 + (CAM_LOAD_NUM / 32) * blockIdx.x;
    //如果访问的点越界了，那这个warp就可以直接结束了
    if (idPoint >= gaussianNum)
        return;
    //初始化当前线程维度的最值
    float bestAngle[4];
    //准备当前的gaussian点
    auto gaussianPoint = gaussianList[idPoint];
    //先进行第一步循环
    bestAngleOneLoop<true, CAM_LOAD_NUM, CAM_LOAD_NUM>(sharedCamCenter, camCenterList, &gaussianPoint,
        bestAngle, cameraNum);
    //进行后续循环
    for (uint32_t idCamera = 0; idCamera < cameraNum; idCamera += CAM_LOAD_NUM)
    {
        __syncthreads();
        bestAngleOneLoop<false, CAM_LOAD_NUM, CAM_LOAD_NUM>(sharedCamCenter,
            camCenterList + idCamera, &gaussianPoint, bestAngle, cameraNum);
    }
}

//调用计算每个3D gaussian的合法观察角度
//camCenterList 这个是cpu内存，gaussianList是gpu内存
//这个dstViewAngle里面是用于存储结果的，并且已经开辟好了空间
void getGaussianViewAngle(float* gaussianList, float* camCenterList,
	float* dstViewAngle, unsigned cameraNum, unsigned gaussianNum)
{
	//把相机光心数据转换到GPU内存
	float* gpuCamCenter;
    cudaHandleError(cudaMalloc((void**)&gpuCamCenter, sizeof(float) * cameraNum * 3));
    cudaHandleError(cudaMemcpy(gpuCamCenter, camCenterList, sizeof(float) * cameraNum * 3,
        cudaMemcpyHostToDevice));
    float* gpuViewAngle;
    cudaHandleError(cudaMalloc((void**)&gpuViewAngle, sizeof(float) * gaussianNum * 4));
    //开辟一段内存，用于记录结果中每个3D gaussian的视角范围
    computeGaussianViewAngle<256> << <(gaussianNum + 1) / 2, 256 >> > (
        (Point3D*)gaussianList, (Point3D*)gpuCamCenter,
        gpuViewAngle, cameraNum, gaussianNum);
}
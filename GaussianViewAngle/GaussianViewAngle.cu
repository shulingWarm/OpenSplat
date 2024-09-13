#include"GaussianViewAngle/GaussianViewAngle.h"
#include <cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<iostream>
#include <thrust/extrema.h>

//保存相机角度时用的数据类型
typedef uint64_t AngleRecorder;

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
    return atan2(pointDiff[1], pointDiff[2]);
}

//计算两个相机的夹角
template<>
__device__ float computeAngle<false>(float* pointDiff) //3D gaussian的坐标
{
    return acos(pointDiff[2]/
        norm3df(pointDiff[0],pointDiff[1], pointDiff[2]));
}

//计算两个角度的绝对差
//如果angle在baseAngle的逆时针方向，则返回正数
//如果angle在baseAngle的顺时针方向，则返回负数
__device__ float getAbsAngleDiff(float angle, float baseAngle)
{
    float angleDiff = fmod(angle - baseAngle + 6.28318530718, 6.28318530718);
    if (angleDiff > 3.14159265359)
    {
        return angleDiff - 6.28318530718;
    }
    return angleDiff;
}

//一个最佳角度的比较周期
template<bool FIRST_LOOP> //判断是否为第一次循环
__device__ float compareAngleOneLoop(Point3D* sharedHead,
    Point3D* gaussianHead,
    AngleRecorder* bestAngle
)
{
    constexpr float X_ANGLE_SEG = 6.28318530718 / 64;
    constexpr float Y_ANGLE_SEG = 3.14159265359 / 64;
    //计算差值向量
    float pointDiff[3] = { sharedHead->x - gaussianHead->x,
        sharedHead->y - gaussianHead->y,
        sharedHead->z - gaussianHead->z };
    //计算x方向的夹角
    float xAngle = computeAngle<true>(pointDiff);
    float yAngle = computeAngle<false>(pointDiff);
    //计算xy角度所在的bit位
    int xBit = xAngle / X_ANGLE_SEG;
    int yBit = yAngle / Y_ANGLE_SEG;
    //对原本的角度范围做与操作
    bestAngle[0] |= (1 << xBit);
    bestAngle[1] |= (1 << yBit);
}

//一个计算周期
template<bool FIRST_LOOP, //是否要作为第一步来处理
    uint32_t CAM_LOAD_NUM,
    uint32_t THREAD_NUM //kernel里面的线程数
>
__device__ void bestAngleOneLoop(Point3D* sharedHead, //共享内存的头指针，会用于往这里面存储数据
    Point3D* camCenterList, //相机光心列表，这是global memory
    Point3D* gaussianPoint, //3D高斯点, 这里面仅仅是一个3D坐标 属于是寄存器上面的指针
    AngleRecorder* bestAngle, //目前的最佳角度 它可能是未初始化过的，这会由FIRST_STEP这个模板参数来决定
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

//把某个角度处理到0-2PI的范围
inline __device__ float getStandardAngle(float angle)
{
    return fmod(angle + 6.28318530718, 6.28318530718);
}

//最终的角度归约操作
__device__ void angleRangeReduction(AngleRecorder* bestAngle,uint32_t cameraNum)
{
    //把线程负责的角度重置到0-2PI
    //bestAngle[4] = getStandardAngle(bestAngle[4]);
    //bestAngle[5] = getStandardAngle(bestAngle[5]);
    //// 按照蝶式归约计算
    //for (int i = 32; i > 1; i/=2)
    //{
    //    //计算x位置角度的最大值和最小值
    //    float xMinAngle = getStandardAngle(bestAngle[4] + bestAngle[0]);
    //    float xMaxAngle = getStandardAngle(bestAngle[4] + bestAngle[1]);
    //    //计算y方向的角度的最大值和最小值
    //    float yMinAngle = getStandardAngle(bestAngle[5] + bestAngle[2]);
    //    float yMaxAngle = getStandardAngle(bestAngle[5] + bestAngle[3]);
    //    //获取相邻位置的xy的角度范围
    //    float neighborAngle[4];
    //    neighborAngle[0] = __shfl_xor_sync(0xffffffff, xMinAngle, i - 1);
    //    neighborAngle[1] = __shfl_xor_sync(0xffffffff, xMaxAngle, i - 1);
    //    neighborAngle[2] = __shfl_xor_sync(0xffffffff, yMinAngle, i - 1);
    //    neighborAngle[3] = __shfl_xor_sync(0xffffffff, yMaxAngle, i - 1);
    //    //计算角度的中值差
    //    neighborAngle[0] = getAbsAngleDiff(neighborAngle[0], bestAngle[4]);
    //    neighborAngle[1] = getAbsAngleDiff(neighborAngle[1], bestAngle[4]);
    //    neighborAngle[2] = getAbsAngleDiff(neighborAngle[2], bestAngle[5]);
    //    neighborAngle[3] = getAbsAngleDiff(neighborAngle[3], bestAngle[5]);
    //    //重新合并角度
    //    bestAngle[0] = fmin(bestAngle[0], neighborAngle[0]);
    //    bestAngle[1] = fmax(bestAngle[1], neighborAngle[1]);
    //    bestAngle[2] = fmin(bestAngle[2], neighborAngle[2]);
    //    bestAngle[3] = fmax(bestAngle[3], neighborAngle[3]);
    //}
    ////计算最小角度 这样它就只有0,1,2,3这4个角度值是有意义的了
    //bestAngle[0] = getStandardAngle(bestAngle[4] + bestAngle[0]);
    //bestAngle[1] = getStandardAngle(bestAngle[4] + bestAngle[1]);
    //bestAngle[1] = getAbsAngleDiff(bestAngle[1], bestAngle[0]);
    //bestAngle[2] = getStandardAngle(bestAngle[5] + bestAngle[2]);
    //bestAngle[3] = getStandardAngle(bestAngle[5] + bestAngle[3]);
    //bestAngle[3] = getAbsAngleDiff(bestAngle[3], bestAngle[2]);
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
    AngleRecorder bestAngle[2] = { 0,0 };
    //准备当前的gaussian点
    //为了和UE里面的变换保持一致，这里把点坐标加载成(x,-z,-y)
    auto gaussianPoint = gaussianList[idPoint];
    {
        float temp = gaussianPoint.y;
        gaussianPoint.y = -gaussianPoint.z;
        gaussianPoint.z = -temp;
    }
    //先进行第一步循环
    bestAngleOneLoop<true, CAM_LOAD_NUM, CAM_LOAD_NUM>(sharedCamCenter, camCenterList, &gaussianPoint,
        bestAngle, cameraNum);
    //进行后续循环
    for (uint32_t idCamera = 1; idCamera < cameraNum; idCamera += CAM_LOAD_NUM)
    {
        __syncthreads();
        bestAngleOneLoop<false, CAM_LOAD_NUM, CAM_LOAD_NUM>(sharedCamCenter,
            camCenterList + idCamera, &gaussianPoint, bestAngle, cameraNum - idCamera * CAM_LOAD_NUM);
    }
    __syncthreads();
    //调用最终的角度归约
    angleRangeReduction(bestAngle, cameraNum);
    //计算当前线程应该写入的归约角度位置
    float* dstViewHead = dstViewAngle + idPoint * 4;
    //把归约后的角度保存到view range
    if (threadIdx.x % 32 == 0)
    {
        for (int i = 0; i < 4; ++i)
            dstViewHead[i] = bestAngle[i];
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
    //把视角范围复制到cpu
    cudaHandleError(cudaMemcpy(dstViewAngle, gpuViewAngle, sizeof(float) * gaussianNum * 4,
        cudaMemcpyDeviceToHost));
    //释放这中间用到的gpu内存
    cudaHandleError(cudaFree(gpuCamCenter));
    cudaHandleError(cudaFree(gpuViewAngle));
}
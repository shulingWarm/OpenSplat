#include"GaussianViewAngle/GaussianViewAngle.h"
#include <cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<iostream>
#include <thrust/extrema.h>
#include<vector>

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
    //进行归约操作，合并一个warp里面的32个线程
    for (int i = 0; i > 1; i>>=1)
    {
        //目前的x范围
        auto xRange = bestAngle[0];
        auto yRange = bestAngle[1];
        bestAngle[0] = xRange | __shfl_xor_sync(0xffffffff, xRange, i - 1);
        bestAngle[1] = yRange | __shfl_xor_sync(0xffffffff, yRange, i - 1);
    }
}

template<uint32_t CAM_LOAD_NUM>
__global__ void computeGaussianViewAngle(
    Point3D* gaussianList, Point3D* camCenterList,
    AngleRecorder* dstViewAngle, unsigned cameraNum, unsigned gaussianNum
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
    auto dstViewHead = dstViewAngle + idPoint * 2;
    //把归约后的角度保存到view range
    if (threadIdx.x % 32 == 0)
    {
        for (int i = 0; i < 2; ++i)
            dstViewHead[i] = bestAngle[i];
    }
}

static void convertSingleRecorder(AngleRecorder* recorder,
    float* viewRange,
    float bitSeg
)
{
    //寻找recorder里面的第一个1出现的位置
    int idBegin = -1;
    for (int i = 0; i < 64; ++i)
    {
        //判断当前位置是不是1
        if (recorder[0] & (1 << i))
        {
            idBegin = i;
            break;
        }
    }
    //如果没有找到有效的1,那就可以直接退出了
    if (idBegin == -1)
    {
        viewRange[0] = 0;
        viewRange[1] = 0;
        return;
    }
    //初始化目前最大的空位置区间
    int maxZeroSize = 0;
    int maxZeroBegin = -1;
    //目前的临时起始位置
    int tempBegin = -1;
    int tempLength = 0;
    //寻找这里面最大的零区间
    for (int i = 1; i < 64; ++i)
    {
        //当前位置的实际偏移量
        int idOffset = (idBegin + i) % 64;
        //判断当前位置是不是1
        if ((1 << idOffset) & recorder[0])
        {
            //判断目前是否存在有效的数据
            if (tempLength > maxZeroSize)
            {
                maxZeroSize = tempLength;
                maxZeroBegin = tempBegin;
            }
            tempLength = 0;
            tempBegin = -1;
        }
        else
        {
            //判断目前是否存在有效的begin
            if (tempBegin == -1)
            {
                tempBegin = idOffset;
            }
            ++tempLength;
        }
    }
    //判断目前是否存在有效的数据
    //这是考虑最后一个位置还没结束的情况
    if (tempLength > maxZeroSize)
    {
        maxZeroSize = tempLength;
        maxZeroBegin = tempBegin;
    }
    //计算正角度的开始位置
    int postiveBegin = (maxZeroBegin + maxZeroSize) % 64;
    int postiveLength = 64 - maxZeroSize;
    //如果负区域角度较大那就按照负区域角度来算
    int finalBegin = postiveBegin;
    int finalLength;
    //如果零区间较小，那说明有效区间是一个U角
    //这种情况下应该存储无效角度区间，然后把角度存储成一个负数
    if (maxZeroSize < postiveLength)
    {
        finalLength = -maxZeroSize;
    }
    else
    {
        finalLength = postiveLength;
    }
    //把bit形式的角度最终转换成float形式的角度范围
    viewRange[0] = finalBegin * bitSeg;
    viewRange[1] = finalLength * bitSeg;
}

//把bit形式的角度范围转换成float形式的角度范围
void convertAngleRecorderToViewRange(AngleRecorder* angleRecorder,
    float* dstViewRange,uint32_t gaussianNum //高斯点的个数
)
{
    constexpr float X_ANGLE_SEG = 6.28318530718 / 64;
    constexpr float Y_ANGLE_SEG = 3.14159265359 / 64;
    //遍历每个需要被处理的高斯点，这就仅仅由一个高斯来完成
    for (uint32_t idPoint = 0; idPoint < gaussianNum; ++idPoint)
    {
        auto viewRangeHead = dstViewRange + idPoint * 4;
        //recorder的头指针
        auto recorderHead = angleRecorder + idPoint * 2;
        //把recorder的x,y转换到y的形式
        convertSingleRecorder(recorderHead, viewRangeHead, X_ANGLE_SEG);
        convertSingleRecorder(recorderHead + 1, viewRangeHead + 2, Y_ANGLE_SEG);
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
    AngleRecorder* gpuViewAngle;
    cudaHandleError(cudaMalloc((void**)&gpuViewAngle, sizeof(AngleRecorder) * gaussianNum * 2));
    //开辟一段内存，用于记录结果中每个3D gaussian的视角范围
    computeGaussianViewAngle<256> << <(gaussianNum + 1) / 2, 256 >> > (
        (Point3D*)gaussianList, (Point3D*)gpuCamCenter,
        gpuViewAngle, cameraNum, gaussianNum);
    //准备角度记录工作的cpu版本
    std::vector<AngleRecorder> recorderVec(gaussianNum * 2);
    //把视角范围保存到cpu
    cudaHandleError(cudaMemcpy(recorderVec.data(), gpuViewAngle,
        sizeof(AngleRecorder) * recorderVec.size(), cudaMemcpyDeviceToHost));
    //传入最终的目标float范围，做最终的角度范围调整
    convertAngleRecorderToViewRange(recorderVec.data(),
        dstViewAngle, gaussianNum);
    //释放这中间用到的gpu内存
    cudaHandleError(cudaFree(gpuCamCenter));
    cudaHandleError(cudaFree(gpuViewAngle));
}
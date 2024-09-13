#include"GaussianViewAngle/GaussianViewAngle.h"
#include <cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<iostream>
#include <thrust/extrema.h>

//��������Ƕ�ʱ�õ���������
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

//�򵥵�3D�����ݽṹ
struct Point3D
{
    float x;
    float y;
    float z;
};

//������������ļн�
template<bool X_ANGLE>
__device__ float computeAngle(float* pointDiff);

//������������ļн� ������X����ļн� ����Ҳ�������ɾ��ȵļн�
template<>
__device__ float computeAngle<true>(float* pointDiff) //3D gaussian������
{
    return atan2(pointDiff[1], pointDiff[2]);
}

//������������ļн�
template<>
__device__ float computeAngle<false>(float* pointDiff) //3D gaussian������
{
    return acos(pointDiff[2]/
        norm3df(pointDiff[0],pointDiff[1], pointDiff[2]));
}

//���������Ƕȵľ��Բ�
//���angle��baseAngle����ʱ�뷽���򷵻�����
//���angle��baseAngle��˳ʱ�뷽���򷵻ظ���
__device__ float getAbsAngleDiff(float angle, float baseAngle)
{
    float angleDiff = fmod(angle - baseAngle + 6.28318530718, 6.28318530718);
    if (angleDiff > 3.14159265359)
    {
        return angleDiff - 6.28318530718;
    }
    return angleDiff;
}

//һ����ѽǶȵıȽ�����
template<bool FIRST_LOOP> //�ж��Ƿ�Ϊ��һ��ѭ��
__device__ float compareAngleOneLoop(Point3D* sharedHead,
    Point3D* gaussianHead,
    AngleRecorder* bestAngle
)
{
    constexpr float X_ANGLE_SEG = 6.28318530718 / 64;
    constexpr float Y_ANGLE_SEG = 3.14159265359 / 64;
    //�����ֵ����
    float pointDiff[3] = { sharedHead->x - gaussianHead->x,
        sharedHead->y - gaussianHead->y,
        sharedHead->z - gaussianHead->z };
    //����x����ļн�
    float xAngle = computeAngle<true>(pointDiff);
    float yAngle = computeAngle<false>(pointDiff);
    //����xy�Ƕ����ڵ�bitλ
    int xBit = xAngle / X_ANGLE_SEG;
    int yBit = yAngle / Y_ANGLE_SEG;
    //��ԭ���ĽǶȷ�Χ�������
    bestAngle[0] |= (1 << xBit);
    bestAngle[1] |= (1 << yBit);
}

//һ����������
template<bool FIRST_LOOP, //�Ƿ�Ҫ��Ϊ��һ��������
    uint32_t CAM_LOAD_NUM,
    uint32_t THREAD_NUM //kernel������߳���
>
__device__ void bestAngleOneLoop(Point3D* sharedHead, //�����ڴ��ͷָ�룬��������������洢����
    Point3D* camCenterList, //��������б�����global memory
    Point3D* gaussianPoint, //3D��˹��, �����������һ��3D���� �����ǼĴ��������ָ��
    AngleRecorder* bestAngle, //Ŀǰ����ѽǶ� ��������δ��ʼ�����ģ������FIRST_STEP���ģ�����������
    unsigned cameraNum //ʣ����Ҫ��ȡ���������
)
{
    //��ǰ������Ҫ��ȡ������ܸ���
    auto loadingCamNum = CAM_LOAD_NUM;
    if (cameraNum < loadingCamNum)
        loadingCamNum = cameraNum;
    for (uint32_t readId = threadIdx.x; readId < loadingCamNum; readId += THREAD_NUM)
    {
        //��global memory����ȡֵ
        auto localCamCenter = camCenterList[readId];
        //��������д�뵽shared memory����
        sharedHead[readId] = localCamCenter;
    }
    __syncthreads();
    //�Ƚ��е�һ�ֿ������ֵıȽ��߼�
    compareAngleOneLoop<FIRST_LOOP>(sharedHead, gaussianPoint, bestAngle);
    //ѭ���ȽϺ����ĽǶ�
    for (uint32_t idCam = 1; idCam < loadingCamNum; ++idCam)
    {
        //���к����ĽǶȱȽ�
        compareAngleOneLoop<false>(sharedHead + idCam, gaussianPoint, bestAngle);
    }
}

//��ĳ���Ƕȴ���0-2PI�ķ�Χ
inline __device__ float getStandardAngle(float angle)
{
    return fmod(angle + 6.28318530718, 6.28318530718);
}

//���յĽǶȹ�Լ����
__device__ void angleRangeReduction(AngleRecorder* bestAngle,uint32_t cameraNum)
{
    //���̸߳���ĽǶ����õ�0-2PI
    //bestAngle[4] = getStandardAngle(bestAngle[4]);
    //bestAngle[5] = getStandardAngle(bestAngle[5]);
    //// ���յ�ʽ��Լ����
    //for (int i = 32; i > 1; i/=2)
    //{
    //    //����xλ�ýǶȵ����ֵ����Сֵ
    //    float xMinAngle = getStandardAngle(bestAngle[4] + bestAngle[0]);
    //    float xMaxAngle = getStandardAngle(bestAngle[4] + bestAngle[1]);
    //    //����y����ĽǶȵ����ֵ����Сֵ
    //    float yMinAngle = getStandardAngle(bestAngle[5] + bestAngle[2]);
    //    float yMaxAngle = getStandardAngle(bestAngle[5] + bestAngle[3]);
    //    //��ȡ����λ�õ�xy�ĽǶȷ�Χ
    //    float neighborAngle[4];
    //    neighborAngle[0] = __shfl_xor_sync(0xffffffff, xMinAngle, i - 1);
    //    neighborAngle[1] = __shfl_xor_sync(0xffffffff, xMaxAngle, i - 1);
    //    neighborAngle[2] = __shfl_xor_sync(0xffffffff, yMinAngle, i - 1);
    //    neighborAngle[3] = __shfl_xor_sync(0xffffffff, yMaxAngle, i - 1);
    //    //����Ƕȵ���ֵ��
    //    neighborAngle[0] = getAbsAngleDiff(neighborAngle[0], bestAngle[4]);
    //    neighborAngle[1] = getAbsAngleDiff(neighborAngle[1], bestAngle[4]);
    //    neighborAngle[2] = getAbsAngleDiff(neighborAngle[2], bestAngle[5]);
    //    neighborAngle[3] = getAbsAngleDiff(neighborAngle[3], bestAngle[5]);
    //    //���ºϲ��Ƕ�
    //    bestAngle[0] = fmin(bestAngle[0], neighborAngle[0]);
    //    bestAngle[1] = fmax(bestAngle[1], neighborAngle[1]);
    //    bestAngle[2] = fmin(bestAngle[2], neighborAngle[2]);
    //    bestAngle[3] = fmax(bestAngle[3], neighborAngle[3]);
    //}
    ////������С�Ƕ� ��������ֻ��0,1,2,3��4���Ƕ�ֵ�����������
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
    //���ڴ洢�����������Ĺ����ڴ�
    __shared__ Point3D sharedCamCenter[CAM_LOAD_NUM];
    //��ǰ�̵߳�warp id
    uint32_t idPoint = threadIdx.x / 32 + (CAM_LOAD_NUM / 32) * blockIdx.x;
    //������ʵĵ�Խ���ˣ������warp�Ϳ���ֱ�ӽ�����
    if (idPoint >= gaussianNum)
        return;
    //��ʼ����ǰ�߳�ά�ȵ���ֵ
    AngleRecorder bestAngle[2] = { 0,0 };
    //׼����ǰ��gaussian��
    //Ϊ�˺�UE����ı任����һ�£�����ѵ�������س�(x,-z,-y)
    auto gaussianPoint = gaussianList[idPoint];
    {
        float temp = gaussianPoint.y;
        gaussianPoint.y = -gaussianPoint.z;
        gaussianPoint.z = -temp;
    }
    //�Ƚ��е�һ��ѭ��
    bestAngleOneLoop<true, CAM_LOAD_NUM, CAM_LOAD_NUM>(sharedCamCenter, camCenterList, &gaussianPoint,
        bestAngle, cameraNum);
    //���к���ѭ��
    for (uint32_t idCamera = 1; idCamera < cameraNum; idCamera += CAM_LOAD_NUM)
    {
        __syncthreads();
        bestAngleOneLoop<false, CAM_LOAD_NUM, CAM_LOAD_NUM>(sharedCamCenter,
            camCenterList + idCamera, &gaussianPoint, bestAngle, cameraNum - idCamera * CAM_LOAD_NUM);
    }
    __syncthreads();
    //�������յĽǶȹ�Լ
    angleRangeReduction(bestAngle, cameraNum);
    //���㵱ǰ�߳�Ӧ��д��Ĺ�Լ�Ƕ�λ��
    float* dstViewHead = dstViewAngle + idPoint * 4;
    //�ѹ�Լ��ĽǶȱ��浽view range
    if (threadIdx.x % 32 == 0)
    {
        for (int i = 0; i < 4; ++i)
            dstViewHead[i] = bestAngle[i];
    }
}

//���ü���ÿ��3D gaussian�ĺϷ��۲�Ƕ�
//camCenterList �����cpu�ڴ棬gaussianList��gpu�ڴ�
//���dstViewAngle���������ڴ洢����ģ������Ѿ����ٺ��˿ռ�
void getGaussianViewAngle(float* gaussianList, float* camCenterList,
	float* dstViewAngle, unsigned cameraNum, unsigned gaussianNum)
{
	//�������������ת����GPU�ڴ�
	float* gpuCamCenter;
    cudaHandleError(cudaMalloc((void**)&gpuCamCenter, sizeof(float) * cameraNum * 3));
    cudaHandleError(cudaMemcpy(gpuCamCenter, camCenterList, sizeof(float) * cameraNum * 3,
        cudaMemcpyHostToDevice));
    float* gpuViewAngle;
    cudaHandleError(cudaMalloc((void**)&gpuViewAngle, sizeof(float) * gaussianNum * 4));
    //����һ���ڴ棬���ڼ�¼�����ÿ��3D gaussian���ӽǷ�Χ
    computeGaussianViewAngle<256> << <(gaussianNum + 1) / 2, 256 >> > (
        (Point3D*)gaussianList, (Point3D*)gpuCamCenter,
        gpuViewAngle, cameraNum, gaussianNum);
    //���ӽǷ�Χ���Ƶ�cpu
    cudaHandleError(cudaMemcpy(dstViewAngle, gpuViewAngle, sizeof(float) * gaussianNum * 4,
        cudaMemcpyDeviceToHost));
    //�ͷ����м��õ���gpu�ڴ�
    cudaHandleError(cudaFree(gpuCamCenter));
    cudaHandleError(cudaFree(gpuViewAngle));
}
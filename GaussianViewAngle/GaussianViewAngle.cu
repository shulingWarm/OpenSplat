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
    return atan2(pointDiff[0], pointDiff[1]);
}

//������������ļн�
template<>
__device__ float computeAngle<false>(float* pointDiff) //3D gaussian������
{
    return atan2(pointDiff[2],
        norm3df(pointDiff[0],pointDiff[1], 0));
}

//һ����ѽǶȵıȽ�����
template<bool FIRST_LOOP> //�ж��Ƿ�Ϊ��һ��ѭ��
__device__ float compareAngleOneLoop(Point3D* sharedHead,
    Point3D* gaussianHead,
    float* bestAngle
)
{
    //�����ֵ����
    float pointDiff[3] = { sharedHead->x - gaussianHead->x,
        sharedHead->y - gaussianHead->y,
        sharedHead->z - gaussianHead->z };
    //����x����ļн�
    float xAngle = computeAngle<true>(pointDiff);
    float yAngle = computeAngle<false>(pointDiff);
    //�ж��ǲ��ǵ�һ�μ�¼
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

//һ����������
template<bool FIRST_LOOP, //�Ƿ�Ҫ��Ϊ��һ��������
    uint32_t CAM_LOAD_NUM,
    uint32_t THREAD_NUM //kernel������߳���
>
__device__ void bestAngleOneLoop(Point3D* sharedHead, //�����ڴ��ͷָ�룬��������������洢����
    Point3D* camCenterList, //��������б�����global memory
    Point3D* gaussianPoint, //3D��˹��, �����������һ��3D���� �����ǼĴ��������ָ��
    float* bestAngle, //Ŀǰ����ѽǶ� ��������δ��ʼ�����ģ������FIRST_STEP���ģ�����������
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
    float bestAngle[4];
    //׼����ǰ��gaussian��
    auto gaussianPoint = gaussianList[idPoint];
    //�Ƚ��е�һ��ѭ��
    bestAngleOneLoop<true, CAM_LOAD_NUM, CAM_LOAD_NUM>(sharedCamCenter, camCenterList, &gaussianPoint,
        bestAngle, cameraNum);
    //���к���ѭ��
    for (uint32_t idCamera = 0; idCamera < cameraNum; idCamera += CAM_LOAD_NUM)
    {
        __syncthreads();
        bestAngleOneLoop<false, CAM_LOAD_NUM, CAM_LOAD_NUM>(sharedCamCenter,
            camCenterList + idCamera, &gaussianPoint, bestAngle, cameraNum);
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
}
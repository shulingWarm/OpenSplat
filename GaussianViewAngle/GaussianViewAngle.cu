#include"GaussianViewAngle/GaussianViewAngle.h"
#include <cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<iostream>
#include <thrust/extrema.h>
#include<vector>

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
    //���й�Լ�������ϲ�һ��warp�����32���߳�
    for (int i = 0; i > 1; i>>=1)
    {
        //Ŀǰ��x��Χ
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
    auto dstViewHead = dstViewAngle + idPoint * 2;
    //�ѹ�Լ��ĽǶȱ��浽view range
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
    //Ѱ��recorder����ĵ�һ��1���ֵ�λ��
    int idBegin = -1;
    for (int i = 0; i < 64; ++i)
    {
        //�жϵ�ǰλ���ǲ���1
        if (recorder[0] & (1 << i))
        {
            idBegin = i;
            break;
        }
    }
    //���û���ҵ���Ч��1,�ǾͿ���ֱ���˳���
    if (idBegin == -1)
    {
        viewRange[0] = 0;
        viewRange[1] = 0;
        return;
    }
    //��ʼ��Ŀǰ���Ŀ�λ������
    int maxZeroSize = 0;
    int maxZeroBegin = -1;
    //Ŀǰ����ʱ��ʼλ��
    int tempBegin = -1;
    int tempLength = 0;
    //Ѱ������������������
    for (int i = 1; i < 64; ++i)
    {
        //��ǰλ�õ�ʵ��ƫ����
        int idOffset = (idBegin + i) % 64;
        //�жϵ�ǰλ���ǲ���1
        if ((1 << idOffset) & recorder[0])
        {
            //�ж�Ŀǰ�Ƿ������Ч������
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
            //�ж�Ŀǰ�Ƿ������Ч��begin
            if (tempBegin == -1)
            {
                tempBegin = idOffset;
            }
            ++tempLength;
        }
    }
    //�ж�Ŀǰ�Ƿ������Ч������
    //���ǿ������һ��λ�û�û���������
    if (tempLength > maxZeroSize)
    {
        maxZeroSize = tempLength;
        maxZeroBegin = tempBegin;
    }
    //�������ǶȵĿ�ʼλ��
    int postiveBegin = (maxZeroBegin + maxZeroSize) % 64;
    int postiveLength = 64 - maxZeroSize;
    //���������ǶȽϴ��ǾͰ��ո�����Ƕ�����
    int finalBegin = postiveBegin;
    int finalLength;
    //����������С����˵����Ч������һ��U��
    //���������Ӧ�ô洢��Ч�Ƕ����䣬Ȼ��ѽǶȴ洢��һ������
    if (maxZeroSize < postiveLength)
    {
        finalLength = -maxZeroSize;
    }
    else
    {
        finalLength = postiveLength;
    }
    //��bit��ʽ�ĽǶ�����ת����float��ʽ�ĽǶȷ�Χ
    viewRange[0] = finalBegin * bitSeg;
    viewRange[1] = finalLength * bitSeg;
}

//��bit��ʽ�ĽǶȷ�Χת����float��ʽ�ĽǶȷ�Χ
void convertAngleRecorderToViewRange(AngleRecorder* angleRecorder,
    float* dstViewRange,uint32_t gaussianNum //��˹��ĸ���
)
{
    constexpr float X_ANGLE_SEG = 6.28318530718 / 64;
    constexpr float Y_ANGLE_SEG = 3.14159265359 / 64;
    //����ÿ����Ҫ������ĸ�˹�㣬��ͽ�����һ����˹�����
    for (uint32_t idPoint = 0; idPoint < gaussianNum; ++idPoint)
    {
        auto viewRangeHead = dstViewRange + idPoint * 4;
        //recorder��ͷָ��
        auto recorderHead = angleRecorder + idPoint * 2;
        //��recorder��x,yת����y����ʽ
        convertSingleRecorder(recorderHead, viewRangeHead, X_ANGLE_SEG);
        convertSingleRecorder(recorderHead + 1, viewRangeHead + 2, Y_ANGLE_SEG);
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
    AngleRecorder* gpuViewAngle;
    cudaHandleError(cudaMalloc((void**)&gpuViewAngle, sizeof(AngleRecorder) * gaussianNum * 2));
    //����һ���ڴ棬���ڼ�¼�����ÿ��3D gaussian���ӽǷ�Χ
    computeGaussianViewAngle<256> << <(gaussianNum + 1) / 2, 256 >> > (
        (Point3D*)gaussianList, (Point3D*)gpuCamCenter,
        gpuViewAngle, cameraNum, gaussianNum);
    //׼���Ƕȼ�¼������cpu�汾
    std::vector<AngleRecorder> recorderVec(gaussianNum * 2);
    //���ӽǷ�Χ���浽cpu
    cudaHandleError(cudaMemcpy(recorderVec.data(), gpuViewAngle,
        sizeof(AngleRecorder) * recorderVec.size(), cudaMemcpyDeviceToHost));
    //�������յ�Ŀ��float��Χ�������յĽǶȷ�Χ����
    convertAngleRecorderToViewRange(recorderVec.data(),
        dstViewAngle, gaussianNum);
    //�ͷ����м��õ���gpu�ڴ�
    cudaHandleError(cudaFree(gpuCamCenter));
    cudaHandleError(cudaFree(gpuViewAngle));
}
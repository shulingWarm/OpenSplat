#pragma once

#include<colmap/controllers/automatic_reconstruction.h>
#include<colmap/controllers/incremental_mapper.h>
#include<iostream>
#include<fstream>
#include<unordered_map>
#include"ColmapServer/SparseScene.hpp"
#include"Debug.hpp"

#define MY_LOG(exp) std::cout<<exp<<std::endl;


using namespace colmap;

//camera����������
using CameraMap = std::unordered_map<camera_t, Camera>;
//colmap�������õĵ���������
using PointMap = std::unordered_map<point3D_t, Point3D>;
//ͼƬ���ݵ���������
using ImageMap = std::unordered_map<image_t, class Image>;
//ÿ��ͼƬ����ĵ��б�
using ViewPointList = std::vector<Point2D>;

//server��controller
//���������̳�colmap��������̿��Ƶ�
class ServerController : public colmap::AutomaticReconstructionController
{
public:

	SparseScene* dstScene; //����ڽ���ᱻ��¼��������

	ServerController(const Options& options,
		std::shared_ptr<ReconstructionManager> reconstruction_manager,
		SparseScene* dstScene //�������ڼ�¼�洢Ŀ��λ�õ�
	) :
		AutomaticReconstructionController(options, reconstruction_manager)
	{
		this->dstScene = dstScene;
	}

	//��¼��������Ϣ
	void recordPointResult(const PointMap& pointMap)
	{
		auto& xyz = dstScene->xyz;
		auto& rgb = dstScene->rgb;
		//��ʼ�������
		xyz.resize(pointMap.size()*3);
		rgb.resize(pointMap.size()*3);
		uint32_t idPoint = 0;
		for (auto& eachPoint : pointMap)
		{
			auto pointHead = static_cast<float*>(&xyz[idPoint * 3]);
			//������xyz����
			auto xyzData = eachPoint.second.xyz.data();
			for (int i = 0; i < 3; ++i) pointHead[i] = xyzData[i];
			//��ȡ��ɫ����
			auto colorData = eachPoint.second.color.data();
			auto rgbHead = static_cast<uint8_t*>(&rgb[idPoint * 3]);
			for (int i = 0; i < 3; ++i) rgbHead[i] = colorData[i];
			++idPoint;
		}
	}

	//��¼�ڲ���Ϣ
	void recordIntrinsic(const CameraMap& cameraMap)
	{
		//��Ҫȷ��ֻ��һ���ڲ�
		MY_ASSERT(cameraMap.size() == 1);
		auto& cameraInfo = cameraMap.begin()->second;
		//�ڲδ洢��Ŀ��λ��
		auto& dstIntrinsic = dstScene->intrinsic;
		dstIntrinsic.push_back(cameraInfo.FocalLengthX());
		dstIntrinsic.push_back(cameraInfo.FocalLengthY());
		dstIntrinsic.push_back(cameraInfo.PrincipalPointX());
		dstIntrinsic.push_back(cameraInfo.PrincipalPointY());
		//��¼ͼƬ��shape
		dstScene->imgShape[0] = cameraInfo.width;
		dstScene->imgShape[1] = cameraInfo.height;
	}

	//��¼�����Ϣ
	void recordExternal(const ImageMap& imageMap)
	{
		auto& dstExternal = dstScene->cameraList;
		dstExternal.reserve(imageMap.size());
		//����ÿ�����
		for (auto& eachImage : imageMap)
		{
			dstExternal.emplace_back();
			auto& newImage = dstExternal.back();
			//��¼��Ԫ��
			auto& rt = eachImage.second.CamFromWorld();
			newImage.rotation[0] = rt.rotation.w();
			newImage.rotation[1] = rt.rotation.x();
			newImage.rotation[2] = rt.rotation.y();
			newImage.rotation[3] = rt.rotation.z();
			newImage.translation[0] = rt.translation.x();
			newImage.translation[1] = rt.translation.y();
			newImage.translation[2] = rt.translation.z();
			//ͼƬ·��
			newImage.imgPath = eachImage.second.Name();
		}
	}

	//��дϡ���ؽ�������
	virtual void RunSparseMapper() override
	{
		IncrementalMapperController mapper(option_manager_.mapper,
			*option_manager_.image_path,
			*option_manager_.database_path,
			reconstruction_manager_);
		mapper.SetCheckIfStoppedFunc([&]() { return IsStopped(); });
		mapper.Run();
		//ȡ���ؽ���������ĵ�����
		auto& pointList = this->reconstruction_manager_->Get(0)->Points3D();
		//��ӡ�ؽ������ĵ����
		std::cout << "point num: " << pointList.size() << std::endl;

		//��ȡ�������
		auto& cameras = this->reconstruction_manager_->Get(0)->Cameras();
		//��ȡ�������
		auto& imageParams = this->reconstruction_manager_->Get(0)->Images();
		//��¼��������Ϣ
		recordPointResult(pointList);
		recordIntrinsic(cameras);
		//��¼�����Ϣ
		recordExternal(imageParams);
	}
};
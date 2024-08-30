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

//camera的容器类型
using CameraMap = std::unordered_map<camera_t, Camera>;
//colmap里面内置的点数据类型
using PointMap = std::unordered_map<point3D_t, Point3D>;
//图片数据的数据类型
using ImageMap = std::unordered_map<image_t, class Image>;
//每个图片里面的点列表
using ViewPointList = std::vector<Point2D>;

//server的controller
//这是用来继承colmap里面的流程控制的
class ServerController : public colmap::AutomaticReconstructionController
{
public:

	SparseScene* dstScene; //最后在结果会被记录在这上面

	ServerController(const Options& options,
		std::shared_ptr<ReconstructionManager> reconstruction_manager,
		SparseScene* dstScene //这是用于记录存储目标位置的
	) :
		AutomaticReconstructionController(options, reconstruction_manager)
	{
		this->dstScene = dstScene;
	}

	//记录点坐标信息
	void recordPointResult(const PointMap& pointMap)
	{
		auto& xyz = dstScene->xyz;
		auto& rgb = dstScene->rgb;
		//初始化点个数
		xyz.resize(pointMap.size()*3);
		rgb.resize(pointMap.size()*3);
		uint32_t idPoint = 0;
		for (auto& eachPoint : pointMap)
		{
			auto pointHead = static_cast<float*>(&xyz[idPoint * 3]);
			//这个点的xyz数据
			auto xyzData = eachPoint.second.xyz.data();
			for (int i = 0; i < 3; ++i) pointHead[i] = xyzData[i];
			//获取颜色数据
			auto colorData = eachPoint.second.color.data();
			auto rgbHead = static_cast<uint8_t*>(&rgb[idPoint * 3]);
			for (int i = 0; i < 3; ++i) rgbHead[i] = colorData[i];
			++idPoint;
		}
	}

	//记录内参信息
	void recordIntrinsic(const CameraMap& cameraMap)
	{
		//需要确保只有一个内参
		MY_ASSERT(cameraMap.size() == 1);
		auto& cameraInfo = cameraMap.begin()->second;
		//内参存储的目标位置
		auto& dstIntrinsic = dstScene->intrinsic;
		dstIntrinsic.push_back(cameraInfo.FocalLengthX());
		dstIntrinsic.push_back(cameraInfo.FocalLengthY());
		dstIntrinsic.push_back(cameraInfo.PrincipalPointX());
		dstIntrinsic.push_back(cameraInfo.PrincipalPointY());
		//记录图片的shape
		dstScene->imgShape[0] = cameraInfo.width;
		dstScene->imgShape[1] = cameraInfo.height;
	}

	//记录外参信息
	void recordExternal(const ImageMap& imageMap)
	{
		auto& dstExternal = dstScene->cameraList;
		dstExternal.reserve(imageMap.size());
		//遍历每个相机
		for (auto& eachImage : imageMap)
		{
			dstExternal.emplace_back();
			auto& newImage = dstExternal.back();
			//记录四元数
			auto& rt = eachImage.second.CamFromWorld();
			newImage.rotation[0] = rt.rotation.w();
			newImage.rotation[1] = rt.rotation.x();
			newImage.rotation[2] = rt.rotation.y();
			newImage.rotation[3] = rt.rotation.z();
			newImage.translation[0] = rt.translation.x();
			newImage.translation[1] = rt.translation.y();
			newImage.translation[2] = rt.translation.z();
			//图片路径
			newImage.imgPath = eachImage.second.Name();
		}
	}

	//重写稀疏重建的流程
	virtual void RunSparseMapper() override
	{
		IncrementalMapperController mapper(option_manager_.mapper,
			*option_manager_.image_path,
			*option_manager_.database_path,
			reconstruction_manager_);
		mapper.SetCheckIfStoppedFunc([&]() { return IsStopped(); });
		mapper.Run();
		//取出重建流程里面的点坐标
		auto& pointList = this->reconstruction_manager_->Get(0)->Points3D();
		//打印重建出来的点个数
		std::cout << "point num: " << pointList.size() << std::endl;

		//获取相机数据
		auto& cameras = this->reconstruction_manager_->Get(0)->Cameras();
		//获取外参数据
		auto& imageParams = this->reconstruction_manager_->Get(0)->Images();
		//记录点坐标信息
		recordPointResult(pointList);
		recordIntrinsic(cameras);
		//记录外参信息
		recordExternal(imageParams);
	}
};
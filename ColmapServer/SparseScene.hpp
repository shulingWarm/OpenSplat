#pragma once

//用于传递colmap计算结果的数据类型
class ColmapCamera
{
public:
	//旋转，用四元数表示的 xyzw
	float rotation[4];
	//平移 xyz
	float translation[3];
	//图片的路径
	std::string imgPath;
};

//用于存储返回结果的接口
class SparseScene
{
public:
	//xyz的点坐标
	std::vector<float> xyz;
	std::vector<uint8_t> rgb;

	//内参信息 依次是焦距和主点
	std::vector<float> intrinsic;
	//图片的shape，这也是内参的一部分
	unsigned imgShape[2];

	//每个相机的外参
	std::vector<ColmapCamera> cameraList;
};
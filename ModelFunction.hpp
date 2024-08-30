#pragma once
#include"model.hpp"

//获取点个数
uint32_t getPointNum(Model& model)
{
	return model.means.size(0);
}

//获取model里面的cpu点坐标
torch::Tensor getCpuPointList(Model& model)
{
	if (model.keepCrs)
	{
		return model.means.cpu() / model.scale + model.translation;
	}
	return model.means.cpu();
}

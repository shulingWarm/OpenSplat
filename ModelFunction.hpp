#pragma once
#include"model.hpp"

//��ȡ�����
uint32_t getPointNum(Model& model)
{
	return model.means.size(0);
}

//��ȡmodel�����cpu������
torch::Tensor getCpuPointList(Model& model)
{
	if (model.keepCrs)
	{
		return model.means.cpu() / model.scale + model.translation;
	}
	return model.means.cpu();
}

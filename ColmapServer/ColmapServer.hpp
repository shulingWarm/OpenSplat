#pragma once

#include<string>
#include<vector>
#include"ColmapServer/SparseScene.hpp"


//调用colmap操作的接口
void callColmap(const std::string& imagePath, const std::string& workspacePath,SparseScene& sparseScene);

#pragma once

#include<string>
#include<vector>
#include"ColmapServer/SparseScene.hpp"


//����colmap�����Ľӿ�
void callColmap(const std::string& imagePath, const std::string& workspacePath,SparseScene& sparseScene);

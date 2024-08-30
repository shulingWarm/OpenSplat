#include "ColmapServer.hpp"
#include <colmap/controllers/option_manager.h>
#include <colmap/util/string.h>
#include<colmap/controllers/automatic_reconstruction.h>
#include"ColmapServer/ServerController.hpp"

//调用colmap操作的接口
void callColmap(const std::string& imagePath, const std::string& workspacePath, SparseScene& sparseScene)
{
    //初始化option
    auto option = AutomaticReconstructionController::Options();
    //指定option里面的图片路径
    option.image_path = imagePath;
    //指定工作路径
    option.workspace_path = workspacePath;
    //不需要做稠密化
    option.dense = false;
    //初始化一个重建的manager
    auto reconstructionManager = std::make_shared<ReconstructionManager>();
    //初始化一个controller
    ServerController controller(option, reconstructionManager, &sparseScene);
    //调用controller的执行
    controller.Start();
    controller.Wait();
}
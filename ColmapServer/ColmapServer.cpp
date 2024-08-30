#include "ColmapServer.hpp"
#include <colmap/controllers/option_manager.h>
#include <colmap/util/string.h>
#include<colmap/controllers/automatic_reconstruction.h>
#include"ColmapServer/ServerController.hpp"

//����colmap�����Ľӿ�
void callColmap(const std::string& imagePath, const std::string& workspacePath, SparseScene& sparseScene)
{
    //��ʼ��option
    auto option = AutomaticReconstructionController::Options();
    //ָ��option�����ͼƬ·��
    option.image_path = imagePath;
    //ָ������·��
    option.workspace_path = workspacePath;
    //����Ҫ�����ܻ�
    option.dense = false;
    //��ʼ��һ���ؽ���manager
    auto reconstructionManager = std::make_shared<ReconstructionManager>();
    //��ʼ��һ��controller
    ServerController controller(option, reconstructionManager, &sparseScene);
    //����controller��ִ��
    controller.Start();
    controller.Wait();
}

# 添加用于计算gaussian角度的kernel
add_library(GaussianViewAngle GaussianViewAngle.cu)

# 链接cuda相关的库
target_link_libraries(GaussianViewAngle PUBLIC cuda)

#添加ColmapServer的库文件
add_library(ColmapServer ColmapServer.cpp)

#链接colmap
target_link_libraries(ColmapServer colmap::colmap)
# CMakeList.txt : CMake project for OpenCV_ColorFilter, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("OpenCV_ColorFilter")



# Add source to this project's executable.
add_executable (OpenCV_ColorFilter "OpenCV_ColorFilter.cpp" "OpenCV_ColorFilter.h")

# TODO: Add tests and install targets if needed.
find_package(OpenCV REQUIRED)
target_link_libraries(OpenCV_ColorFilter ${OpenCV_LIBS})

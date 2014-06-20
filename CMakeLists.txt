# CMakeLists.txt for fisheye calibration using local OpenCV/aruco libraries
#
# By Grace Vesom 

cmake_minimum_required(VERSION 2.8)
project(multi_fisheye_calib)

add_definitions(-DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

get_filename_component(ABS_OpenCV_DIR opencv/install REALPATH)
set(ENV{OpenCV_DIR} ${ABS_OpenCV_DIR})

get_filename_component(ABS_aruco_DIR 3rd_party/aruco/trunk/install REALPATH)
set(ENV{aruco_DIR} ${ABS_aruco_DIR})

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find aruco
find_package(aruco REQUIRED)
include_directories(${aruco_INCLUDE_DIRS})

add_subdirectory(modules)

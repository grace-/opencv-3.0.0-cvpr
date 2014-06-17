# CMakeLists.txt for fisheye calibration using local OpenCV/aruco libraries
#
# By Grace Vesom 

PROJECT(multi_fisheye_calib)

cmake_minimum_required(VERSION 2.8)

ADD_DEFINITIONS(-DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

GET_FILENAME_COMPONENT(ABS_OpenCV_DIR ../../../opencv/install REALPATH)
SET(ENV{OpenCV_DIR} ${ABS_OpenCV_DIR})

GET_FILENAME_COMPONENT(ABS_aruco_DIR ../../../3rd_party/aruco/trunk/install REALPATH)
SET(ENV{aruco_DIR} ${ABS_aruco_DIR})

# Find OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# Find aruco
FIND_PACKAGE(aruco REQUIRED)
INCLUDE_DIRECTORIES(${aruco_INCLUDE_DIRS})

# Project files
FILE(GLOB folder_source *.cpp)
FILE(GLOB folder_header *.h)

ADD_EXECUTABLE(multi_fisheye_calib ${folder_source} ${folder_header})

TARGET_LINK_LIBRARIES(multi_fisheye_calib ${OpenCV_LIBRARIES} ${aruco_LIBRARIES})
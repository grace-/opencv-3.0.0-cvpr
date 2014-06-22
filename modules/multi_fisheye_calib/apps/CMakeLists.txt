# CMakeLists.txt for fisheye calibration using local OpenCV/aruco libraries
#
# By Grace Vesom 

PROJECT(multi_fisheye_calib)

cmake_minimum_required(VERSION 2.8)

# Project files
FILE(GLOB folder_source *.cpp)
FILE(GLOB folder_header *.h)

ADD_EXECUTABLE(multi_fisheye_calib ${folder_source} ${folder_header})

TARGET_LINK_LIBRARIES(multi_fisheye_calib ${OpenCV_LIBRARIES} ${aruco_LIBRARIES})
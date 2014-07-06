# CMakeLists.txt for fisheye calibration using local OpenCV/aruco libraries
#
# By Grace Vesom 

cmake_minimum_required(VERSION 2.8)

ADD_EXECUTABLE(multi_calib_test_video multi_calib_test_video.cpp)
TARGET_LINK_LIBRARIES(multi_calib_test_video ${OpenCV_LIBRARIES} ${aruco_LIBRARIES})

ADD_EXECUTABLE(multi_calib_live multi_calib_live.cpp)
TARGET_LINK_LIBRARIES(multi_calib_live ${OpenCV_LIBRARIES} ${aruco_LIBRARIES})
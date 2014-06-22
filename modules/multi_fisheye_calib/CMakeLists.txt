set(DATA_DIR_PATH \"${CMAKE_CURRENT_SOURCE_DIR}/data/\")
configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/include/aruco_data.h.in"
  "${PROJECT_BINARY_DIR}/include/aruco_data.h")
include_directories(${PROJECT_BINARY_DIR}/include)
add_subdirectory(apps)



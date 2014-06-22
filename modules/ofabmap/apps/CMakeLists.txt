project(fabmap)

find_package(Boost COMPONENTS system filesystem REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

set(LIB_DEPENDENCIES ${OpenCV_LIBRARIES}
                     ${Boost_LIBRARIES} )
add_executable(build_vocab_tree build_vocab_tree.cc)
target_link_libraries(build_vocab_tree ${LIB_DEPENDENCIES})

add_executable(build_training_map build_training_map.cc)
target_link_libraries(build_training_map ${LIB_DEPENDENCIES})

add_executable(fabmap_sample fabmap_sample.cpp)
target_link_libraries(fabmap_sample ${LIB_DEPENDENCIES})

add_executable(log_images log_images.cc)
target_link_libraries(log_images ${LIB_DEPENDENCIES})

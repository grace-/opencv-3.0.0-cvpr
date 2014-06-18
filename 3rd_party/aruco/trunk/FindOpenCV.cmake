# FindOpenCV.cmake -- Find OpenCV in local directory
#
# Before running the following variable must be set:
# OpenCV_DIR = install location of OpenCV (must have run 'make install')
#
# After running the following variables will be set:
#
# OpenCV_LIBRARIES    = list of the opencv libraries
# OpenCV_INCLUDE_DIRS = directory that contains opencv's headers folder
# OpenCV_FOUND        = True

SET(OpenCV_FOUND false)
SET(OpenCV_LIBRARIES "")

SET(OpenCV_DIR ../../../opencv/install)
MESSAGE("-- Looking for OpenCV libraries in " ${OpenCV_DIR})

SET(OpenCV_LIB_DIR ${OpenCV_DIR}/lib)
FILE(GLOB_RECURSE OpenCV_LIBRARIES ${OpenCV_DIR}/lib/*.so)
SET(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}/include)

IF (OpenCV_LIBRARIES)
  SET(OpenCV_FOUND True)
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
  SET(OpenCV_LIBS ${OpenCV_LIBRARIES})
  MESSAGE("-- OpenCV found") 
ELSE()
  MESSAGE("-- OpenCV NOT FOUND")
ENDIF()
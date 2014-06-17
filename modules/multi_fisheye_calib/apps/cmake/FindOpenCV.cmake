# FindOpenCV.cmake -- Find OpenCV in local directory set in the environment
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

MESSAGE("-- Looking for OpenCV libraries in " $ENV{OpenCV_DIR})

FILE(GLOB_RECURSE OpenCV_LIBRARIES $ENV{OpenCV_DIR}/lib/*.so)
SET(OpenCV_INCLUDE_DIRS $ENV{OpenCV_DIR}/include)

IF (OpenCV_LIBRARIES)
  SET(OpenCV_FOUND True)
  INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
  MESSAGE("-- OpenCV found")
ELSE()
  MESSAGE("-- OpenCV NOT FOUND")
ENDIF()
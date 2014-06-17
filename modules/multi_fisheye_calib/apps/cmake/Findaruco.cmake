# Findaruco.cmake -- Find aruco libraries in local directory set in the environment
#
# Before running the following variable must be set:
# aruco_DIR = install location of aruco (must have run 'make install')
#
# After running the following variables will be set:
#
# aruco_LIBRARIES    = list of the opencv libraries
# aruco_INCLUDE_DIRS = directory that contains opencv's headers folder
# aruco_FOUND        = True

SET(aruco_FOUND false)
SET(aruco_LIBRARIES "")

MESSAGE("-- Looking for aruco libraries in " $ENV{aruco_DIR})

FILE(GLOB_RECURSE aruco_LIBRARIES $ENV{aruco_DIR}/lib/*.so)
SET(aruco_INCLUDE_DIRS $ENV{aruco_DIR}/include)

IF (aruco_LIBRARIES)
  SET(aruco_FOUND True)
  INCLUDE_DIRECTORIES(${aruco_INCLUDE_DIRS})
  MESSAGE("-- aruco found")
ELSE()
  MESSAGE("-- aruco NOT FOUND")
ENDIF()
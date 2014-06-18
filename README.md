OpenCV 3.0.0 CVPR Tutorial
=================
This repository contains the code demonstrated at the OpenCV 3.0 Tutorial for CVPR 2014.

## Getting started

### Installing local OpenCV 3.0.0
````
cd opencv
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
make -j<# processors to use> install
````

### Installing local Aruco 
````
cd 3rd_party/aruco/trunk/
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
make -j<# processors to use> install
````

## Apps

### Multiple fisheye camera calibration

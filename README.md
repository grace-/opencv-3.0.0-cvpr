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

### Install
Requires system install of Boost.
````
sudo apt-get install libboost-all-dev

mkdir build
cd build
cmake ..
make -j<# processors to use>
````

### Multiple fisheye camera calibration

````
build/modules/multi_fisheye_calib/apps/./multi_fisheye_calib
````

### OpenFabMap sample

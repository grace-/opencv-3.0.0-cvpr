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
Three important binaries to get live_fabmap working. Samples for running each of them are given below in sequence.
To know the meaning of the parameters, say ./binary --help
For eg. ./build_vocab_tree --help
In general, for both build_vocab_tree and build_training_map more images the better. Vocabulary should have enough
samples to model visual appearance through its words.

Currenly, the features are hardcoded to be SURF detector and SURF descriptor. To change the features, change the features
in all three sources corresponding to the binary.
````
$ cd build/modules/ofabmap/apps
$ ./build_vocab_tree dummy_vocab_dir 1
$ ./build_training_map vocab_big.yml train_images 1
$ ./live_fabmap . train_images 1
````

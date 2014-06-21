#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <cstdio>

using cv::Mat;
using cv::imshow;
using cv::waitKey;

int main(int argc, char *argv[]) {
  // read images
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    printf("Opening the default camera did not succeed\n"); 
    return -1;
  }

  for (;;) {
    Mat frame;
    cap >> frame;
    imshow("input RGB image", frame);
    if (waitKey(30) >= 0)
      break;
  }

  // build vocabulary tree
  //
  // FabMap2 fabmap_obj;
  // std::vector<cv::Mat> query_img_descriptors;
  // fabmap_obj.addTraining(query_img_descriptors);

  return 0;
}

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <boost/filesystem.hpp>

#include <cstdio>

// Based on opencv/samples/cpp/bagofwords_classification.cpp
//
/*
using cv::Mat;
using cv::imshow;
using cv::waitKey;
using cv::Ptr;
*/
using namespace cv;
using std::string;

/*
 Keyboard controls: <ESC>       - quits image capturing mode and starts
                                  vocabulary tree building
                    <SPACE BAR> - saves the image during data collection
                                - moves to next image while showing
                                  keypoints and building a vocabulary tree
 Sample usage:

 ./build_vocab_tree [img_save_dir]

arguments:
   img_save_dir -- directory where vocabulary images are saved
TODO: vocab_file   -- yml file that stores vocabulary
 */
int main(int argc, char *argv[]) {
  string img_save_dir;
  if (argc == 1) {
    img_save_dir = "map/";
  } else if (argc == 2) {
    img_save_dir = string(argv[1]);
    // FIXME: Assumes linux style directory structure
    // need to generalize to windows
    if (img_save_dir[img_save_dir.length() - 1] != '/') {
      img_save_dir += "/";
    }
    boost::filesystem::path dir(img_save_dir);
    if (boost::filesystem::exists(dir)) {
      printf("Sequence directory already exists. Change dir name. Aborting.\n");
      return -1;
    }
    if (boost::filesystem::create_directory(dir)) {
      printf("Directory '%s' created\n", img_save_dir.c_str());
    }
  } else {
    //incorrect arguments
    printf("Usage: build_vocab_tree <img_save_dir>\n");
    return -1;
  }

  // read images from camera
  cv::VideoCapture cap(1);
  if (!cap.isOpened()) {
    printf("Opening the default camera did not succeed\n");
    return -1;
  }

  int framenum = 0;
  for (;;) {
    Mat frame;
    cap >> frame;
    imshow("input RGB image", frame);
    int keyp = waitKey(30);
    if (keyp == 27) {
      printf("Done capturing.\n");
      break;
    } else if (keyp == 32) {
      printf("Saving frame\n");
      string frame_filename;
      frame_filename = img_save_dir + "map_%06d.png";
      cv::imwrite(cv::format(frame_filename.c_str(), framenum), frame);
      framenum++;
    }
  }

  return 0;
}

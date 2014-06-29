/**
 * This code takes images, extracts keypoints and descriptors and clusters them into a vocabulary tree
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

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

static bool WriteVocabulary(const string& filename, const Mat& vocabulary) {
  printf("Saving vocabulary...\n");
  FileStorage fs(filename, FileStorage::WRITE);
  if (fs.isOpened()) {
    fs << "Vocabulary" << vocabulary;
    return true;
  }
  return false;
}


static cv::Mat TrainVocabulary(const string &vocab_dir, int total_frames,
    float descriptor_proportion) {
//If you have fewer descriptors than kvocab_size, it will crash in kmeans. But, set this appropriately large
  const int kvocab_size = 300; //Should be large enough to create a rich vocab relative to your data
  cv::RNG rng(1234);
  Ptr<FeatureDetector> fdetector(new DynamicAdaptedFeatureDetector(
            AdjusterAdapter::create("SURF"), 100, 130, 5));
  Ptr<DescriptorExtractor> dextractor = 
    // DescriptorExtractor::create("SIFT");
    (new SurfDescriptorExtractor(1000, 4, 2, false, true));

  TermCriteria terminate_criterion; 
  terminate_criterion.epsilon = FLT_EPSILON;
  BOWKMeansTrainer bow_trainer(kvocab_size, terminate_criterion, 3, KMEANS_PP_CENTERS);

  for (int i = 0; i < total_frames; i++) {
    Mat vocab_frame;
    // TODO: Hardcoded prefix
    string frame_filename = vocab_dir + "vocab_%06d.png";
    vocab_frame = cv::imread(cv::format(frame_filename.c_str(), i));

    std::vector<cv::KeyPoint> image_keypoints;
    fdetector->detect(vocab_frame, image_keypoints);
    Mat image_descriptors;
    dextractor->compute(vocab_frame, image_keypoints, image_descriptors);
#if 1
    // TODO: pull out of the loop. Using it here is inefficient
    CV_Assert( dextractor->descriptorType() == CV_32FC1 );
    const int elem_size = CV_ELEM_SIZE(dextractor->descriptorType());
    const int desc_byte_size = dextractor->descriptorSize() * elem_size;
    const long int bytes_in_MB = 1048576;
    const long int memory_use = 100000;

    // Total number of descs to use for training.
    const long int max_desc_count = (memory_use * bytes_in_MB) / desc_byte_size;
    if (bow_trainer.descripotorsCount() > max_desc_count) {
      printf("Breaking due to full memory ( descriptors count =");
      printf("%d; descriptor size in bytes = %d; all used memory = %d\n",
          bow_trainer.descripotorsCount(), desc_byte_size,
          bow_trainer.descripotorsCount()*desc_byte_size);
      break;
    }

    if (!image_descriptors.empty()) {
      int desc_count = image_descriptors.rows;

      // Extracting desc_proportion descriptors from the image
      int descs_to_extract = static_cast<int>(descriptor_proportion
                               *static_cast<float>(desc_count));
      vector<char> used_mask( desc_count, false );
      fill( used_mask.begin(), used_mask.begin() + descs_to_extract, true );
      for( int i = 0; i < desc_count; i++ ) {
        int i1 = rng(desc_count), i2 = rng(desc_count);
        char tmp = used_mask[i1]; used_mask[i1] = used_mask[i2]; used_mask[i2] = tmp;
      }

      for(int i = 0; i < desc_count; i++) {
        if(used_mask[i] && bow_trainer.descripotorsCount() < max_desc_count)
          bow_trainer.add(image_descriptors.row(i));
      }
    }
#endif
    drawKeypoints(vocab_frame, image_keypoints, vocab_frame);
    cv::imshow("image read in", vocab_frame);
    int keyp = cv::waitKey(0);
    if (keyp == 27) {
      printf("Done building vocabulary tree\n");
      break;
    }
  }

  Mat vocabulary = bow_trainer.cluster();
  printf("Done building vocabulary tree\n");
  return vocabulary; 
}

// directory creation utility function
static void make_img_directory(string *img_save_dir) {
  // FIXME: Assumes linux style directory structure
  // need to generalize to windows
  if ((*img_save_dir)[img_save_dir->length() - 1] != '/') {
    *img_save_dir += "/";
  }
  boost::filesystem::path dir(*img_save_dir);
  if (boost::filesystem::exists(dir)) {
    printf("Sequence directory already exists. Change dir name. Aborting. \n");
    exit(-1);
  }
  if (boost::filesystem::create_directory(dir)) {
    printf("Directory '%s' created\n", img_save_dir->c_str());
  }
}

void help()
{
	printf("\n"
" Keyboard controls: <ESC>       - quits image capturing mode and starts\n"
"                                  vocabulary tree building \n"
"                    <SPACE BAR> - saves the image during data collection\n"
"                                - moves to next image while showing\n"
"                                  keypoints and building a vocabulary tree \n"
" Sample usage:\n"
"\n"
" ./build_vocab_tree [<img_save_dir>] [<camera-deviceid>]\n"
"\n"
"arguments:\n"

"   <img_save_dir>    -- directory where vocabulary images are saved\n"
"   <camera-deviceid> -- should be non-negative integer. default camera is 0\n\n"
);
}

//
// Keyboard controls: <ESC>       - quits image capturing mode and starts
//                                  vocabulary tree building 
//                    <SPACE BAR> - saves the image during data collection
//                                - moves to next image while showing
//                                   keypoints and building a vocabulary tree 
// Sample usage:
//
// ./build_vocab_tree [<img_save_dir>] [<camera-deviceid>]
//
// arguments:
//   <img_save_dir>    -- directory where vocabulary images are saved
//   <camera-deviceid> -- should be non-negative integer. default camera is 0
//
//   TODO: <vocab_file>   -- yml file that stores vocabulary

int main(int argc, char *argv[]) {
  string img_save_dir;
  int cam_deviceid = 0;
  if (argc == 1) {
    help();
    return(0);
  } else if (argc == 2) {
    if (!strcmp(argv[1],"-h") || !strcmp(argv[1],"--help")) {
      help();
      return(0);
    }
    img_save_dir = string(argv[1]);
    make_img_directory(&img_save_dir);
  } else if (argc == 3) {
    if (!strcmp(argv[1],"-h") || !strcmp(argv[1],"--help")) {
      help();
      return(0);
    }
    img_save_dir = string(argv[1]);
    make_img_directory(&img_save_dir);
    cam_deviceid = boost::lexical_cast<int>(argv[2]);
  } else {
    //incorrect arguments
    help();
    return -1; 
  }

  // read images from camera
  cv::VideoCapture cap(cam_deviceid);
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
    if(keyp == 'h')
      help();
    else if (keyp == 27) {
      printf("Done capturing.\n");
      break;
    } else if (keyp == 32) {
      printf("Saving frame\n");
      string frame_filename;
      frame_filename = img_save_dir + "vocab_%06d.png";
      cv::imwrite(cv::format(frame_filename.c_str(), framenum), frame);
      framenum++;
    }
  }

  destroyWindow("input RGB image");
  const float kdescriptor_proportion = 0.7;
  cv::Mat vocabulary = TrainVocabulary(img_save_dir, framenum - 1, kdescriptor_proportion);
  string filename = "vocab_big.yml";
  if (!WriteVocabulary(filename, vocabulary)) {
    printf("Error: file %s cannot be opened to write\n", filename.c_str());
    exit(-1);
  }

  return 0;
}

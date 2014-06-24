/**
 * After running build_vocab_tree, this code will turn the resulting descriptors
 * into a bag of words (BOW)
 */
 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <cstdio>

using namespace cv;
using std::string;

static cv::Mat LoadVocabFile(const std::string &vocab_file) {
  //Hardcoded vocab data dir
  string data_dir = "./";
  string vocab_path = data_dir + vocab_file; 

  //load/generate vocab
  printf("Loading Vocabulary: %s\n", vocab_path.c_str()); 
  FileStorage fs;
  fs.open(vocab_path.c_str(), FileStorage::READ);

  Mat vocab;
  fs["Vocabulary"] >> vocab;
  if (vocab.empty()) { 
    printf("Vocabulary not found\n");
    return cv::Mat();
  }
  printf("Vocabulary loaded\n");
  fs.release();

  return vocab;
}

static bool WriteTrainingData(string filename, 
    const vector<Mat> visualword_training) {
  cv::Mat training_mapdata;
  vconcat(visualword_training, training_mapdata); 
  printf("Saving training data...\n");
  FileStorage fs(filename, FileStorage::WRITE);
  if (fs.isOpened()) {
    fs << "BOWImageDescs" << training_mapdata;
    return true;
  }
  return false;
}

static void help()
{
  printf("\n\n"
  "Usage:\n"
  "   ./build_training_map <vocab file name> <directory to put logged images into> [<camera device-id>]\n"
  "<camera device-id> is optional - should be a non-negative integer. default camera is zero"
  "\n"
  "Press space collect each image to put into the map\n"
  "ESC will end\n"
     "\n\n");
}

static void make_img_directory(string *img_save_dir) {
  if ((*img_save_dir)[img_save_dir->length() - 1] != '/') {
      *img_save_dir += "/";
    }
    boost::filesystem::path dir(*img_save_dir);
    if (boost::filesystem::exists(dir)) {
	  printf("Sequence directory already exists. Change dir name. Aborting.\n");
	  exit(-1);
    }
    if (boost::filesystem::create_directory(dir)) {
	  printf("Directory '%s' created\n", img_save_dir->c_str());
    }
}

int main(int argc, char *argv[]) {
  string vocab_file;
  string img_save_dir;
  int cam_deviceid = 0;
  if (argc == 1) {
    help();
    return 0;
  } else if (argc == 2) {
    if(!strcmp(argv[1],"-h") || !strcmp(argv[1],"--help")) {
      help();
      return(0);
    }
    vocab_file = string(argv[1]);
    img_save_dir = "./";
  } else if (argc == 3) {
    vocab_file = string(argv[1]);
    img_save_dir = string(argv[2]);
    make_img_directory(&img_save_dir);
  } else if (argc == 4) {
    vocab_file = string(argv[1]);
    img_save_dir = string(argv[2]);
    make_img_directory(&img_save_dir);
    cam_deviceid = boost::lexical_cast<int>(argv[3]);
  }


  cv::Mat vocabulary = LoadVocabFile(vocab_file);
  if (vocabulary.empty()) {
    printf("aborting\n");
    exit(-1);
  }

  Ptr<FeatureDetector> detector =
    new DynamicAdaptedFeatureDetector(
      AdjusterAdapter::create("SURF"), 100, 130, 5);
  Ptr<DescriptorExtractor> extractor =
   // DescriptorExtractor::create("SIFT");
     new SurfDescriptorExtractor(1000, 4, 2, false, true);
  Ptr<DescriptorMatcher> matcher =
    DescriptorMatcher::create("FlannBased");

  BOWImgDescriptorExtractor bide(extractor, matcher); 
  bide.setVocabulary(vocabulary);
  vector<Mat> visualword_training;

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
    if (keyp == 27) {
      printf("Done capturing.\n");
      break;
    } else if (keyp == 'h')
    {
		help();
	} 
    else if (keyp == 32) {
      Mat visualword_descriptor;
      vector<KeyPoint> kpts;
      detector->detect(frame, kpts);
      bide.compute(frame, kpts, visualword_descriptor);
      
      visualword_training.push_back(visualword_descriptor);

      printf("Saving frame\n");
      string frame_filename;
      
      // TODO: Hardcoded prefix
      frame_filename = img_save_dir + "training_%06d.png";
      cv::imwrite(cv::format(frame_filename.c_str(), framenum), frame);
      framenum++;
    }
  }
  
  string filename = "training_data.yml";
  if (!WriteTrainingData(filename, visualword_training)) {
    printf("Error: file %s cannot to opened to write\n", filename.c_str());
    exit(-1);
  }

  return 0;
}

/* this file is to demonstrate live FAB-MAP matching 
 * fabmap approach is based on the openfabmap implementation
 * @author - Prasanna Krishnasamy (pras.bits@gmail.com)
 */

#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <cstdio>

using namespace cv;
using namespace std;

static void help() {
  printf("Sample usage:\n"
    "live_fabmap <ymlfile directory> <map data dir> <cameraid>\n\n"
    "  <ymlfile directory> where vocab and chow-liu tree ymls are stored\n"
    "  <map data dir>      where images for map are stored\n"
    "  <cameraid>          non-negative integer indicating cameraid\n"
    );
}

int main(int argc, char * argv[]) {
  printf("This sample program demonstrates live FAB-MAP image matching "
         "algorithm\n\n");

  string dataDir, mapDir;
  int cam_deviceid = 0;
  if (argc == 1) {
    //incorrect arguments
    help();
    return (-1);
  } else if (argc == 2) {
    if(!strcmp(argv[1],"-h") || !strcmp(argv[1],"--help")) {
      help();
      return(-1);
    }
    dataDir = string(argv[1]);
    dataDir += "/";
  } else if (argc == 3) {
    if(!strcmp(argv[1],"-h") || !strcmp(argv[1],"--help")) {
      help();
      return(-1);
    }
    dataDir = string(argv[1]);
    dataDir += "/";
    mapDir = string(argv[2]);
    mapDir += "/";
  } else if (argc == 4) {
    if(!strcmp(argv[1],"-h") || !strcmp(argv[1],"--help")) {
      help();
      return(-1);
    }
    dataDir = string(argv[1]);
    dataDir += "/";
    mapDir = string(argv[2]);
    mapDir += "/";
    cam_deviceid = boost::lexical_cast<int>(argv[3]);
  } else {
    //incorrect arguments
    help();
    return -1;
  }

  FileStorage fs;

  //load/generate vocab
  cout << "Loading Vocabulary: " <<
    dataDir + string("vocab_big.yml") << endl << endl;
  fs.open(dataDir + string("vocab_big.yml"), FileStorage::READ);
  Mat vocab;
  fs["Vocabulary"] >> vocab;
  if (vocab.empty()) {
    cerr << "Vocabulary not found" << endl;
    return -1;
  }
  fs.release();

  //load/generate training data

  cout << "Loading Training Data: " <<
      dataDir + string("training_data.yml") << endl << endl;
  fs.open(dataDir + string("training_data.yml"), FileStorage::READ);
  Mat trainData;
  fs["BOWImageDescs"] >> trainData;
  if (trainData.empty()) {
    cerr << "Training Data not found" << endl;
    return -1;
  }
  fs.release();

  //create Chow-liu tree
  printf("Making Chow-Liu Tree from training data\n");
  of2::ChowLiuTree treeBuilder;
  treeBuilder.add(trainData);
  Mat tree = treeBuilder.make();

  //generate test data
  printf("Extracting Test Data from images\n");
  Ptr<FeatureDetector> detector =
      new DynamicAdaptedFeatureDetector(
      AdjusterAdapter::create("SURF"), 100, 130, 5);
  Ptr<DescriptorExtractor> extractor =
      // DescriptorExtractor::create("SIFT");
      new SurfDescriptorExtractor(1000, 4, 2, false, true);
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create("FlannBased");

  BOWImgDescriptorExtractor bide(extractor, matcher);
  bide.setVocabulary(vocab);

  vector<string> imageNames;
  namespace bfs = boost::filesystem;

  bfs::path p(mapDir);
  bfs::directory_iterator end_iter;
  if (bfs::is_directory(p)) {
    for (bfs::directory_iterator dir_iter(p); dir_iter != end_iter;
        ++dir_iter) {
      if (bfs::is_regular_file(dir_iter->status()) ) {
        printf("%s\n", dir_iter->path().c_str());
        imageNames.push_back(dir_iter->path().native());
      }
    }
  }

  std::sort(imageNames.begin(), imageNames.end());

  printf("Number of image files being processed is %lu\n", imageNames.size());

  Mat testData;
  Mat frame;
  Mat bow;
  vector<KeyPoint> kpts;
  vector<Mat> map_images;

  for(size_t i = 0; i < imageNames.size(); i++) {
    cout << imageNames[i] << endl;
    frame = imread(imageNames[i]);
    if (frame.empty()) {
      printf("Test images not found\n");
      return -1;
    }

    map_images.push_back(frame);
    detector->detect(frame, kpts);
    bide.compute(frame, kpts, bow);
    testData.push_back(bow);

    // drawKeypoints(frame, kpts, frame);
    // imshow(imageNames[i], frame);
    // waitKey(10);
  }

  //run fabmap
  printf("Running FAB-MAP algorithm\n\n");
  Ptr<of2::FabMap> fabmap;

  fabmap = new of2::FabMap2(tree, 0.39, 0, of2::FabMap::SAMPLED |
                            of2::FabMap::CHOW_LIU);
  fabmap->addTraining(trainData);

  vector<of2::IMatch> matches;
  fabmap->compare(testData, matches, true);
  
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

    Mat visualword_descriptor;
    vector<KeyPoint> kpts;
    detector->detect(frame, kpts);

    drawKeypoints(frame, kpts, frame);
    imshow("input RGB image", frame);
    bide.compute(frame, kpts, visualword_descriptor);

    vector<of2::IMatch> matches;
    fabmap->compare(visualword_descriptor, matches);

    vector<of2::IMatch>::iterator l;

    double max_prob = 0;
    int max_img_idx = -1;
    printf("---------------------\n");
    for (l = matches.begin(); l != matches.end(); l++) {
      printf("prob is %f\n", l->match);
      if (max_prob < l->match) {
        max_prob = l->match;
        max_img_idx = l->imgIdx;
      }
    }

    // cout << " Max idx is " << max_img_idx << endl;
    cv::Mat disp_image;
    if (max_img_idx == -1) {
      disp_image = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
    } else {
      disp_image = map_images[max_img_idx];
    }
    cv::imshow("Closest Map image", disp_image);
    int keyp = cv::waitKey(10);
    if (keyp == 27) {
      printf("Done capturing.\n");
      break;
    } else if (keyp == 'h') {
      help();
    }
  }
  return 0;
}

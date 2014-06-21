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

static bool WriteVocabulary( const string& filename, const Mat& vocabulary ) {
    printf("Saving vocabulary...\n");
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() ) {
        fs << "vocabulary" << vocabulary;
        return true;
    }
    return false;
}


static cv::Mat TrainVocabulary(const string &vocab_dir, int total_frames,
    float descriptor_proportion) {
  const int kvocab_size = 100;
  cv::RNG rng(1234);
  Ptr<FeatureDetector> fdetector(new DynamicAdaptedFeatureDetector(
            AdjusterAdapter::create("STAR"), 130, 150, 5));
  // Ptr<DescriptorExtractor> dextractor(new BriefDescriptorExtractor(32));
  Ptr<DescriptorExtractor> dextractor(new SurfDescriptorExtractor(1000, 4, 2, false, true));

  TermCriteria terminate_criterion; 
  terminate_criterion.epsilon = FLT_EPSILON;
  BOWKMeansTrainer bow_trainer(kvocab_size, terminate_criterion, 3, KMEANS_PP_CENTERS);

  for (int i = 0; i < total_frames; i++) {
    Mat vocab_frame;
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

/*
 Sample usage:

 ./build_vocab_tree [img_save_dir]

arguments:
   img_save_dir -- directory where vocabulary images are saved
 */
int main(int argc, char *argv[]) {
  string img_save_dir;
  if (argc == 1) {
    img_save_dir = "fabmap/";
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

  // read images
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
      frame_filename = img_save_dir + "vocab_%06d.png";
      cv::imwrite(cv::format(frame_filename.c_str(), framenum), frame);
      framenum++;
    }
  }

  const float kdescriptor_proportion = 0.7;
  cv::Mat vocabulary = TrainVocabulary(img_save_dir, framenum - 1, kdescriptor_proportion);
  string filename = "vocab_big.yml";
  if (!WriteVocabulary(filename, vocabulary)) {
    printf("Error: file %s cannot be opened to write\n");
    exit(-1);
  }
  // build vocabulary tree
  //
  // FabMap2 fabmap_obj;
  // std::vector<cv::Mat> query_img_descriptors;
  // fabmap_obj.addTraining(query_img_descriptors);

  return 0;
}

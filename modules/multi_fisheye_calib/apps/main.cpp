#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <aruco/aruco.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>

void Autobalance(cv::Mat* im);
bool DetectArucoMarkers(
    cv::Mat* im,
    std::vector<cv::Point2f>* aruco_marker_detects_2f,
    std::vector<cv::Point3f>* aruco_marker_detects_3f);

struct Camera {
  cv::Matx33f K;
  cv::Matx<double, 4, 1> distorsion;
  cv::Size size;
};

aruco::MarkerDetector aruco_detector;
aruco::CameraParameters aruco_camera;
std::vector<int> aruco_marker_map;
std::vector<std::vector<cv::Point3f> > aruco_board_3f;
std::vector<cv::Point3f> aruco_board_pts;

cv::RNG rng;

std::vector<std::vector<int> > num_detected_markers;
std::vector<std::vector<std::vector<cv::Point2f> > > marker_pnts_;
std::vector<std::vector<std::vector<cv::Point3f> > > marker_pnts_map_;
std::vector<std::vector<cv::Mat> > rvecs_;
std::vector<std::vector<cv::Mat> > tvecs_;

float square;
bool first_calib;
int num_markers_calc;

bool AUTOBALANCE = true;
bool DRAWMARKERS = true;
bool DUMPIMAGES =  true;
bool READIMAGES = false;
std::string path_write = "../data/images/test1/";

int main(int argc, char *argv[]) {
  
  if (argc < 2) { 
    std::cerr << "Usage: ./fisheye_calibration <Aruco square size> "
              << "<first device #> \n";
    return -1;
  }
  
  /****************************************************************************/
  /* Initialize video streams and camera objects                              */
  /****************************************************************************/
  cv::VideoCapture video_stream;
  Camera camera;
  int video_num;
  video_num = atoi(argv[2]);

  if (!READIMAGES) {

    video_stream.open(CV_CAP_FIREWIRE + video_num);  
    video_stream.set(CV_CAP_PROP_RECTIFICATION, 1);
    video_stream.set(CV_CAP_PROP_FPS, 10);
    if (!video_stream.isOpened()) {
      std::cerr << "Cannot open firewire video stream: "
                << video_num << std::endl;
      return -1;
    } else {
      std::cout << "Video stream /dev/fw" << video_num << " opened.\n";
    }
    camera.size = cv::Size(0, 0);
    camera.K = cv::Matx33f::eye();
    camera.distorsion = cv::Matx<double, 4, 1>::zeros();
    
  }
  /****************************************************************************/
  /* Load Aruco board configuration and detection data structures             */
  /****************************************************************************/
  square = atof(argv[1]);
  {
    aruco::BoardConfiguration aruco_board_config;
    aruco_board_config.readFromFile("../data/aruco/aruco20x10_meters.yml");
    //    aruco_board_config.readFromFile("../data/aruco/aruco6x4_meters.yml");

    aruco_marker_map.resize(1024, -1); // in place of hashtable  
    int num_markers = aruco_board_config.size();
    aruco_board_3f.reserve(num_markers);
    std::vector<cv::Point3f> aruco_marker_3f(4);

    cv::Point3f minpt(1000, 1000, 0);
    for (int i = 0; i < num_markers; ++i) {
      aruco_marker_map[aruco_board_config[i].id] = i;
      for (int j = 0; j < 4; ++j ) {              
        aruco_marker_3f[j] = cv::Point3f(aruco_board_config[i][j]) * square;
        if ((aruco_marker_3f[j].x < minpt.x) &&
            (aruco_marker_3f[j].y < minpt.y)) minpt = aruco_marker_3f[j];
        // aruco_marker_3f[j].x = aruco_board_config[i][j].x * square;
        // aruco_marker_3f[j].y = aruco_board_config[i][j].y * square;
        // aruco_marker_3f[j].z = aruco_board_config[i][j].z * square;
      }
      aruco_board_3f.push_back(aruco_marker_3f);
    } 
    for (int i = 0; i < aruco_board_3f.size(); ++i) 
      for (int j = 0; j < 4; ++j) 
        aruco_board_3f[i][j] -= minpt;
   
    aruco_detector.enableErosion(false);
    //    aruco_detector.setCornerRefinementMethod(aruco::MarkerDetector::HARRIS);
    // aruco_detector.setCornerRefinementMethod(aruco::MarkerDetector::NONE);
    aruco_detector.setCornerRefinementMethod(aruco::MarkerDetector::SUBPIX);
    // aruco_detector.setCornerRefinementMethod(aruco::MarkerDetector::LINES);
  }

  std::vector<cv::Point2f> aruco_markers_detected_2f;
  std::vector<cv::Point3f> aruco_markers_detected_3f;
  std::vector<std::vector<cv::Point2f> > all_rig_detections_2f;
  std::vector<std::vector<cv::Point3f> > all_rig_detections_3f;

  /****************************************************************************/
  /* Initialize camera calibration data structures                            */
  /****************************************************************************/


  /****************************************************************************/
  /* Start reading video stream                                               */
  /****************************************************************************/
  std::string window_name = "Video stream";
  cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE); 
  cv::Mat frame;

  bool FOV_intersect;
  char keypress;
  int num_calib = 0;
  char num_calib_buffer[10];
  bool sane = false;
  first_calib = true;

  cv::Mat debug_im;

  while(video_stream.read(frame)) {
    // Initalize rig with first camera
    if (!sane) camera.size = frame.size();
    debug_im = frame.clone();
    FOV_intersect = DetectArucoMarkers(&(frame),
                                       &aruco_markers_detected_2f,
                                       &aruco_markers_detected_3f);
    if (!sane) sane = true;

    // Display status in window
    sprintf(num_calib_buffer, "%d", num_calib);
    std::string str(num_calib_buffer);
    cv::putText(frame, "# calibration captures = " + str, cv::Point(50, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 8);
    cv::putText(frame, "# calibration captures = " + str, cv::Point(50, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3);

    // Send image to display
    cv::imshow(window_name, frame);
    keypress = cv::waitKey(30);

    // Esc key press -- EXIT
    if (keypress == 27) break;
    // SPACEBAR key press -- Collect data
    if (keypress == 32) {      
      if (FOV_intersect) {
        // Aruco markers detected in every camera
        //        if (aruco_markers_detected_3f.size() == 23) {
          all_rig_detections_2f.push_back(aruco_markers_detected_2f);
          all_rig_detections_3f.push_back(aruco_markers_detected_3f);
          // if (DUMPIMAGES) {
          //   char num[5];
          //   sprintf(num, "%d", im_num++);
          //   cv::imwrite(path_write + "im" + num + ".png", debug_im);          
          // }          
          num_calib++;
       
      } else {
        std::cout << "Markers not seen by every camera.\n";
      }
    }
    // RETURN key press -- Calibrate
    if (keypress == 10) {
      if (all_rig_detections_2f.size() == 0) {
        std::cout << "Cannot calibrate!  No data collected.\n";
      } else {
        std::cout << "Calibrating from " << num_calib << " "
                  << "captured collections.\n";
      
        if (!first_calib) {       
          cv::Matx33d K = cv::Matx33d::eye();// = cv::Mat::eye(3, 3, CV_64F);
          cv::Vec4d D(0,0,0,0);   
          std::vector<cv::Vec3d> rvec;
          std::vector<cv::Vec3d> tvec;

          int flag = 0;
          flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
          flag |= cv::fisheye::CALIB_CHECK_COND;
          flag |= cv::fisheye::CALIB_FIX_SKEW;

          for ( int i = 0; i < all_rig_detections_3f.size(); ++i ) {
            std::cout << "rig detections " << i << " = "
                      << all_rig_detections_3f[i].size() << std::endl;
          }

          double rms = cv::fisheye::calibrate(all_rig_detections_3f,
                                              all_rig_detections_2f,
                                              camera.size,
                                              K, D, rvec, tvec,
                                              flag, cv::TermCriteria(3, 20, 1e-6));
          std::cout << "fisheye rms = " << rms << std::endl;
          std::cout << "fisheye camera K = " << K << std::endl;
          std::cout << "fisheye camera D = " << cv::Mat(D) << std::endl;
          std::cout << "fisheye camera rvecs = " << rvec.size() << std::endl;
          std::cout << "fisheye camera tvecs = " << tvec.size() << std::endl;
        }
        { 
          cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
          cv::Mat D = cv::Mat::ones(6,1,CV_64F);
          std::vector<cv::Mat> rvec;
          std::vector<cv::Mat> tvec;       
          
          double rms = cv::calibrateCamera(all_rig_detections_3f,
                                    all_rig_detections_2f,
                                    camera.size,
                                    K, D, rvec, tvec);
          std::cout << "camera K = " << K << std::endl;
          std::cout << "camera D = " << D << std::endl;
          std::cout << "camera rvecs = " << rvec.size() << std::endl;
          std::cout << "camera tvecs = " << tvec.size() << std::endl;
          if (first_calib) aruco_camera = aruco::CameraParameters(K, D, camera.size);
        }
        
        if (first_calib) { // TEST
          const int n_images = 34;
          std::vector<std::vector<cv::Point2d> > imagePoints(n_images);
          std::vector<std::vector<cv::Point3d> > objectPoints(n_images);

          const std::string folder = std::string("/home/gvesom/Downloads/opencv_extra/testdata/cv/cameracalibration/fisheye/calib-3_stereo_from_JY/");
          cv::FileStorage fs_left(folder + "left.xml", cv::FileStorage::READ);
          CV_Assert(fs_left.isOpened());
          for(int i = 0; i < n_images; ++i)
            fs_left[cv::format("image_%d", i )] >> imagePoints[i];
          fs_left.release();
          
          cv::FileStorage fs_object(folder + "object.xml", cv::FileStorage::READ);
          CV_Assert(fs_object.isOpened());
          for(int i = 0; i < n_images; ++i)
            fs_object[cv::format("image_%d", i )] >> objectPoints[i];
          fs_object.release();
          
          int flag = 0;
          flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
          flag |= cv::fisheye::CALIB_CHECK_COND;
          flag |= cv::fisheye::CALIB_FIX_SKEW;
          
          cv::Matx33d K;
          cv::Vec4d D;
          std::vector<cv::Vec3d> rvec;
          std::vector<cv::Vec3d> tvec;
          
          for ( int i = 0; i < n_images; ++i ) {
            for ( int j = 0; j < objectPoints[i].size(); ++j ) {
              std::cout << "(" << i << "," << j << ") = "
                        << objectPoints[i][j] << " --> "
                        << imagePoints[i][j] << std::endl;
            }
          }

          double rms = 
            cv::fisheye::calibrate(objectPoints, imagePoints, cv::Size(1280,800), K, D,
                                   rvec, tvec, flag, cv::TermCriteria(3, 20, 1e-6));
          std::cout << "camera TEST rms = " << rms << std::endl;
          std::cout << "camera TEST K = " << K << std::endl;
          std::cout << "camera TEST D = " << D << std::endl;
          std::cout << "camera TEST rvecs = " << rvec.size() << std::endl;
          std::cout << "camera TEST tvecs = " << tvec.size() << std::endl;
          first_calib = false;      
        }
        all_rig_detections_3f.clear();
        all_rig_detections_2f.clear();
        num_calib = 0;
      }
    }
    
  }

  /****************************************************************************/
  /* Close down                                                               *
  /****************************************************************************/
  video_stream.release();
  return 0;
}

// This function uses the Aruco library to detect any Aruco markers in image *im
// and collects the corner image locations of the markers detected and the
// corresponding 3F coordinates from the original board configuration. 
bool DetectArucoMarkers(
    cv::Mat* im,
    std::vector<cv::Point2f>* aruco_marker_detects_2f,
    std::vector<cv::Point3f>* aruco_marker_detects_3f) {
  cv::Mat im_copy;
  if (!(im == NULL)) {
    cv::cvtColor(*im, im_copy, CV_BGR2GRAY);     
    if (AUTOBALANCE) Autobalance(&im_copy);
    aruco_marker_detects_2f->clear();
    aruco_marker_detects_3f->clear();

    std::vector<aruco::Marker> markers;
    if (first_calib) {
      aruco_detector.detect(im_copy, markers);    
    } else aruco_detector.detect(im_copy, markers, aruco_camera);
    if (markers.size() == 0) {
      return false;
    } else {
      int marker_idx, i_ = 0;
      //      aruco_marker_detects_2f->reserve(markers.size() * 4);
      //      aruco_marker_detects_2f->reserve(markers.size());
      aruco_marker_detects_2f->reserve(N);
      aruco_marker_detects_3f->reserve(aruco_marker_detects_2f->size());     
      //      aruco_marker_detects_2f->resize(markers.size() * 4);
      //      aruco_marker_detects_3f->resize(aruco_marker_detects_2f->size());     
      //      for (int i = 0; i < markers.size(); ++i) {
      for (int i = 0; i < N; ++i) {
        int u = rng.uniform(0, markers.size());
        marker_idx = aruco_marker_map[markers[u].id];
        if (marker_idx >=0) {
          int v = rng.uniform(0, 4);
          aruco_marker_detects_2f->push_back(cv::Point2f(markers[u][v].x,
                                                         markers[u][v].y));
          aruco_marker_detects_3f->push_back(aruco_board_3f[aruco_marker_map[markers[u].id]][v]);
          //          if (DRAWMARKERS) markers[i].draw(*im, cv::Scalar(0, 0, 255), 2);
          //          for (int j = 0; j < 1; ++j) {
          //            aruco_marker_detects_2f->push_back(cv::Point2f(markers[i][j].x,
          //                                                           markers[i][j].y));
          //            aruco_marker_detects_3f->push_back(aruco_board_3f[aruco_marker_map[markers[i].id]][j]);
            // (*aruco_marker_detects_2f)[i_ + j] =
            //   cv::Point2f(markers[i][j]);
            // (*aruco_marker_detects_3f)[i_ + j] =
            //   cv::Point3f(aruco_board_3f[aruco_marker_map[markers[i].id]][j]);
            //          }
          i_ = i + 4;
        }
      }
      if (DRAWMARKERS) 
        for (int i =0; i < markers.size(); ++i) 
          markers[i].draw(*im, cv::Scalar(0,0,255),2);
      return true;
    }
  } else return false;
}  

// This function normalizes the image intensities over [0, 256)
void Autobalance(cv::Mat* im) {
  if (!(im == NULL)) {
    double min_px, max_px;
    cv::minMaxLoc(*im, &min_px, &max_px);
    double d = 255.0f/(max_px - min_px);
    int n = im->rows * im->cols;
    for (int i = 0; i < n; ++i) {
      im->data[i] = static_cast<unsigned char>(
          (static_cast<double>(im->data[i]) - min_px) * d);
    }
  }
}

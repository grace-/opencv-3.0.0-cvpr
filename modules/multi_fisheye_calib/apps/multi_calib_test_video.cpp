#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <aruco/aruco.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <map>

#include <aruco_data.h>

void Autobalance(cv::Mat* im);
bool DetectArucoMarkers(
    cv::Mat* im,
    std::vector<cv::Point2f>* aruco_marker_detects_2f,
    std::vector<cv::Point3f>* aruco_marker_detects_3f,
    std::vector<int>* aruco_marker_detects_id);

struct Camera {
  cv::Matx33f K;
  cv::Matx<double, 4, 1> distorsion;
  cv::Size size;
};

aruco::MarkerDetector aruco_detector;
std::map<int, int> aruco_marker_map;
std::vector<cv::Point3f> aruco_board_3f;

std::vector<std::vector<int> > num_detected_markers;
std::vector<std::vector<std::vector<cv::Point2f> > > marker_pnts_;
std::vector<std::vector<std::vector<cv::Point3f> > > marker_pnts_map_;
std::vector<std::vector<cv::Mat> > rvecs_;
std::vector<std::vector<cv::Mat> > tvecs_;

float square;

bool AUTOBALANCE = true;
bool DRAWMARKERS = true;

int main(int argc, char *argv[]) {
  
  if (argc < 2) { 
    std::cerr << "Usage: ./multi_calib_test_video " 
              << "<true fisheye == 1 or perspective warp = 0> ";
    return -1;
  }

  bool use_fisheye_model = atoi(argv[1]);
  std::string camera_model_type;
  if (use_fisheye_model) {
    std::cout << "Calibration using the fisheye model.\n";
    camera_model_type = "fisheye";
    std::cerr << "Whoops, this actually only works for perspective warp right "
              << "now... TBD.\n";
    return -1;
  }
  else {
    std::cout << "Calibration using 6th-order perspective warp.\n";
    camera_model_type = "perspective";
  }

  std::string data_dir(DATA_DIR_PATH);

  /****************************************************************************/
  /* Initialize video streams and camera objects                              */
  /****************************************************************************/
  int num_cameras = 2;

  cv::VideoCapture video_stream;
  std::vector<Camera> cameras(num_cameras);
  std::vector<int> video_num(num_cameras);   
  std::string input_file(data_dir + "/test_video.mp4");
  
  video_stream.open(input_file);
  video_stream.set(CV_CAP_PROP_RECTIFICATION, 1);
  video_stream.set(CV_CAP_PROP_FPS, 10);
  
  cv::Size video_size = cv::Size(1280,800);
  
  if (!video_stream.isOpened()) {
    std::cerr << "Cannot open video file: "
              << input_file << std::endl << std::endl;
    return -1;
  } else {
    std::cout << "Video file " << input_file << " opened.\n";
  }
  std::cout << std::endl;
  
  for (int i = 0; i < num_cameras; ++i ) {
    cameras[i].size = cv::Size(640, 400);
    cameras[i].K = cv::Matx33f::eye();
    cameras[i].distorsion = cv::Matx<double, 4, 1>::zeros();
  }

  /****************************************************************************/
  /* Load Aruco board configuration and detection data structures             */
  /****************************************************************************/
  square = 0.15;
  {
    aruco::BoardConfiguration aruco_board_config;
    aruco_board_config.readFromFile(data_dir + "/aruco14x8.yml");

    int num_markers = aruco_board_config.size();
    aruco_board_3f.reserve(num_markers * 4);

    for (int i = 0; i < num_markers; ++i) {
      aruco_marker_map.insert(std::pair<int, int>(aruco_board_config[i].id, i));
      for (int j = 0; j < 4; ++j )
        aruco_board_3f.push_back(cv::Point3f(aruco_board_config[i][j]) * square);
    }
  }

  std::vector<cv::Point2f> aruco_markers_detected_2f;
  std::vector<cv::Point3f> aruco_markers_detected_3f;
  std::vector<std::vector<cv::Point2f> > single_rig_detections_2f(num_cameras);
  std::vector<std::vector<cv::Point3f> > single_rig_detections_3f(num_cameras);
  std::vector<std::vector<std::vector<cv::Point2f> > >
      all_rig_detections_2f(num_cameras);
  std::vector<std::vector<std::vector<cv::Point3f> > >
      all_rig_detections_3f(num_cameras);

  std::vector<int> aruco_markers_detected_id;
  std::vector<std::vector<int> > single_rig_detections_id(num_cameras);
  std::vector<std::vector<std::vector<int> > >
      all_rig_detections_id(num_cameras);

  /****************************************************************************/
  /* Initialize camera calibration data structures                            */
  /****************************************************************************/
  std::vector<cv::Mat> Ks(num_cameras);
  std::vector<cv::Matx<double, 5, 1> > Ds(num_cameras);
  std::vector<std::vector<cv::Mat> > rvecs(num_cameras);
  std::vector<std::vector<cv::Mat> > tvecs(num_cameras);
  std::vector<cv::Mat> Rs(num_cameras); Rs[0] = cv::Mat::eye(3, 3, CV_64F);
  std::vector<cv::Mat> Oms(num_cameras); Oms[0] = cv::Mat::zeros(3, 1, CV_64F);
  std::vector<cv::Mat> Ts(num_cameras); Ts[0] = cv::Mat::zeros(3, 1, CV_64F);
  
  int flag = 0 || 
    cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC ||
    cv::fisheye::CALIB_CHECK_COND ||
    cv::fisheye::CALIB_FIX_SKEW;
  double rms;

  /****************************************************************************/
  /* Start reading video stream                                               */
  /****************************************************************************/
  std::string window_name = "Video stream";
  cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE); 
  std::string window_name2 = "Undistorted";
  cv::namedWindow(window_name2, CV_WINDOW_AUTOSIZE); 

  cv::Mat frames, frames_undistorted, frames_reprojected, frame_from_file;
  std::vector<cv::Mat> frame(num_cameras), frame_undistorted(num_cameras);
  std::vector<cv::Point2f> frame_pts;
  
  bool FOV_intersect;
  char keypress;
  int num_calib = 0;
  char num_calib_buffer[10];
  bool sane = false;
  bool calibrated = false;
  
  while(video_stream.read(frame_from_file)) { 
    frame_from_file(cv::Rect(0,0,1280,800)).copyTo(frame[0]);
    frame_from_file(cv::Rect(0,800,1280,800)).copyTo(frame[1]);
    
    for (int i = 0; i < num_cameras; ++i) {
      cv::resize(frame[i], frame[i], cv::Size(), 0.5, 0.5);
      
      FOV_intersect = DetectArucoMarkers(&(frame[i]),
                                         &aruco_markers_detected_2f,
                                         &aruco_markers_detected_3f,
                                         &aruco_markers_detected_id);
      single_rig_detections_2f[i] = aruco_markers_detected_2f;
      single_rig_detections_3f[i] = aruco_markers_detected_3f;
      single_rig_detections_id[i] = aruco_markers_detected_id;
    }
    hconcat(frame[0], frame[1], frames);
    
    if (calibrated) {
      for (int i = 0; i < num_cameras; ++i) {
        cv::undistort(frame[i], frame_undistorted[i], Ks[i], Ds[i]);
      }
      hconcat(frame_undistorted[0], frame_undistorted[1], frames_undistorted);
    }
    
    if (!sane) sane = true;
    // Display status in window
    sprintf(num_calib_buffer, "%d", num_calib);
    std::string str(num_calib_buffer);    
    
    cv::putText(frames, "Using " + camera_model_type + " camera model",
                cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, 
                cv::Scalar(255, 255, 255), 8);
    cv::putText(frames, "Using " + camera_model_type + " camera model",
                cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, 
                cv::Scalar(0, 0, 0), 3);
    cv::putText(frames, "# calibration captures = " + str, cv::Point(50, 100),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 8);
    cv::putText(frames, "# calibration captures = " + str, cv::Point(50, 100),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3);

    // Send image to display
    cv::imshow(window_name, frames);
    if (calibrated) {
      cv::imshow(window_name2, frames_undistorted);
      //      cv::imshow(window_name3, frames_reprojected);
    }
    keypress = cv::waitKey(30);

    // Esc key press -- EXIT
    if (keypress == 27) break;
    // SPACEBAR key press -- Collect data
    if (keypress == 32) {      
      if (FOV_intersect) {
        // Aruco markers detected in every camera
        for (int i = 0; i < num_cameras; ++i) {
          all_rig_detections_2f[i].push_back(single_rig_detections_2f[i]);
          all_rig_detections_3f[i].push_back(single_rig_detections_3f[i]);
          all_rig_detections_id[i].push_back(single_rig_detections_id[i]);
        }
        num_calib++;
      } else {
        std::cout << "Markers not seen by every camera.\n";
      }
    }
    // RETURN key press -- Calibrate
    if (keypress == 10) {
      std::cout << "--------------------------------------------------------\n";
      if (all_rig_detections_2f[0].size() == 0) {
        std::cout << "Cannot calibrate!  No data collected.\n";
      } else {
        std::cout << "Calibrating from " << num_calib << " "
                  << "captured collections.\n";

        // Calibrate intrinsics for each individual camera
        for (int i = 0; i < num_cameras; ++i) {
          rms = cv::calibrateCamera(all_rig_detections_3f[i],
                                    all_rig_detections_2f[i],
                                    cameras[i].size,
                                    Ks[i], Ds[i], rvecs[i], tvecs[i]);
          std::cout << "Camera " << i << ": " << std::endl
                    << "  K = " << Ks[i] << std::endl
                    << "  D = " << Ds[i].t() << std::endl
                    << "  rms = " << rms << "\n\n";
          calibrated = true;          
        }
        // If we have more than one camera, do stereo calibration with respect
        // to the camera at index 0
        if (num_cameras > 1) {
          std::vector<std::map<int, int> > all_stereo_board_map(
              all_rig_detections_id[0].size());
          // For each image capture, record which Aruco marker IDs were detected
          for (int j = 0; j < all_rig_detections_id[0].size(); ++j)
            for (int k = 0; k < all_rig_detections_id[0][j].size(); ++k)
              all_stereo_board_map[j].insert(std::pair<int, int>(
                  all_rig_detections_id[0][j][k], k));        
          
          // For each camera we stereo calibrate, find the 2D image and 3D board
          // geometry locations of mutually observed Aruco markers
          for (int i = 1; i < num_cameras; ++i) {
            std::vector<std::vector<std::pair<int, int> > >
                all_stereo_match_idx;
            std::vector<std::vector<cv::Point3f> > all_stereo_match_3f;
            
            for (int j = 0; j < all_rig_detections_id[i].size(); ++j) {
              std::vector<std::pair<int, int> > stereo_match_idx;      
              std::vector<cv::Point3f> stereo_match_3f;            
              for (int k = 0; k < all_rig_detections_id[i][j].size(); ++k) {          
                int marker_id = all_rig_detections_id[i][j][k];
                int found = all_stereo_board_map[j].count(marker_id);
                if (found) {
                  stereo_match_idx.push_back(std::pair<int, int>(
                      all_stereo_board_map[j].find(marker_id)->second, k));
                  for (int l = 0; l < 4; ++l) 
                    stereo_match_3f.push_back(aruco_board_3f[
                        aruco_marker_map.find(marker_id)->second * 4 + l]);
                }
              }
              all_stereo_match_idx.push_back(stereo_match_idx);
              all_stereo_match_3f.push_back(stereo_match_3f);
            }
          
            // Rebuild lists with matching stereo points
            std::vector<std::vector<cv::Point2f> >
                imagePoints1(all_stereo_match_idx.size());
            std::vector<std::vector<cv::Point2f> >
              imagePoints2(all_stereo_match_idx.size());
            for (int j = 0; j < all_stereo_match_idx.size(); ++j) {
              for (int k = 0; k < all_stereo_match_idx[j].size(); ++k) {
                const std::pair<int, int> &stereo_pair =
                    all_stereo_match_idx[j][k];
                for (int l = 0; l < 4; ++l) {
                  imagePoints1[j].push_back(
                      all_rig_detections_2f[0][j][stereo_pair.first * 4 + l]);
                  imagePoints2[j].push_back(
                      all_rig_detections_2f[i][j][stereo_pair.second * 4 + l]);
                }
              }         
              CV_Assert(
                  (all_stereo_match_3f[j].size() == imagePoints1[j].size()) &&
                  (all_stereo_match_3f[j].size() == imagePoints2[j].size()));
            }
            cv::Mat E, F;            
            rms = cv::stereoCalibrate(
                all_stereo_match_3f,
                imagePoints1,
                imagePoints2,
                Ks[0], Ds[0], 
                Ks[i], Ds[i],
                cameras[i].size, 
                Rs[i], Ts[i], E, F,
                cv::TermCriteria(cv::TermCriteria::COUNT, 30, 0),
                CV_CALIB_FIX_INTRINSIC);
            cv::Rodrigues(Rs[i], Oms[i]);
            std::cout << "Camera stereo pair (0, " << i << "):" << std::endl
                      << "  R = " << Rs[i] << std::endl
                      << " om = " << Oms[i].t() << std::endl
                      << "  t = " << Ts[i].t() << std::endl
                      << "  rms = " << rms << std::endl;
          }          
        }
        for (int i = 0; i < num_cameras; ++i) {
          all_rig_detections_3f[i].clear();
          all_rig_detections_2f[i].clear();
          all_rig_detections_id[i].clear();
        }
        num_calib = 0;
      }
    }    
  }

  /****************************************************************************/
  /* Close down                                                               *
  /****************************************************************************/
  if (calibrated) {
    // Print to file
    cv::FileStorage fs("test_video_calibration.xml", cv::FileStorage::WRITE);
    std::string camera_name;
    char camera_name_buffer[20];
    for (int i = 0; i < num_cameras; ++i) {
      sprintf(camera_name_buffer, "Camera%d", i);
      camera_name = std::string(camera_name_buffer);
      fs << camera_name + "_K" << cv::Mat(Ks[i]);
      fs << camera_name + "_D" << cv::Mat(Ds[i]);
    }
    if (num_cameras > 1) {
      for (int i = 1; i < num_cameras; ++i) {
        sprintf(camera_name_buffer, "CameraPair0_%d", i);
        camera_name = std::string(camera_name_buffer);
        fs << camera_name + "_R" << cv::Mat(Rs[i]);
        fs << camera_name + "_T" << cv::Mat(Ts[i]);
      }
    }     
    fs.release();
  }
  video_stream.release();
  return 0;
}

// This function uses the Aruco library to detect any Aruco markers in image *im
// and collects the corner image locations of the markers detected and the
// corresponding 3F coordinates from the original board configuration. 
bool DetectArucoMarkers(
    cv::Mat* im,
    std::vector<cv::Point2f>* aruco_marker_detects_2f,
    std::vector<cv::Point3f>* aruco_marker_detects_3f,
    std::vector<int>* aruco_marker_detects_id) {
  cv::Mat im_copy;
  if (!(im == NULL)) {
    cv::cvtColor(*im, im_copy, CV_BGR2GRAY);     
    if (AUTOBALANCE) Autobalance(&im_copy);
    aruco_marker_detects_2f->clear();
    aruco_marker_detects_3f->clear();
    aruco_marker_detects_id->clear();

    std::vector<aruco::Marker> markers;
    // Use the Aruco library detector
    aruco_detector.detect(im_copy, markers);
    if (markers.size() == 0) {
      return false;
    } else {
      int marker_idx;
      aruco_marker_detects_id->reserve(markers.size());
      aruco_marker_detects_2f->reserve(markers.size() * 4);
      aruco_marker_detects_3f->reserve(aruco_marker_detects_2f->size());  
      // If a found marker is on our board, record its location in the image
      // and from the board geometry in 3D
      for (int i = 0; i < markers.size(); ++i) {
        if (aruco_marker_map.count(markers[i].id) > 0) {
          marker_idx = aruco_marker_map.find(markers[i].id)->second;
          aruco_marker_detects_id->push_back(markers[i].id);
          if (DRAWMARKERS) markers[i].draw(*im, cv::Scalar(0, 0, 255), 2);
          for (int j = 0; j < 4; ++j) {
            aruco_marker_detects_2f->push_back(cv::Point2f(markers[i][j].x,
                                                           markers[i][j].y));
            aruco_marker_detects_3f->push_back(
                aruco_board_3f[marker_idx * 4 + j]);
          }
        }
      }
      return true;
    }
  } else return false;
}  

// This function normalizes the image intensities over [0, 256)
void Autobalance(cv::Mat* im) {
  if (!(im == NULL)) {
    cv::normalize(*im, *im, 1, 0, cv::NORM_MINMAX, CV_64F);
    *im = *im * 255.0f;
    im->convertTo(*im, CV_8U);
  }
}

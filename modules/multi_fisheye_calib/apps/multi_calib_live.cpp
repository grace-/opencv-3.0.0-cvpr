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

float square;
int num_markers;
int cam_calib_idx;

bool AUTOBALANCE = true;
bool DRAWMARKERS = true;

int main(int argc, char *argv[]) {
  
  if (argc < 4) { 
    std::cerr << "Usage: ./multi_fisheye_calibration <Aruco square size> "
              << "<true fisheye == 1 or perspective warp = 0> "
              << "<first device #> <OPTIONAL: # cameras>\n";
    return -1;
  }

  bool use_fisheye_model = atoi(argv[2]);
  std::string camera_model_type;
  if (use_fisheye_model) {
    std::cout << "Calibration using the fisheye model "
              << "-- use the 4x6 aruco board!\n";
    camera_model_type = "fisheye";
  }
  else {
    std::cout << "Calibration using 6th-order perspective warp "
              << "-- use the 10x20 aruco board!\n";
    camera_model_type = "perspective";
  }
  
  /****************************************************************************/
  /* Initialize video streams and camera objects                              */
  /****************************************************************************/
  int num_cameras;
  if (argc == 5) num_cameras = atoi(argv[4]);
  else num_cameras = 1;

  std::vector<cv::VideoCapture> video_stream(num_cameras);
  std::vector<Camera> cameras(num_cameras);
  std::vector<int> video_num(num_cameras);   
  video_num[0] = atoi(argv[3]);

  for (int i = 0; i < num_cameras; ++i) {
    video_num[i] = video_num[0] + i;
    video_stream[i].open(CV_CAP_FIREWIRE + video_num[i]);  
    video_stream[i].set(CV_CAP_PROP_RECTIFICATION, 1);
    video_stream[i].set(CV_CAP_PROP_FPS, 10);
    if (!video_stream[i].isOpened()) {
      std::cerr << "Cannot open firewire video stream: "
                << video_num[i] << std::endl << std::endl;
      return -1;
    } else {
      std::cout << "Video stream /dev/fw" << video_num[i] << " opened.\n";
    }
    cameras[i].size = cv::Size(0, 0);
    cameras[i].K = cv::Matx33f::eye();
    cameras[i].distorsion = cv::Matx<double, 4, 1>::zeros();
  }
  std::cout << std::endl;

  /****************************************************************************/
  /* Load Aruco board configuration and detection data structures             */
  /****************************************************************************/
  square = atof(argv[1]);
  {
    aruco::BoardConfiguration aruco_board_config;
    // add other board if doing fisheye
    std::string data_dir(DATA_DIR_PATH);
    if (use_fisheye_model) 
      aruco_board_config.readFromFile(data_dir + "/aruco6x4_meters.yml");
    else 
      aruco_board_config.readFromFile(data_dir + "/aruco20x10_meters.yml");

    num_markers = aruco_board_config.size();
    aruco_board_3f.reserve(num_markers * 4);

    for (int i = 0; i < num_markers; ++i) {
      aruco_marker_map.insert(std::pair<int, int>(aruco_board_config[i].id, i));
      for (int j = 0; j < 4; ++j )
        aruco_board_3f.push_back(
            cv::Point3f(aruco_board_config[i][j]) * square);
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
  
  std::vector<cv::Mat> Rs(num_cameras);
  std::vector<cv::Mat> Ts(num_cameras);

  std::vector<cv::Matx33d> Ks_f(num_cameras);
  std::vector<cv::Vec4d> Ds_f(num_cameras);
  std::vector<std::vector<cv::Vec3d> > rvecs_f(num_cameras);
  std::vector<std::vector<cv::Vec3d> > tvecs_f(num_cameras);
 
  std::vector<cv::Matx33d> Rs_f(num_cameras);
  std::vector<cv::Vec3d> Ts_f(num_cameras);

  int flag = 0 || 
    cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC ||
    cv::fisheye::CALIB_CHECK_COND ||
    cv::fisheye::CALIB_FIX_SKEW ||
    cv::fisheye::CALIB_FIX_INTRINSIC;
  double rms;
  cam_calib_idx = 0;

  /****************************************************************************/
  /* Start reading video stream                                               */
  /****************************************************************************/
  std::string window_name = "Video stream";
  cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE); 
  std::string window_name2 = "Undistorted";
  cv::namedWindow(window_name2, CV_WINDOW_AUTOSIZE); 

  cv::Mat frames, frames_undistorted, frame_temp;
  std::vector<cv::Mat> frame(num_cameras);

  bool FOV_intersect;
  char keypress;
  int num_calib = 0;
  char num_calib_buffer[10];
  bool sane = false;
  bool calibrated = false;
  int x_loc = 50;

  if (use_fisheye_model) std::cout << "Calibrating camera " << cam_calib_idx
                                   << " using fisheye model.\n";
  else std::cout << "Calibrating all cameras mutually.\n"; 

  while(video_stream[0].read(frame[0])) {
    // Initalize rig with first camera
    if (!sane) cameras[0].size = frame[0].size();
    if ((use_fisheye_model && ((cam_calib_idx == 0) || calibrated)) ||
        !use_fisheye_model) {
      FOV_intersect = DetectArucoMarkers(&(frame[0]),
                                         &aruco_markers_detected_2f,
                                         &aruco_markers_detected_3f,
                                         &aruco_markers_detected_id);
      single_rig_detections_2f[0] = aruco_markers_detected_2f;
      single_rig_detections_3f[0] = aruco_markers_detected_3f;
      single_rig_detections_id[0] = aruco_markers_detected_id;
    }
    frames = frame[0];    
    if (calibrated) 
      if (use_fisheye_model) cv::fisheye::undistortPoints(frame[0], 
                                                          frames_undistorted, 
                                                          Ks_f[0], Ds_f[0]);
      else cv::undistort(frame[0], frames_undistorted, Ks[0], Ds[0]);

    // Read the other cameras
    for (int i = 1; i < num_cameras; ++i) {      
      video_stream[i].read(frame[i]);  
      if (!sane) cameras[i].size = frame[i].size();
      if ((use_fisheye_model && (cam_calib_idx == i)) ||
          !use_fisheye_model) {
        FOV_intersect = DetectArucoMarkers(&(frame[i]),
                                           &aruco_markers_detected_2f,
                                           &aruco_markers_detected_3f,
                                           &aruco_markers_detected_id);
        single_rig_detections_2f[i] = aruco_markers_detected_2f;
        single_rig_detections_3f[i] = aruco_markers_detected_3f;
        single_rig_detections_id[i] = aruco_markers_detected_id;
      }
      cv::hconcat(frames, frame[i], frames);
      if (calibrated) {
        if (use_fisheye_model) cv::fisheye::undistortPoints(frame[i], frame_temp, Ks_f[i], Ds_f[i]);
        else cv::undistort(frame[i], frame_temp, Ks[i], Ds[i]);
        cv::hconcat(frames_undistorted, frame_temp, frames_undistorted);
      }
    }   
    if (!sane) sane = true;
    
    // Display status in window
    sprintf(num_calib_buffer, "%d", num_calib);
    std::string str(num_calib_buffer);    
    
    cv::putText(frames, "Using " + camera_model_type + " camera model",
                cv::Point(x_loc, 50), cv::FONT_HERSHEY_SIMPLEX, 1, 
                cv::Scalar(255, 255, 255), 8);
    cv::putText(frames, "Using " + camera_model_type + " camera model",
                cv::Point(x_loc, 50), cv::FONT_HERSHEY_SIMPLEX, 1, 
                cv::Scalar(0, 0, 0), 3);
    cv::putText(frames, "# calibration captures = " + str, cv::Point(x_loc, 100),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 8);
    cv::putText(frames, "# calibration captures = " + str, cv::Point(x_loc, 100),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 3);

    // Send image to display
    cv::imshow(window_name, frames);
    if (calibrated) cv::imshow(window_name2, frames_undistorted);
    keypress = cv::waitKey(30);

    // KEYPRESS Esc -- EXIT
    if (keypress == 27) break;
    // KEYPRESS SPACEBAR -- Collect data
    if (keypress == 32) { 
      if (use_fisheye_model) {
        if (calibrated &&
            ((single_rig_detections_id[0].size() == num_markers) &&
             (single_rig_detections_id[cam_calib_idx].size() == num_markers))) {
          all_rig_detections_2f[0].push_back(single_rig_detections_2f[0]);
          all_rig_detections_3f[0].push_back(single_rig_detections_3f[0]);
          all_rig_detections_id[0].push_back(single_rig_detections_id[0]);
          all_rig_detections_2f[cam_calib_idx]
              .push_back(single_rig_detections_2f[cam_calib_idx]);
          all_rig_detections_3f[cam_calib_idx]
              .push_back(single_rig_detections_3f[cam_calib_idx]);
          all_rig_detections_id[cam_calib_idx]
              .push_back(single_rig_detections_id[cam_calib_idx]);
          num_calib++;              
        } else
          if (single_rig_detections_id[cam_calib_idx].size() == num_markers) {
          all_rig_detections_2f[cam_calib_idx]
              .push_back(single_rig_detections_2f[cam_calib_idx]);
          all_rig_detections_3f[cam_calib_idx]
              .push_back(single_rig_detections_3f[cam_calib_idx]);
          all_rig_detections_id[cam_calib_idx]
              .push_back(single_rig_detections_id[cam_calib_idx]);
          num_calib++;
        } else {
          std::cout << "All markers must be detected in camera "
                    << cam_calib_idx << " to calibrate.\n";
        }
      } else {
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
    }
    // KEYPRESS RETURN -- Calibrate
    if (keypress == 10) {
      std::cout << "--------------------------------------------------------\n";
      if ((use_fisheye_model &&
           (all_rig_detections_2f[cam_calib_idx].size() > 0)) ||
          (all_rig_detections_2f[0].size() > 0)) {
        std::cout << "Calibrating from " << num_calib << " "
                  << "captured collections.\n";     
        if (!calibrated) {
          if (use_fisheye_model) {
            rms = cv::fisheye::calibrate(all_rig_detections_3f[cam_calib_idx],
                                         all_rig_detections_2f[cam_calib_idx],
                                         cameras[cam_calib_idx].size,
                                         Ks_f[cam_calib_idx],
                                         Ds_f[cam_calib_idx],
                                         rvecs_f[cam_calib_idx],
                                         tvecs_f[cam_calib_idx]);
            std::cout << "Camera " << cam_calib_idx << ": " << std::endl
                      << "  K = " << Ks_f[cam_calib_idx] << std::endl
                      << "  D = " << Ds_f[cam_calib_idx].t() << std::endl
                      << "  rms = " << rms << "\n\n";
            
            x_loc += cameras[cam_calib_idx].size.width;          
            cam_calib_idx++;
            if (cam_calib_idx == num_cameras) {
              calibrated = true;
              cam_calib_idx = 1;
              x_loc = 50 + cameras[0].size.width;
            }
            else 
              std::cout << "Calibrating camera " << cam_calib_idx
                        << " using fisheye model.\n";
          } else {
            for (int i = 0; i < num_cameras; ++i) {
              rms = cv::calibrateCamera(all_rig_detections_3f[i],
                                        all_rig_detections_2f[i],
                                        cameras[i].size,
                                        Ks[i], Ds[i], rvecs[i], tvecs[i]);
              std::cout << "camera " << i << ": " << std::endl
                        << "  K = " << Ks[i] << std::endl
                        << "  D = " << Ds[i].t() << std::endl
                        << "  rms = " << rms << "\n\n";
              calibrated = true;          
            }
          }    
        }  
                
        // Stereo calibration
        if (calibrated && (num_cameras > 1)) { 
          if (use_fisheye_model) {
            cv::Matx33d R;
            cv::Matx31d om;
            cv::Vec3d t;
            cv::Vec4d D1, D2;
            rms = cv::fisheye::stereoCalibrate(
                all_rig_detections_3f[0],
                all_rig_detections_2f[0],
                all_rig_detections_2f[cam_calib_idx],
                Ks_f[0], Ds_f[0], 
                Ks_f[cam_calib_idx], Ds_f[cam_calib_idx],
                cameras[cam_calib_idx].size,
                Rs_f[cam_calib_idx], Ts_f[cam_calib_idx], flag);
            cv::Rodrigues(Rs_f[cam_calib_idx], om);
            std::cout << "Fisheye camera stereo pair (0, " << cam_calib_idx
                      << "):" << std::endl
                      << "  R = " << Rs_f[cam_calib_idx] << std::endl
                      << " om = " << om.t() << std::endl
                      << "  t = " << Ts_f[cam_calib_idx].t() << std::endl
                      << "  rms = " << rms << std::endl;
            cam_calib_idx++;
          } else {
            // Build hashmap of arucos in camera 0 to match
            std::vector<std::map<int, int> > all_stereo_board_map(
                all_rig_detections_id[0].size());
            for (int j = 0; j < all_rig_detections_id[0].size(); ++j)
              for (int k = 0; k < all_rig_detections_id[0][j].size(); ++k)
                all_stereo_board_map[j].insert(std::pair<int, int>(
                     all_rig_detections_id[0][j][k], k));        
            
            // For each stereo pair, find indices of co-detected aruco markers
            // for each capture
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
                      stereo_match_3f.push_back(
                          aruco_board_3f[
                              aruco_marker_map.find(marker_id)->second * l]);
                  }
                }
                all_stereo_match_idx.push_back(stereo_match_idx);
                all_stereo_match_3f.push_back(stereo_match_3f);
              }            
              // Rebuild lists with matching stereo points
              std::vector<std::vector<cv::Point2f> > imagePoints1(
                  all_stereo_match_idx.size());
              std::vector<std::vector<cv::Point2f> > imagePoints2(
                  all_stereo_match_idx.size());
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
                CV_Assert((all_stereo_match_3f[j].size()
                           == imagePoints1[j].size()) &&
                          (all_stereo_match_3f[j].size()
                           == imagePoints2[j].size()));
              }
              // Calibrate
              cv::Mat R = cv::Mat::eye(3, 3, CV_64F), t, E, F, om;            
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
              cv::Rodrigues(Rs[i], om);
              std::cout << "Camera stereo pair (0, " << i << "):" << std::endl
                        << "  R = " << Rs[i] << std::endl
                        << " om = " << om.t() << std::endl
                        << "  t = " << Ts[i].t() << std::endl
                        << "  rms = " << rms << std::endl;
            }          
          }
        } 
        for (int i = 0; i < num_cameras; ++i) {
          all_rig_detections_3f[i].clear();
          all_rig_detections_2f[i].clear();
          all_rig_detections_id[i].clear();
        }
        num_calib = 0;
      } else {
        std::cout << "Cannot calibrate!  No data collected.\n";
      }
    }    
  }

  /****************************************************************************/
  /* Close down                                                               *
  /****************************************************************************/
  if (calibrated) {
    // Print to file
    cv::FileStorage fs("multi_cam_calibration.xml", cv::FileStorage::WRITE);
    fs << "CameraModelType" << camera_model_type;
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
  for (int i = 0; i < num_cameras; ++i) video_stream[i].release();
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
            aruco_marker_detects_3f->push_back(aruco_board_3f[marker_idx * j]);
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

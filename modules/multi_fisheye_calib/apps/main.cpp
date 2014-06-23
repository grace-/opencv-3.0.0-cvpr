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

std::vector<std::vector<int> > num_detected_markers;
std::vector<std::vector<std::vector<cv::Point2f> > > marker_pnts_;
std::vector<std::vector<std::vector<cv::Point3f> > > marker_pnts_map_;
std::vector<std::vector<cv::Mat> > rvecs_;
std::vector<std::vector<cv::Mat> > tvecs_;

float square;

bool AUTOBALANCE = true;
bool DRAWMARKERS = true;

int main(int argc, char *argv[]) {
  
  if (argc < 4) { 
    std::cerr << "Usage: ./fisheye_calibration <Aruco square size> "
              << "<true fisheye == 1 or perspective warp = 0> "
              << "<first device #> <OPTIONAL: # cameras>\n";
    return -1;
  }

  bool use_fisheye_model = atoi(argv[2]);
  std::string camera_model_type;
  if (use_fisheye_model) {
    std::cout << "Calibration using the fisheye model "
              << "-- have two boards ready!\n";
    camera_model_type = "fisheye";
  }
  else {
    std::cout << "Calibration using 6th-order perspective warp "
              << "-- have the long aruco board ready!\n";
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

  // Totem
  std::vector<cv::Point3f> gnomon;
  gnomon.push_back(cv::Point3f(0, 0, 0));
  gnomon.push_back(cv::Point3f(1, 0, 0));
  gnomon.push_back(cv::Point3f(0, 1, 0));
  gnomon.push_back(cv::Point3f(0, 0, 1));

  /****************************************************************************/
  /* Load Aruco board configuration and detection data structures             */
  /****************************************************************************/
  square = atof(argv[1]);
  {
    aruco::BoardConfiguration aruco_board_config;
    //aruco_board_config.readFromFile("../data/aruco6x4_meters.yml");
    // add other board if doing fisheye
    std::string data_dir(DATA_DIR_PATH);
    aruco_board_config.readFromFile(data_dir + "/aruco20x10_meters.yml");

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
  std::string window_name3 = "Gnomon";
  //  cv::namedWindow(window_name3, CV_WINDOW_AUTOSIZE); 

  cv::Mat frames, frames_undistorted, frames_reprojected, frame_temp;
  std::vector<cv::Mat> frame(num_cameras);
  std::vector<cv::Point2f> frame_pts;

  bool FOV_intersect;
  char keypress;
  int num_calib = 0;
  char num_calib_buffer[10];
  bool sane = false;
  bool calibrated = false;

  while(video_stream[0].read(frame[0])) {
    // Initalize rig with first camera
    if (!sane) cameras[0].size = frame[0].size();
    frame_temp = frame[0].clone();
    FOV_intersect = DetectArucoMarkers(&(frame[0]),
                                       &aruco_markers_detected_2f,
                                       &aruco_markers_detected_3f,
                                       &aruco_markers_detected_id);
    single_rig_detections_2f[0] = aruco_markers_detected_2f;
    single_rig_detections_3f[0] = aruco_markers_detected_3f;
    single_rig_detections_id[0] = aruco_markers_detected_id;
    frames = frame[0];    
    if (calibrated) {
      // if (aruco_markers_detected_3f.size() > 0) {
      //   cv::Mat R, T;        
      //   std::vector<cv::Point3f> temp1;
      //   std::vector<cv::Point2f> temp2;
      //   for (int k = 0; k < 4; ++k ) {
      //     temp1.push_back(aruco_markers_detected_3f[k]);
      //     temp2.push_back(aruco_markers_detected_2f[k]);
      //   }
      //   cv::solvePnP(temp1, temp2,
      //                Ks[0], Ds[0], 
      //                R, T, false, CV_P3P);
      //   std::vector<cv::Point2f> imgPts;     
        
      //   cv::projectPoints(gnomon, R, T, Ks[0], Ds[0], imgPts);
      //   std::cout << R << std::endl;
      //   std::cout << T << std::endl;
      //   frames_reprojected = frame_temp.clone();
      //   cv::line(frames_reprojected, imgPts[0], imgPts[1], cv::Scalar(0, 0, 255), 2);
      //   cv::line(frames_reprojected, imgPts[0], imgPts[2], cv::Scalar(0, 255, 0), 2);
      //   cv::line(frames_reprojected, imgPts[0], imgPts[3], cv::Scalar(255, 0, 0), 2);      
      // }
      cv::undistort(frame[0], frames_undistorted, Ks[0], Ds[0]);
    }
    
    // Read the other cameras
    for (int i = 1; i < num_cameras; ++i) {      
      video_stream[i].read(frame[i]);  
      if (!sane) cameras[i].size = frame[i].size();
      FOV_intersect = DetectArucoMarkers(&(frame[i]),
                                         &aruco_markers_detected_2f,
                                         &aruco_markers_detected_3f,
                                         &aruco_markers_detected_id);
      single_rig_detections_2f[i] = aruco_markers_detected_2f;
      single_rig_detections_3f[i] = aruco_markers_detected_3f;
      single_rig_detections_id[i] = aruco_markers_detected_id;
      cv::hconcat(frames, frame[i], frames);
      if (calibrated) {
        cv::undistort(frame[i], frame_temp, Ks[i], Ds[i]);
        cv::hconcat(frames_undistorted, frame_temp, frames_undistorted); 
      }
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

        if (num_cameras > 1) {
          std::vector<std::map<int, int> > all_stereo_board_map(all_rig_detections_id[0].size());
          for (int j = 0; j < all_rig_detections_id[0].size(); ++j)
            for (int k = 0; k < all_rig_detections_id[0][j].size(); ++k)
              all_stereo_board_map[j].insert(std::pair<int, int>(all_rig_detections_id[0][j][k], k));        
          
          for (int i = 1; i < num_cameras; ++i) {
            std::vector<std::vector<std::pair<int, int> > > all_stereo_match_idx; 
            std::vector<std::vector<cv::Point3f> > all_stereo_match_3f;
            
            for (int j = 0; j < all_rig_detections_id[i].size(); ++j) {
              std::vector<std::pair<int, int> > stereo_match_idx;      
              std::vector<cv::Point3f> stereo_match_3f;            
              for (int k = 0; k < all_rig_detections_id[i][j].size(); ++k) {          
                int marker_id = all_rig_detections_id[i][j][k];
                int found = all_stereo_board_map[j].count(marker_id);
                if (found) {
                  stereo_match_idx.push_back(std::pair<int, int>(all_stereo_board_map[j].find(marker_id)->second, k));
                  for (int l = 0; l < 4; ++l) 
                    stereo_match_3f.push_back(aruco_board_3f[aruco_marker_map.find(marker_id)->second * 4 + l]);
                }
              }
              all_stereo_match_idx.push_back(stereo_match_idx);
              all_stereo_match_3f.push_back(stereo_match_3f);
            }
          
            // Rebuild lists with matching stereo points
            std::vector<std::vector<cv::Point2f> > imagePoints1(all_stereo_match_idx.size());
            std::vector<std::vector<cv::Point2f> > imagePoints2(all_stereo_match_idx.size());
            for (int j = 0; j < all_stereo_match_idx.size(); ++j) {
              for (int k = 0; k < all_stereo_match_idx[j].size(); ++k) {
                const std::pair<int, int> &stereo_pair = all_stereo_match_idx[j][k];
                for (int l = 0; l < 4; ++l) {
                  imagePoints1[j].push_back(all_rig_detections_2f[0][j][stereo_pair.first * 4 + l]);
                  imagePoints2[j].push_back(all_rig_detections_2f[i][j][stereo_pair.second * 4 + l]);
                }
              }         
              CV_Assert(all_stereo_match_3f[j].size() == imagePoints1[j].size() &&
                        all_stereo_match_3f[j].size() == imagePoints2[j].size());     
            }
            cv::Mat E, F;            
            rms = cv::stereoCalibrate(all_stereo_match_3f,
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
    aruco_detector.detect(im_copy, markers);    
    if (markers.size() == 0) {
      return false;
    } else {
      int marker_idx;
      aruco_marker_detects_id->reserve(markers.size());
      aruco_marker_detects_2f->reserve(markers.size() * 4);
      aruco_marker_detects_3f->reserve(aruco_marker_detects_2f->size());  
      for (int i = 0; i < markers.size(); ++i) {
        if (aruco_marker_map.count(markers[i].id) > 0) {
          marker_idx = aruco_marker_map.find(markers[i].id)->second;
          aruco_marker_detects_id->push_back(markers[i].id);
          if (DRAWMARKERS) markers[i].draw(*im, cv::Scalar(0, 0, 255), 2);
          for (int j = 0; j < 4; ++j) {
            aruco_marker_detects_2f->push_back(cv::Point2f(markers[i][j].x,
                                                           markers[i][j].y));
            aruco_marker_detects_3f->push_back(aruco_board_3f[marker_idx * 4 + j]);
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

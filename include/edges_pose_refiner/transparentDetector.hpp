/*
 * transparentDetector.hpp
 *
 *  Created on: Dec 13, 2011
 *      Author: ilysenkov
 */

#ifndef TRANSPARENTDETECTOR_HPP_
#define TRANSPARENTDETECTOR_HPP_

#include "edges_pose_refiner/poseEstimator.hpp"

struct TransparentDetectorParams
{
  //plane estimation parameters
  int kSearch;
  float distanceThreshold;
  float clusterTolerance;
  cv::Point3f verticalDirection;

  TransparentDetectorParams()
  {
    kSearch = 10;
    distanceThreshold = 0.02f;
    clusterTolerance = 0.05f;
    verticalDirection = cv::Point3f(0.0f, -1.0f, 0.0f);
  }
};

class TransparentDetector
{
public:
  TransparentDetector(const PinholeCamera &camera = PinholeCamera(), const TransparentDetectorParams &params = TransparentDetectorParams());
  void initialize(const PinholeCamera &camera = PinholeCamera(), const TransparentDetectorParams &params = TransparentDetectorParams());

  void addObject(const std::string &name, const PoseEstimator &poseEstimator);
  void detect(const cv::Mat &bgrImage, const cv::Mat &depth, const cv::Mat &registrationMask, const pcl::PointCloud<pcl::PointXYZ> &sceneCloud, std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities, std::vector<std::string> &objectNames) const;

  void visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames, cv::Mat &image) const;
  void visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames, pcl::PointCloud<pcl::PointXYZ> &cloud) const;
private:
  int getObjectIndex(const std::string &name) const;

  TransparentDetectorParams params;
  PinholeCamera camera;
  std::vector<PoseEstimator> poseEstimators;
  std::vector<std::string> objectNames;
};


#endif /* TRANSPARENTDETECTOR_HPP_ */

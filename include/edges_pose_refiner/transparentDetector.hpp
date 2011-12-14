/*
 * transparentDetector.hpp
 *
 *  Created on: Dec 13, 2011
 *      Author: ilysenkov
 */

#ifndef TRANSPARENTDETECTOR_HPP_
#define TRANSPARENTDETECTOR_HPP_

#include "edges_pose_refiner/poseEstimator.hpp"

class TransparentDetector
{
public:
  TransparentDetector(const PinholeCamera &camera = PinholeCamera());
  void initialize(const PinholeCamera &camera = PinholeCamera());

  void addObject(const std::string &name, const PoseEstimator &poseEstimator);
  void detect(const cv::Mat &bgrImage, const cv::Mat &depth, const cv::Mat &registrationMask, const pcl::PointCloud<pcl::PointXYZ> &sceneCloud, std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities, std::vector<std::string> &objectNames) const;

  void visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames, cv::Mat &image) const;
  void visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames, pcl::PointCloud<pcl::PointXYZ> &cloud) const;
private:
  int getObjectIndex(const std::string &name) const;

  PinholeCamera camera;
  std::vector<PoseEstimator> poseEstimators;
  std::vector<std::string> objectNames;
};


#endif /* TRANSPARENTDETECTOR_HPP_ */

/*
 * pinholeCamera.hpp
 *
 *  Created on: Oct 14, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef PINHOLECAMERA_HPP_
#define PINHOLECAMERA_HPP_

#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/poseRT.hpp"

struct PinholeCamera
{
  PinholeCamera(const cv::Mat &cameraMatrix = cv::Mat(), const cv::Mat &distCoeffs = cv::Mat(), const PoseRT &extrinsics = PoseRT(), const cv::Size &imageSize = cv::Size(-1, -1));
  PinholeCamera(const PinholeCamera &camera);
  PinholeCamera& operator=(const PinholeCamera &camera);

  cv::Point2f projectPoints(cv::Point3f point, const PoseRT &pose) const;
  void projectPoints(const std::vector<cv::Point3f> &points, const PoseRT &pose, std::vector<cv::Point2f> &projectedPoints) const;

  cv::Point3f reprojectPoints(cv::Point2f point) const;
  void reprojectPoints(const std::vector<cv::Point2f> &points, std::vector<cv::Point3f> &rays) const;

  cv::Point3f reprojectPointsOnTable(cv::Point2f point, const cv::Vec4f &tablePlane) const;
  void reprojectPointsOnTable(const std::vector<cv::Point2f> &points, const cv::Vec4f &tablePlane,
                              std::vector<cv::Point3f> &reprojectedPoints) const;

  void resize(cv::Size destinationSize);


  void write(const std::string &filename) const;
  void write(cv::FileStorage &fs) const;

  void read(const std::string &filename);
  void read(const cv::FileNode &fn);

  cv::Mat cameraMatrix, distCoeffs;
  PoseRT extrinsics;
  cv::Size imageSize;
};

#endif /* PINHOLECAMERA_HPP_ */

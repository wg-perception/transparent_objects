/*
 * pinholeCamera.cpp
 *
 *  Created on: Oct 14, 2011
 *      Author: Ilya Lysenkov
 */

#include "edges_pose_refiner/pinholeCamera.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

using std::cout;
using std::endl;

PinholeCamera::PinholeCamera(const cv::Mat &_cameraMatrix, const cv::Mat &_distCoeffs, const PoseRT &_extrinsics, const cv::Size &_imageSize)
{
  if (_cameraMatrix.type() == CV_64FC1)
  {
    cameraMatrix = _cameraMatrix;
  }
  else
  {
    _cameraMatrix.convertTo(cameraMatrix, CV_64FC1);
  }

  if (_distCoeffs.empty())
  {
    const int distCoeffsCount = 5;
    distCoeffs = Mat::zeros(distCoeffsCount, 1, CV_32FC1);
  }
  else
  {
    distCoeffs = _distCoeffs;
  }
  extrinsics = _extrinsics;
  imageSize = _imageSize;
}

void PinholeCamera::projectPoints(const std::vector<cv::Point3f> &points, const PoseRT &pose_cam, std::vector<cv::Point2f> &projectedPoints) const
{
  PoseRT fullPose = extrinsics * pose_cam;
  cv::projectPoints(Mat(points), fullPose.getRvec(), fullPose.getTvec(), cameraMatrix, distCoeffs, projectedPoints);
}

cv::Point2f PinholeCamera::projectPoints(cv::Point3f point, const PoseRT &pose) const
{
    vector<Point2f> projectedPoints;
    projectPoints(vector<Point3f>(1, point), pose, projectedPoints);
    return projectedPoints[0];
}

void PinholeCamera::reprojectPoints(const std::vector<cv::Point2f> &points, std::vector<cv::Point3f> &rays) const
{
  const float eps = 1e-4;
  CV_Assert(norm(distCoeffs) < eps);

  Mat homogeneousPoints;
  convertPointsToHomogeneous(points, homogeneousPoints);
  Mat cameraMatrixFloat;
  cameraMatrix.convertTo(cameraMatrixFloat, CV_32FC1);
  Mat reprojectedRaysMat = homogeneousPoints.reshape(1) * cameraMatrixFloat.inv().t();
  CV_Assert(reprojectedRaysMat.type() == CV_32FC1);

  //TODO: check that it is a deep copy
  rays = reprojectedRaysMat.reshape(3);

  //TODO: remove
  for (size_t i = 0; i < rays.size(); ++i)
  {
      CV_Assert(fabs(rays[i].z - 1.0) < eps);
  }
}

cv::Point3f PinholeCamera::reprojectPoints(cv::Point2f point) const
{
    vector<Point3f> allRays;
    reprojectPoints(vector<Point2f>(1, point), allRays);
    return allRays[0];
}

PinholeCamera::PinholeCamera(const PinholeCamera &camera)
{
  *this = camera;
}

PinholeCamera& PinholeCamera::operator=(const PinholeCamera &camera)
{
  if (this != &camera)
  {
    cameraMatrix = camera.cameraMatrix.clone();
    distCoeffs = camera.distCoeffs.clone();
    extrinsics = camera.extrinsics;
    imageSize = camera.imageSize;
  }
  return *this;
}

void PinholeCamera::write(const std::string &filename) const
{
  FileStorage fs(filename, FileStorage::WRITE);
  if(!fs.isOpened())
  {
    CV_Error(CV_StsBadArg, "Cannot open pinhole camera file: " + filename);
  }
  write(fs);
  fs.release();
}

void PinholeCamera::write(cv::FileStorage &fs) const
{
  fs << "camera" << "{";
  fs << "K" << cameraMatrix;
  fs << "D" << distCoeffs;
  fs << "width" << imageSize.width;
  fs << "height" << imageSize.height;

  fs << "pose" << "{";
  fs << "rvec" << extrinsics.getRvec();
  fs << "tvec" << extrinsics.getTvec();
  fs << "}" << "}";
}

void PinholeCamera::read(const std::string &filename)
{
  FileStorage fs(filename, FileStorage::READ);
  if(!fs.isOpened())
  {
    CV_Error(CV_StsBadArg, "Cannot open pinhole camera file: " + filename);
  }

  read(fs.root());
  fs.release();
}

void PinholeCamera::read(const cv::FileNode &fn)
{
  fn["camera"]["K"] >> cameraMatrix;
  fn["camera"]["D"] >> distCoeffs;
  fn["camera"]["width"] >> imageSize.width;
  fn["camera"]["height"] >> imageSize.height;

  Mat rvec, tvec;
  fn["camera"]["pose"]["rvec"] >> rvec;
  fn["camera"]["pose"]["tvec"] >> tvec;
  
  Mat rvec64, tvec64;
  rvec.convertTo(rvec64, CV_64FC1);
  tvec.convertTo(tvec64, CV_64FC1);
  extrinsics = PoseRT(rvec64, tvec64);
}

void PinholeCamera::resize(cv::Size destinationSize)
{
  CV_Assert(imageSize.width > 0 && imageSize.height > 0);
  double fx = destinationSize.width / static_cast<double>(imageSize.width);
  double fy = destinationSize.height / static_cast<double>(imageSize.height);
  Mat xRow = cameraMatrix.row(0);
  Mat newXRow = cameraMatrix.row(0) * fx;
  newXRow.copyTo(xRow);

  Mat yRow = cameraMatrix.row(1);
  Mat newYRow = cameraMatrix.row(1) * fy;
  newYRow.copyTo(yRow);

  imageSize = destinationSize;
}

void PinholeCamera::reprojectPointsOnTable(const std::vector<cv::Point2f> &points, const cv::Vec4f &tablePlane,
                                           std::vector<cv::Point3f> &reprojectedPoints) const
{
  vector<Point3f> reprojectedRays;
  reprojectPoints(points, reprojectedRays);

  reprojectedPoints.clear();
  reprojectedPoints.reserve(points.size());
  for (size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex)
  {
    Point3f ray = reprojectedRays[pointIndex];
    double denominator = tablePlane[0] * ray.x +
                         tablePlane[1] * ray.y +
                         tablePlane[2] * ray.z;
    const float eps = 1e-4;
    CV_Assert(fabs(denominator) > eps);
    double t = -tablePlane[3] / denominator;
    Point3f finalPoint = ray * t;
    reprojectedPoints.push_back(finalPoint);
  }
}

cv::Point3f PinholeCamera::reprojectPointsOnTable(cv::Point2f point, const cv::Vec4f &tablePlane) const
{
    vector<Point3f> reprojectedPoints;
    reprojectPointsOnTable(vector<Point2f>(1, point), tablePlane, reprojectedPoints);
    return reprojectedPoints[0];
}

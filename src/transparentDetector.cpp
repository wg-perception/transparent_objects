/*
 * transparentDetector.cpp
 *
 *  Created on: Dec 13, 2011
 *      Author: ilysenkov
 */

#include "edges_pose_refiner/transparentDetector.hpp"
#include "edges_pose_refiner/utils.hpp"
#include "edges_pose_refiner/glassDetector.hpp"
#include "edges_pose_refiner/pclProcessing.hpp"

#include <boost/thread/thread.hpp>

TransparentDetector::TransparentDetector(const PinholeCamera &_camera)
{
  initialize(_camera);
}

void TransparentDetector::initialize(const PinholeCamera &_camera)
{
  camera = _camera;
}

void TransparentDetector::addObject(const std::string &name, const PoseEstimator &estimator)
{
  poseEstimators.push_back(estimator);
  objectNames.push_back(name);
}

void TransparentDetector::detect(const cv::Mat &bgrImage, const cv::Mat &depth, const cv::Mat &registrationMask, const pcl::PointCloud<pcl::PointXYZ> &sceneCloud, std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities, std::vector<std::string> &detectedObjectNames) const
{
  cv::Vec4f tablePlane;
  pcl::PointCloud<pcl::PointXYZ> tableHull;
  int kSearch = 10;
  float distanceThreshold = 0.02f;
  std::cout << "WARNING: hard-coded parameters" << std::endl;
  //TODO: fix
  bool isEstimated = computeTableOrientation(kSearch, distanceThreshold, sceneCloud, tablePlane, &tableHull);
  if (!isEstimated)
  {
    std::cerr << "Cannot find a table plane" << std::endl;
    return;
  }
  else
  {
    std::cout << "table plane is estimated" << std::endl;
  }

  int numberOfComponents;
  cv::Mat glassMask;
  GlassSegmentator glassSegmentator;
  glassSegmentator.segment(bgrImage, depth, registrationMask, numberOfComponents, glassMask, &camera, &tablePlane, &tableHull);

  std::cout << "glass is segmented" << std::endl;

#ifdef VISUALIZE_DETECTION
  cv::Mat segmentation = drawSegmentation(*color_, glassMask);
  imshow("glassMask", glassMask);
  imshow("segmentation", segmentation);
  cv::waitKey(100);
#endif

#ifdef TRANSPARENT_DEBUG
  cv::imwrite("color.png", *color_);
  cv::imwrite("depth.png", *depth_);
  cv::imwrite("glass.png", glassMask);
  cv::FileStorage fs("input.xml", cv::FileStorage::WRITE);
  fs << "K" << *K_;
  fs << "image" << *color_;
  fs << "depth" << *depth_;
  fs << "points3d" << *cloud_;
  fs.release();
#endif


  poses_cam.clear();
  detectedObjectNames.clear();
  posesQualities.clear();
  for (size_t i = 0; i < poseEstimators.size(); ++i)
  {
    std::cout << "starting to estimate pose..." << std::endl;
    std::vector<PoseRT> currentPoses;
    std::vector<float> currentPosesQualities;

    poseEstimators[i].estimatePose(bgrImage, glassMask, currentPoses, currentPosesQualities, &tablePlane);
    std::cout << "done." << std::endl;
    if (!currentPoses.empty())
    {
      poses_cam.push_back(currentPoses[0]);
      posesQualities.push_back(currentPosesQualities[0]);
      detectedObjectNames.push_back(objectNames[i]);
    }
  }
}

int TransparentDetector::getObjectIndex(const std::string &name) const
{
  std::vector<std::string>::const_iterator it = std::find(objectNames.begin(), objectNames.end(), name);
  CV_Assert(it != objectNames.end());

  return std::distance(objectNames.begin(), it);
}

void TransparentDetector::visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames, cv::Mat &image) const
{
  CV_Assert(poses.size() == objectNames.size());

  for (size_t i = 0; i < poses.size(); ++i)
  {
    cv::Scalar color(128 + rand() % 128, 128 + rand() % 128, 128 + rand() % 128);
    int objectIndex = getObjectIndex(objectNames[i]);
    poseEstimators[objectIndex].visualize(poses[i], image, color);
  }
}

void TransparentDetector::visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames, pcl::PointCloud<pcl::PointXYZ> &cloud) const
{
  CV_Assert(poses.size() == objectNames.size());

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("detected objects"));
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> sceneColor(cloud.makeShared(), 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud.makeShared(), sceneColor, "scene");

  for (size_t i = 0; i < poses.size(); ++i)
  {
    cv::Scalar color(128 + rand() % 128, 128 + rand() % 128, 128 + rand() % 128);
    int objectIndex = getObjectIndex(objectNames[i]);
    poseEstimators[objectIndex].visualize(poses[i], viewer, color, objectNames[i]);
  }

  while (!viewer->wasStopped ())
   {
     viewer->spinOnce (100);
     boost::this_thread::sleep (boost::posix_time::microseconds (100000));
   }
}


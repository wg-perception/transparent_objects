/*
 * poseEstimator.hpp
 *
 *  Created on: Dec 2, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef POSEESTIMATOR_HPP_
#define POSEESTIMATOR_HPP_

#include "edges_pose_refiner/edgeModel.hpp"
#include "edges_pose_refiner/pinholeCamera.hpp"
#include "edges_pose_refiner/poseRT.hpp"

#include <opencv2/core/core.hpp>

struct PoseEstimatorParams
{
  //training parameters
  int silhouetteCount;
  float downFactor;
  int closingIterationsCount;

  //plane estimation parameters
  float downLeafSize;
  int kSearch;
  float distanceThreshold;

  //edge detection parameters
  double cannyThreshold1;
  double cannyThreshold2;
  int dilationsForEdgesRemovalCount;

  size_t minGlassContourLength;

  float confidentDomination;

  PoseEstimatorParams()
  {
    silhouetteCount = 10;
    downFactor = 1.0f;
    closingIterationsCount = 10;

    minGlassContourLength = 10;

    downLeafSize = 0.001f;
    kSearch = 10;
    distanceThreshold = 0.02f;

    cannyThreshold1 = 25;
    cannyThreshold2 = 50;
    dilationsForEdgesRemovalCount = 10;

    confidentDomination = 1.5f;
  }

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;
};

class PoseEstimator
{
public:
  PoseEstimator(const PinholeCamera &kinectCamera, const PoseEstimatorParams &params = PoseEstimatorParams());
  void addObject(const EdgeModel &edgeModel);
  void estimatePose(const cv::Mat &kinectBgrImage, const cv::Mat &glassMask, const pcl::PointCloud<pcl::PointXYZ> &sceneCloud, std::vector<PoseRT> &poses_cam, std::vector<float> &poseQualities, const cv::Vec4f *tablePlane = 0) const;

  void read(const std::string &filename);
  void read(const cv::FileNode& fn);
  void write(const std::string &filename) const;
  void write(cv::FileStorage& fs) const;

  void visualize(const cv::Mat &image, const PoseRT &pose, const std::string &title = "estimated pose");
  void visualize(const pcl::PointCloud<pcl::PointXYZ> &scene, const PoseRT &pose, const std::string &title = "estimated pose 3D");
private:
  void computeCentralEdges(const cv::Mat &centralBgrImage, const cv::Mat &glassMask, cv::Mat &centralEdges, cv::Mat &silhouetteEdges) const;
  void getInitialPoses(const cv::Mat &glassMask, std::vector<PoseRT> &initialPoses, std::vector<float> &initialPosesQualities) const;
  void refineInitialPoses(const cv::Mat &centralBgrImage, const cv::Mat &glassMask, std::vector<PoseRT> &initPoses_cam, std::vector<float> &initPosesQualities) const;
  bool tmpComputeTableOrientation(const cv::Mat &centralBgrImage, cv::Vec4f &tablePlane, ros::Publisher *pt_pub) const;
  void findTransformationToTable(PoseRT &pose_cam, const cv::Vec4f &tablePlane, float &rotationAngle, ros::Publisher *pt_pub = 0, const cv::Mat finalJacobian = cv::Mat()) const;
  void refinePosesByTableOrientation(const cv::Vec4f &tablePlane, const cv::Mat &centralBgrImage, const cv::Mat &glassMask, std::vector<PoseRT> &poses_cam, std::vector<float> &initPosesQualities, ros::Publisher *pt_pub = 0) const;

  EdgeModel edgeModel;
  std::vector<Silhouette> silhouettes;

  PoseEstimatorParams params;
  PinholeCamera kinectCamera;
};

#endif /* POSEESTIMATOR_HPP_ */

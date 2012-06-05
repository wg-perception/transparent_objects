/*
 * transparentDetector.hpp
 *
 *  Created on: Dec 13, 2011
 *      Author: ilysenkov
 */

#ifndef TRANSPARENTDETECTOR_HPP_
#define TRANSPARENTDETECTOR_HPP_

#include "edges_pose_refiner/poseEstimator.hpp"
#include "edges_pose_refiner/glassSegmentator.hpp"

namespace transpod
{
  struct DetectorParams
  {
    //plane estimation parameters
    float downLeafSize;
    int kSearch;
    float distanceThreshold;
    float clusterTolerance;
    cv::Point3f verticalDirection;

    GlassSegmentatorParams glassSegmentationParams;

    DetectorParams()
    {
      downLeafSize = 0.002f;
      kSearch = 10;
      distanceThreshold = 0.02f;
      clusterTolerance = 0.05f;
      verticalDirection = cv::Point3f(0.0f, -1.0f, 0.0f);
      glassSegmentationParams = GlassSegmentatorParams();
    }
  };

  class Detector
  {
  public:
    struct DebugInfo;

    Detector(const PinholeCamera &camera = PinholeCamera(), const DetectorParams &params = DetectorParams());
    void initialize(const PinholeCamera &camera = PinholeCamera(), const DetectorParams &params = DetectorParams());

    void addTrainObject(const std::string &objectName, const std::vector<cv::Point3f> &points,
                        bool isModelUpsideDown, bool centralize);
    void addTrainObject(const std::string &objectName, const EdgeModel &edgeModel);
    void addTrainObject(const std::string &objectName, const PoseEstimator &poseEstimator);

    void detect(const cv::Mat &bgrImage, const cv::Mat &depth, const cv::Mat &registrationMask, const pcl::PointCloud<pcl::PointXYZ> &sceneCloud,
                std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities, std::vector<std::string> &objectNames,
                DebugInfo *debugInfo = 0) const;

    int getTrainObjectIndex(const std::string &name) const;

    void visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames,
                   cv::Mat &image) const;
    void visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames,
                   pcl::PointCloud<pcl::PointXYZ> &cloud) const;
  private:
    bool tmpComputeTableOrientation(const PinholeCamera &camera, const cv::Mat &centralBgrImage,
                                    cv::Vec4f &tablePlane) const;

    DetectorParams params;
    PinholeCamera srcCamera;
    std::vector<PoseEstimator> poseEstimators;
    std::vector<std::string> objectNames;
    cv::Size validTestImageSize;
  };

  struct Detector::DebugInfo
  {
      cv::Mat glassMask;
      std::vector<cv::Mat> initialSilhouettes;
  };
}

#endif /* TRANSPARENTDETECTOR_HPP_ */

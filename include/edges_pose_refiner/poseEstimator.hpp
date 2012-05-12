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
#include "edges_pose_refiner/localPoseRefiner.hpp"

#ifdef USE_3D_VISUALIZATION
#include <pcl/visualization/pcl_visualizer.h>
#endif

#include <opencv2/core/core.hpp>

struct PoseEstimatorParams
{
  //training parameters
  int silhouetteCount;
  float downFactor;
  int closingIterationsCount;

  //edge detection parameters
  double cannyThreshold1;
  double cannyThreshold2;
  int dilationsForEdgesRemovalCount;

  size_t minGlassContourLength;
  double minGlassContourArea;

  float confidentDomination;
  int icp2dIterationsCount;
  float min2dScaleChange;
  bool useClosedFormPnP;

  float ghGranularity;
  int ghBasisStep;
  float ghMinDistanceBetweenBasisPoints;

  int ghTestBasisStep;

  //length of the object contour relative to length of the whole extracted contour
  float ghObjectContourProportion;

  //probability to find a basis which belongs to the object
  float ghSuccessProbability;

  int votesWindowSize;
  float votesConfidentSuppression;
  float basisConfidentSuppression;

  float maxRotation3D;
  float maxTranslation3D;
  float confidentSuppresion3D;

  float minScale;

  //suppresion after alignment to a table
  float ratioToMinimum;
  float neighborMaxRotation;
  float neighborMaxTranslation;

  LocalPoseRefinerParams lmParams;

  PoseEstimatorParams()
  {
    silhouetteCount = 10;
    downFactor = 1.0f;
    closingIterationsCount = 10;

    minGlassContourLength = 20;
    minGlassContourArea = 64.0;

    cannyThreshold1 = 25;
    cannyThreshold2 = 50;
    dilationsForEdgesRemovalCount = 10;

    confidentDomination = 1.5f;
    icp2dIterationsCount = 50;
    min2dScaleChange = 0.001f;

    useClosedFormPnP = true;

    ghBasisStep = 2;
    ghMinDistanceBetweenBasisPoints = 0.1f;
    ghGranularity = 0.04f;

    ghTestBasisStep = 4;

    ghObjectContourProportion = 0.1f;
    ghSuccessProbability = 0.99f;

    votesWindowSize = 5;
    votesConfidentSuppression = 1.1f;
    basisConfidentSuppression = 1.5f;

    maxRotation3D = 0.8f;
    maxTranslation3D = 0.15f;
    confidentSuppresion3D = 1.3f;

    minScale = 0.2f;

    ratioToMinimum = 2.0f;
    neighborMaxRotation = 0.1f;
    neighborMaxTranslation = 0.02f;
  }

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;
};

class PoseEstimator
{
public:
  PoseEstimator(const PinholeCamera &kinectCamera = PinholeCamera(), const PoseEstimatorParams &params = PoseEstimatorParams());
  void setModel(const EdgeModel &edgeModel);
  void estimatePose(const cv::Mat &kinectBgrImage, const cv::Mat &glassMask, std::vector<PoseRT> &poses_cam, std::vector<float> &poseQualities, const cv::Vec4f *tablePlane = 0, std::vector<cv::Mat> *initialSilhouettes = 0) const;

  void read(const std::string &filename);
  void read(const cv::FileNode& fn);
  void write(const std::string &filename) const;
  void write(cv::FileStorage& fs) const;

  cv::Size getValidTestImageSize() const;

  void visualize(const PoseRT &pose, cv::Mat &image, cv::Scalar color = cv::Scalar(0, 0, 255)) const;
#ifdef USE_3D_VISUALIZATION
  void visualize(const PoseRT &pose, const boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, cv::Scalar color = cv::Scalar(0, 0, 255), const std::string &title = "object") const;
#endif
private:
  typedef std::pair<int, int> Basis;
  struct BasisMatch
  {
    float confidence;

    Basis trainBasis;
    Basis testBasis;

    int silhouetteIndex;

    cv::Mat similarityTransformation_cam, similarityTransformation_obj;
    PoseRT pose;

    BasisMatch();
  };

  static void suppressNonMinimum(std::vector<float> errors, float absoluteSuppressionFactor, std::vector<bool> &isSuppressed, bool useNeighbors = true);

  //it is not thread-safe
  void findBasisMatches(const std::vector<cv::Point2f> &contour, const Basis &testBasis, std::vector<BasisMatch> &basisMatches) const;

  void estimateSimilarityTransformations(const std::vector<cv::Point> &contour, std::vector<BasisMatch> &matches) const;

  void estimatePoses(std::vector<BasisMatch> &matches) const;


  void suppressBasisMatches(const std::vector<BasisMatch> &matches, std::vector<BasisMatch> &filteredMatches) const;
  void suppressSimilarityTransformations(const std::vector<BasisMatch> &matches, const std::vector<cv::Mat> &similarityTransformaitons_obj, std::vector<bool> &isSuppressed) const;

  void suppressBasisMatchesIn3D(std::vector<BasisMatch> &matches) const;
  void filterOut3DPoses(const std::vector<float> &errors, const std::vector<PoseRT> &poses_cam,
                        float ratioToMinimum, float neighborMaxRotation, float neighborMaxTranslation,
                        std::vector<bool> &isFilteredOut) const;

  void generateGeometricHashes();
  void computeCentralEdges(const cv::Mat &centralBgrImage, const cv::Mat &glassMask, cv::Mat &centralEdges, cv::Mat &silhouetteEdges) const;
  void getInitialPoses(const cv::Mat &glassMask, std::vector<PoseRT> &initialPoses, std::vector<float> &initialPosesQualities) const;
  void getInitialPosesByGeometricHashing(const cv::Mat &glassMask, std::vector<PoseRT> &initialPoses, std::vector<float> &initialPosesQualities, std::vector<cv::Mat> *initialSilhouettes) const;

  void refineInitialPoses(const cv::Mat &testEdges, const cv::Mat &silhouetteEdges, std::vector<PoseRT> &initPoses_cam, std::vector<float> &initPosesQualities, std::vector<cv::Mat> *jacobians = 0) const;
  void findTransformationToTable(PoseRT &pose_cam, const cv::Vec4f &tablePlane, float &rotationAngle, const cv::Mat finalJacobian = cv::Mat()) const;
  void refinePosesByTableOrientation(const cv::Vec4f &tablePlane, const cv::Mat &testEdges, const cv::Mat &silhouetteEdges, const cv::Mat &glassMask, std::vector<PoseRT> &poses_cam, std::vector<float> &initPosesQualities) const;

  EdgeModel edgeModel;
  std::vector<Silhouette> silhouettes;
  std::vector<cv::Mat> canonicScales;
  //TODO: remove mutable
  mutable GHTable ghTable;
  mutable PoseEstimatorParams params;
  mutable std::vector<cv::Mat> votes;

  PinholeCamera kinectCamera;
};

#endif /* POSEESTIMATOR_HPP_ */

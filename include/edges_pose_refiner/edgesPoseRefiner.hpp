/*
 * edges_pose_refiner.hpp
 *
 *  Created on: Dec 22, 2010
 *      Author: ilysenkov
 */

#ifndef EDGES_POSE_REFINER_HPP_
#define EDGES_POSE_REFINER_HPP_

#include <opencv2/core/core.hpp>
#include <nlopt.hpp>

#include "edges_pose_refiner/edgeModel.hpp"
#include "edges_pose_refiner/localPoseRefiner.hpp"

/** \brief Parameters of pose refinement by edges */
struct EdgesPoseRefinerParams
{
  /** \brief percentile of partial directed Hausdorff distance (the cost function of global optimization) */
  float hTrimmedError;

  /** \brief maximum allowed rotations for several runs (in radians) */
  std::vector<double> maxRotationAngles;

  /** \brief maximum allowed translations for several runs (in units of camera coordinate system, e.g. for Kinect in meters) */
  std::vector<double> maxTranslations;

  /** \brief maximum allowed translation in direction of normal to a table (in units of camera coordinate system, e.g. for Kinect in meters)
   *
   *  It is used only when a table is used in global optimization.
   */
  double maxTranslationZ;


  /** \brief stopping criteria of NLopt
   *
   * For details see http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Stopping_criteria
   **/
  int maxNumberOfFunctionEvaluations;

  /** \brief stopping criteria of NLopt
   *
   * For details see http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Stopping_criteria
   **/
  double absoluteToleranceOnFunctionValue;

  /** \brief stopping criteria of NLopt
   *
   * For details see http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#Stopping_criteria
   **/
  double maxTime;

  /** \brief the algorithm used to optimize cost function */
  nlopt::algorithm globalOptimizationAlgorithm;

  /** \brief parameters of local optimization (LM-ICP) */
  LocalPoseRefinerParams localParams;

  EdgesPoseRefinerParams()
  {
    hTrimmedError = 0.8f;
    maxRotationAngles = {CV_PI / 2.0};
    maxTranslations = {0.01};
    maxNumberOfFunctionEvaluations = 32000;
    absoluteToleranceOnFunctionValue = 1e-13;
    maxTime = -1; //disable by default
    globalOptimizationAlgorithm = nlopt::GN_DIRECT;
  }
};

/** \brief The main class of pose refinement by edges
 *
 *  This class should be used to refine pose of the object.
 *  It will perform global optimization of the pose quality cost function and then refine computed pose by local optimization
 */
class EdgesPoseRefiner
{
public:
  /** \brief Create a class instance which will be able to refine poses of the specific object
   *
   * \param edgeModel The edge model of the object
   * \param camera The test camera
   * \param params Parameters of pose refinement by edges
   */
  EdgesPoseRefiner(const EdgeModel &edgeModel, const PinholeCamera &camera, const EdgesPoseRefinerParams &params = EdgesPoseRefinerParams());

  /** \brief Create a class instance which will be able to refine poses of the specific object using several test cameras
   *
   * \param edgeModel The edge model of the object
   * \param allCameras All test cameras
   * \param params Parameters of pose refinement by edges
   */
  EdgesPoseRefiner(const EdgeModel &edgeModel, const std::vector<PinholeCamera> &allCameras, const EdgesPoseRefinerParams &params = EdgesPoseRefinerParams());

  /** \brief Set parameters of pose refinement by edges
   * \param params Parameters of pose refinement by edges
   */
  void setParams(const EdgesPoseRefinerParams &params);

  /** \brief Set a mask of possible locations of the object center
   *  \param centerCamera A camera to project 3D points on mask
   *  \param mask A mask of the object center
   */
  void setCenterMask(const PinholeCamera &centerCamera, const cv::Mat &mask);

  /** \brief Refine pose of the object
   *
   *  \param testEdges Edges of your test image (e.g. it can be computed by Canny) where the object is located
   *  \param rvec The rotation vector of refined pose (in the test camera coordinate system)
   *  \param tvec The translation vector of refined pose (in the test camera coordinate system)
   *  \param usePoseGuess If true the function will use provided rvec and tvec as the initial
approximations of the rotation and translation vectors, respectively, and will further optimize
them. Else the function will use zero rotation and translation as initial guess.
   *  \return The minimum value of the cost function
   */
  double refine(const cv::Mat &testEdges, cv::Mat &rvec, cv::Mat &tvec, bool usePoseGuess = false) const;

  //TODO: move to private
  //use to analyze quality of global optimization and local one
  double refine(const cv::Mat &testEdges, cv::Mat &rvecFinal_cam, cv::Mat &tvecFinal_cam, cv::Mat &rvecGlobal_cam, cv::Mat &tvecGlobal_cam, bool usePoseGuess = false, const cv::Vec4f &tablePlane = cv::Vec4f(0.0f)) const;

  /** \brief Refine pose of the object using several test cameras
   *
   *  \param testEdges Edges of your test images from several cameras
   *  \param rvec The rotation vector of refined pose (in the test camera coordinate system)
   *  \param tvec The translation vector of refined pose (in the test camera coordinate system)
   *  \param usePoseGuess If true the function will use provided rvec and tvec as the initial
approximations of the rotation and translation vectors, respectively, and will further optimize
them. Else the function will use zero rotation and translation as initial guess.
   *  \return The minimum value of the cost function
   */
  double refine(const std::vector<cv::Mat> &testEdges, cv::Mat &rvecFinal_cam, cv::Mat &tvecFinal_cam, cv::Mat &rvecGlobal_cam, cv::Mat &tvecGlobal_cam, bool usePoseGuess = false, const cv::Vec4f &tablePlane = cv::Vec4f(0.0f)) const;
  double refine(const std::vector<cv::Mat> &testEdges, cv::Mat &rvecFinal_cam, cv::Mat &tvecFinal_cam, bool usePoseGuess = false, const cv::Vec4f &tablePlane = cv::Vec4f(0.0f)) const;

private:
  void initEdgesPoseRefiner(const EdgeModel &edgeModel, const std::vector<PinholeCamera> &allCameras, const EdgesPoseRefinerParams &params);
  void setBounds(nlopt::opt &opt, size_t iteration) const;
  double runGlobalOptimization(std::vector<cv::Ptr<PoseQualityEstimator> > *estimators, size_t iteration, const cv::Vec4f &tablePlane, cv::Mat &rvec_cam, cv::Mat &tvec_cam) const;
  void findTransformationToTable(cv::Mat &rvecFinal_cam, cv::Mat &tvecFinal_cam, const cv::Vec4f &tablePlane);
  void addPoseQualityEstimator(const EdgeModel &edgeModel, const cv::Mat &edges, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Mat &extrinsicsRt, const cv::Mat &rvec_cam, const cv::Mat &tvec_cam, std::vector<cv::Ptr<PoseQualityEstimator> > &poseQualityEstimators, bool usePoseGuess = false, const cv::Mat centerMask = cv::Mat()) const;

  EdgeModel edgeModel;
  std::vector<PinholeCamera> allCameras;
  PinholeCamera centerCamera;
  cv::Mat centerMask;

  EdgesPoseRefinerParams params;
  int dim;
};


#endif /* EDGES_POSE_REFINER_HPP_ */

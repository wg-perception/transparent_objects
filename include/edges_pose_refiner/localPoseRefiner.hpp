/*
 * localPoseRefiner.hpp
 *
 *  Created on: Apr 21, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef LOCALPOSEREFINER_HPP_
#define LOCALPOSEREFINER_HPP_

#include <opencv2/core/core.hpp>

#include "edges_pose_refiner/edgeModel.hpp"
#include "edges_pose_refiner/utils.hpp"


/** \brief Parameters of pose refinement by local optimization */
struct LocalPoseRefinerParams
{
  /** \brief Threshold of outlier distance
   *
   *   If distance between a projected edgel of the model and the nearest test edgel is larger than outlierDistance then the point is outlier
   */
  float outlierDistance;

  /** \brief error of outliers in cost function */
  float outlierError;

  /** \brief jacobian of outliers in Levenberg-Marquardt */
  double outlierJacobian;

  //Levenberg-Marquardt parameters
  /** \brief downsampling factor to use when computing silhouette edgels*/
  float lmDownFactor;

  /** \brief number of closing operations when computing silhouette edges*/
  int lmClosingIterationsCount;

  /** \brief ratio of inliers when applying Levenberg-Marquardt*/
  float lmInliersRatio;

  /** \brief Use only view-dependent edges or view-independent */
  bool useViewDependentEdges;

  /** \brief Number of closing operation iterations which are used to get view-dependent edges */
  int closingIterations;

  /** \brief Downsampling factor of a test image when projecting model points onto the image*/
  float downFactor;

  /** \brief Use orientation in chamfer matching or not */
  bool useOrientedChamferMatching;

  /** \brief Half of edgels count which are used to estimate orientation of an edgel on a test image
   * 
   *   Orientation of the edgel is computed by taking into account M edgels before and M edgels after.
   */
  int testM;
  
  /** \brief Half of edgels count which are used to estimate orientation of an edgel on a object contour
   * 
   *   Orientation of the edgel is computed by taking into account M edgels before and M edgels after.
   */  
  int objectM;  

  /** \brief Weight of the view-dependent edges term in the chamfer matching cost function
   *
   *   Must be between 0 and 1. The weight of stable edges is (1 - viewDependentEdgesWeight).
   */
  float viewDependentEdgesWeight;

  /** \brief Weight of the edges term in the oriented chamfer matching cost function */
  float edgesWeight;


  LocalPoseRefinerParams()
  {
//    outlierDistance = 200.0f;
    outlierDistance = std::numeric_limits<float>::max();
    outlierError = 320.0f;
    outlierJacobian = 0.0;

    lmDownFactor = 1.0f;
    lmClosingIterationsCount = 10;
    lmInliersRatio = 0.8f;

    useViewDependentEdges = false;
    closingIterations = 3;
    downFactor = 0.25f;

    useOrientedChamferMatching = false;
    testM = 5;
    objectM = 10;
    viewDependentEdgesWeight = 1.0 / 3.0;
    edgesWeight = 0.1f;
  }
};


/** \brief Pose refinement by local optimization */
class LocalPoseRefiner
{
public:

  /** \brief Create a class instance which will refine poses of the specific object by local optimization
   *
   *  \param edgeModel The edge model of the object
   *  \param edgesImage Edges of the test images. They can be computed by Canny.
   *  \param cameraMatrix Intrinsic parameters of the test camera
   *  \param distCoeffs Distortion coefficients of the test camera
   *  \param extrinsicsRt Extrinsic parameters of the test camera
   *  \param params Parameters of local optimization
   */
  LocalPoseRefiner(const EdgeModel &edgeModel, const cv::Mat &edgesImage, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Mat &extrinsicsRt, const LocalPoseRefinerParams &params = LocalPoseRefinerParams());

  /** \brief Set a mask of possible locations of the object centers
   * \param mask A mask of the center locations
   */
  void setCenterMask(const cv::Mat &mask);

  /** \brief Set test edges for silhouette edgels
   * \param edges silhouette test edges
   */
  void setSilhouetteEdges(const cv::Mat &edges);

  /** \brief Refine pose of the object
   *
   *  \param rvec The rotation vector of refined pose (in the test camera coordinate system)
   *  \param tvec The translation vector of refined pose (in the test camera coordinate system)
   *  \param usePoseGuess If true the function will use provided rvec and tvec as the initial
approximations of the rotation and translation vectors, respectively, and will further optimize
them. Else the function will use zero rotation and translation as initial guess.
   */
  void refine(cv::Mat &rvec, cv::Mat &tvec, bool usePoseGuess = false);

  /** \brief Refine pose of the object using silhouette edges also
   *
   *  \param pose_cam The refined pose (in the test camera coordinate system)
   *  \param usePoseGuess If true the function will use provided rvec and tvec as the initial
approximations of the rotation and translation vectors, respectively, and will further optimize
them. Else the function will use zero rotation and translation as initial guess.
   *  \param tablePlane coefficients of a table plane equation which is used as additional constraint
   *  \return registration error
   */
  float refineUsingSilhouette(PoseRT &pose_cam, bool usePoseGuess = false, const cv::Vec4f &tablePlane = cv::Vec4f::all(0.0f), cv::Mat *finalJacobian = 0);
private:
  static void computeDistanceTransform(const cv::Mat &edges, cv::Mat &distanceImage, cv::Mat &dx, cv::Mat &dy);

  //rotate_cam model points
  void setInitialPose(const PoseRT &pose_cam);

  //cache Rt_obj2cam and Rt_cam2obj
  void setObjectCoordinateSystem(const cv::Mat &Rt_obj2cam);
  void getObjectCoordinateSystem(cv::Mat &Rt_obj2cam) const;

  double estimatePoseQuality(const cv::Mat &rvec_obj, const cv::Mat &tvec_obj, float hTrimmedError, double *detOfCovarianceMatrix = 0) const;
  void computeResidualsForTrimmedError(cv::Mat &projectedPoints, std::vector<float> &residuals) const;
  //Attention! projectedPoints is not const for efficiency
  double calcTrimmedError(cv::Mat &projectedPoints, bool useInterpolation, float h) const;

  void computeJacobian(const cv::Mat &projectedPoints, const cv::Mat &JaW, const cv::Mat &distanceImage, const cv::Mat &dx, const cv::Mat &dy, cv::Mat &J);
  void computeObjectJacobian(const cv::Mat &projectedPoints, const cv::Mat &JaW, const cv::Mat &distanceImage, const cv::Mat &dx, const cv::Mat &dy, const cv::Mat &R_obj2cam, const cv::Mat &t_obj2cam, const cv::Mat &rvec_obj, const cv::Mat &tvec_obj, cv::Mat &J);
  void computeWeightsObjectJacobian(const std::vector<cv::Point3f> &points, const cv::Mat &silhouetteEdges, const PoseRT &pose_obj, cv::Mat &weightsJacobian) const;
  void computeResiduals(const cv::Mat &projectedPoints, cv::Mat &residuals, double inlierMaxDistance, double outlierError, const cv::Mat &distanceTransform = cv::Mat(), const bool useInterpolation = true) const;
  void computeResidualsWithInliersMask(const cv::Mat &projectedPoints, cv::Mat &residuals, double inlierMaxDistance, double outlierError, const cv::Mat &distanceTransform, const bool useInterpolation, float inliersRatio, cv::Mat &inliersMask) const;
  double getError(const cv::Mat &residuals) const;

  void computeWeights(const std::vector<cv::Point2f> &projectedPointsVector, const cv::Mat &silhouetteEdges, cv::Mat &weights) const;

  bool isOutlier(cv::Point2f pt) const;
  double getFilteredDistance(cv::Point2f pt, bool useInterpolation, double inlierMaxDistance, double outlierError = 0., const cv::Mat &distanceTransform = cv::Mat()) const;

  void displayProjection(const cv::Mat &projectedPoints, const std::string &title ) const;

  void projectPoints_obj(const cv::Mat &points, const cv::Mat &rvec_obj, const cv::Mat &tvec_obj, cv::Mat &rvec_cam, cv::Mat &tvec_cam, cv::Mat &Rt_cam, std::vector<cv::Point2f> &projectedPoints, cv::Mat *dpdrot = 0, cv::Mat *dpdt = 0) const;
  void object2cameraTransformation(const cv::Mat &rvec_obj, const cv::Mat &tvec_obj, cv::Mat &Rt_cam) const;

  EdgeModel originalEdgeModel;
  EdgeModel rotatedEdgeModel;

  cv::Mat edgesImage;
  cv::Mat cameraMatrix, distCoeffs;
  cv::Mat extrinsicsRt;

  cv::Mat silhouetteEdges;
  cv::Mat silhouetteDtImage;
  cv::Mat silhouetteDtDx, silhouetteDtDy;

  cv::Mat dtImage;
  cv::Mat dtDx, dtDy;

  cv::Mat Rt_obj2cam_cached, Rt_cam2obj_cached;

  cv::Mat orientationImage;
  cv::Mat cameraMatrix64F;

  cv::Mat centerMask, dtCenter;

  LocalPoseRefinerParams params;
  int dim;

  friend class PoseQualityEstimator;
};

/** \brief Estimate pose quality */
class PoseQualityEstimator
{
public:

  /**
   * \param hTrimmedError percentile of the partial directed Hausdorff function in the cost function
   * \param poseRefiner Local pose refiner of the current test image and the object
   */
  PoseQualityEstimator(const cv::Ptr<LocalPoseRefiner> &poseRefiner, float hTrimmedError);

  /** \brief Set initial pose of the object
   *
   *  Use the test camera coordinate system
   *
   *  \param pose_cam The object pose
   */
  void setInitialPose(const PoseRT &pose_cam);

  /** \brief Estimate pose quality
   *
   *  \param point [rvec, tvec] as one vector in the object coordinate system
   *  \return quality of pose
   */
  double evaluate(const std::vector<double> &point);

  /** \brief Convert from the object coordinate system to the camera coordinate system
   *
   *  \param point_obj [rvec, tvec] as one vector in the object coordinate system
   *  \param rvec_cam The corresponding rotation vector in the camera coordinate system
   *  \param tvec_cam The corresponding translation vector in the camera coordinate system
   */
  void obj2cam(const std::vector<double> &point_obj, cv::Mat &rvec_cam, cv::Mat &tvec_cam) const;
private:
  cv::Ptr<LocalPoseRefiner> poseRefiner;
  float hTrimmedError;
  //cv::Mat rvecInit_cam, tvecInit_cam;
  cv::Mat Rt_init_cam;
};

#endif /* LOCALPOSEREFINER_HPP_ */

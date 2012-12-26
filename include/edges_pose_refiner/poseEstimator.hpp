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

namespace transpod
{
  struct PoseEstimatorParams
  {
    /** \brief number of silhouettes to use in training */
    int silhouetteCount;

    /** \brief factor by which to reduce resolution when generating silhouettes */
    float downFactor;

    /** \brief number of iterations in morphology closing to generate silhouettes */
    int closingIterationsCount;

    /** \brief the first Canny threshold to find edges on a test scene */
    double cannyThreshold1;

    /** \brief the second Canny threshold to find edges on a test scene */
    double cannyThreshold2;

    /** \brief number of iterations in morhology dilation to suppress Canny edges
     *
     * Mask of segmented glass is dilated and Canny edges which are not in the dilated mask are removed
     */
    int dilationsForEdgesRemovalCount;

    /** \brief minimum number of points in a contour of a glass mask
     *
     * If a contour has less points than this number then it is skipped and initial poses are not generated for this contour.
     */
    size_t minGlassContourLength;

    /** \brief minimum area of a contour of a glass mask
     *
     * If a contour has area less than this number then it is skipped and initial poses are not generated for this contour.
     */
    double minGlassContourArea;

    /** \brief minimum area of a contour of a glass mask
     *
     * If a contour has area less than this number then it is skipped and initial poses are not generated for this contour.
     */

//    float confidentDomination;
//    int icp2dIterationsCount;
//    float min2dScaleChange;

    /** \brief use the closed form solution to get 3D pose from 2D correspondences */
    bool useClosedFormPnP;

    /** \brief granularity of a hash table in geometric hashing */
    float ghGranularity;

    /** \brief step with which train silhouettes will be sampled */
    int ghBasisStep;

    /** \brief minimum distance between basis points to be used in hashing */
    float ghMinDistanceBetweenBasisPoints;

    /** \brief step with which test silhouettes will be sampled */
    int ghTestBasisStep;

    /** \brief length of the object contour relative to length of the whole extracted contour */
    float ghObjectContourProportion;

    /** \brief probability to find a basis which belongs to the object */
    float ghSuccessProbability;

    /** \brief size of a window to be used in suppression of geometric hashing votes */
    int votesWindowSize;

    /** \brief ratio between maximum vote and current vote to suppress it */
    float votesConfidentSuppression;

    /** \brief ratio between maximum confidence and current confidence to suppress it */
    float basisConfidentSuppression;

    /** \brief angular distance between 3D poses which are considered as neighbors in suppression */
    float maxRotation3D;

    /** \brief translational distance between 3D poses which are considered as neighbors in suppression */
    float maxTranslation3D;

    /** \brief minimum scale of the object on a test scene */
    float minScale;

    float maxScale;

    /** \brief ratio between maximum confidence and current confidence to suppress it after alignment to a table */
    float ratioToMinimum;

    /** \brief angular distance between 3D poses which are considered as neighbors in suppression after alignment to a table */
    float neighborMaxRotation;

    /** \brief translational distance between 3D poses which are considered as neighbors in suppression after alignment to a table */
    float neighborMaxTranslation;

    /** \brief parameters to refine poses while finding the correct pose */
    LocalPoseRefinerParams lmInitialParams;

    /** \brief parameters to refine the found final pose */
    LocalPoseRefinerParams lmFinalParams;

    /** \brief criteria for Levenberg-Marquardt to get the jacobian value */
    cv::TermCriteria lmJacobianCriteria;

    /** \brief criteria for Levenberg-Marquardt to get the error value */
    cv::TermCriteria lmErrorCriteria;

    PoseEstimatorParams()
    {
      silhouetteCount = 10;
//      silhouetteCount = 60;
      downFactor = 1.0f;
      closingIterationsCount = 10;

      minGlassContourLength = 20;
      minGlassContourArea = 64.0;

      cannyThreshold1 = 25;
      cannyThreshold2 = 50;
      dilationsForEdgesRemovalCount = 10;

//      confidentDomination = 1.5f;
//      icp2dIterationsCount = 50;
//      min2dScaleChange = 0.001f;

      useClosedFormPnP = true;

      ghBasisStep = 2;
      ghMinDistanceBetweenBasisPoints = 0.1f;
      ghGranularity = 0.04f;

      ghTestBasisStep = 4;

      ghObjectContourProportion = 0.1f;
      ghSuccessProbability = 0.99f;

      votesWindowSize = 5;
      votesConfidentSuppression = 1.1f;
      basisConfidentSuppression = 1.3f;

      maxRotation3D = 0.8f;
      maxTranslation3D = 0.15f;

//      minScale = 0.2f;
      minScale = 1.0 / 0.9;
      maxScale = 1.0 / 0.4;

      ratioToMinimum = 2.0f;
      neighborMaxRotation = 0.1f;
      neighborMaxTranslation = 0.02f;

      lmInitialParams.lmDownFactor = 0.5f;
      lmInitialParams.lmClosingIterationsCount = 5;
      lmFinalParams.lmDownFactor = 1.0f;
      lmFinalParams.lmClosingIterationsCount = 10;
      lmJacobianCriteria = cv::TermCriteria(CV_TERMCRIT_ITER, 5, 0.0);
      lmErrorCriteria = cv::TermCriteria(CV_TERMCRIT_ITER, 1, 0.0);
    }

    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
  };

  /** \brief The class to estimate pose of one transparent object */
  class PoseEstimator
  {
  public:
    /** \brief The constructor
     *
     * \param camera test camera
     * \param params parameters of the pose estimator
     */
    PoseEstimator(const PinholeCamera &camera = PinholeCamera(), const PoseEstimatorParams &params = PoseEstimatorParams());

    /** \brief Sets the edge model of a transparent object
     *
     * \param edgeModel edge model of a transparent object for which you want to estimate poses
     */
    void setModel(const EdgeModel &edgeModel);

    EdgeModel getModel() const;

    /** \brief Estimates pose of a transparent object
     *
     * \param edgeModel edge model of a transparent object for which you want to estimate poses
     * \param bgrImage BGR image of a test scene
     * \param glassMask mask of segmented glass
     * \param poses_cam estimated poses of the object
     * \param poseQualities qualities of the corresponding estimated poses (less is better)
     * \param tablePlane equation of a table plane if the object is known to stay on this plane
     * \param initialSilhouettes silhouettes of initial poses (used for debugging)
     */
    void estimatePose(const cv::Mat &bgrImage, const cv::Mat &glassMask,
                      std::vector<PoseRT> &poses_cam, std::vector<float> &poseQualities,
                      const cv::Vec4f *tablePlane = 0,
                      std::vector<cv::Mat> *initialSilhouettes = 0, std::vector<PoseRT> *initialPoses = 0) const;

    void refinePosesBySupportPlane(const cv::Mat &bgrImage, const cv::Mat &glassMask, const cv::Vec4f &tablePlane,
                                   std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities) const;

    /** \brief Reads a pose estimator from a file
     *
     * \param filename Name of a file where a pose estimator is stored
     */
    void read(const std::string &filename);

    /** \brief Reads a pose estimator from a file node
     *
     * \param fn file node where a pose estimator is stored
     */
    void read(const cv::FileNode &fn);

    /** \brief Writes a pose estimator to a file
     *
     * \param filename Name of a file where a pose estimator will be stored
     */
    void write(const std::string &filename) const;

    /** \brief Writes a pose estimator to a file storage
     *
     * \param fs file storage where a pose estimator will be stored
     */
    void write(cv::FileStorage &fs) const;

    /** \brief Get size of a test image
     *
     * \return size of a test image which is valid for this pose estimator
     */
    cv::Size getValidTestImageSize() const;

    /** \brief Visualizes an estimated pose
     *
     * \param pose pose to visualize
     * \param image image where pose will be visualized
     * \param color color of the object in visualization
     */
    void visualize(const PoseRT &pose, cv::Mat &image,
                   cv::Scalar color = cv::Scalar(0, 0, 255), float blendingFactor = 1.0f) const;

    float computeBlendingFactor(float error) const;

  #ifdef USE_3D_VISUALIZATION
    /** \brief Visualizes an estimated pose in 3D
     *
     * \param pose pose to visualize
     * \param viewer viewer to visualize the object
     * \param color color of the object in visualization
     * \param title title of a point cloud of the object
     */
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


    void suppressBasisMatches(std::vector<BasisMatch> &matches) const;

    void suppressBasisMatchesIn3D(std::vector<BasisMatch> &matches) const;
    void suppress3DPoses(const std::vector<float> &errors, const std::vector<PoseRT> &poses_cam,
                         float neighborMaxRotation, float neighborMaxTranslation,
                         std::vector<bool> &isFilteredOut) const;

    void generateGeometricHashes();
    void computeCentralEdges(const cv::Mat &centralBgrImage, const cv::Mat &glassMask, cv::Mat &centralEdges, cv::Mat &silhouetteEdges) const;
    void getInitialPoses(const cv::Mat &glassMask, std::vector<PoseRT> &initialPoses, std::vector<float> &initialPosesQualities) const;
    void getInitialPosesByGeometricHashing(const cv::Mat &glassMask, std::vector<PoseRT> &initialPoses, std::vector<float> &initialPosesQualities, std::vector<cv::Mat> *initialSilhouettes) const;

    void refineInitialPoses(const cv::Mat &testBgrImage, const cv::Mat &testEdges, const cv::Mat &silhouetteEdges,
                            std::vector<PoseRT> &initPoses_cam, std::vector<float> &initPosesQualities,
                            const LocalPoseRefinerParams &lmParams = LocalPoseRefinerParams(), std::vector<cv::Mat> *jacobians = 0) const;
    void findTransformationToTable(PoseRT &pose_cam, const cv::Vec4f &tablePlane, float &rotationAngle, const cv::Mat finalJacobian = cv::Mat()) const;
    void refinePosesByTableOrientation(const cv::Vec4f &tablePlane, const cv::Mat &testBgrImage, const cv::Mat &testEdges, const cv::Mat &silhouetteEdges, std::vector<PoseRT> &poses_cam, std::vector<float> &initPosesQualities) const;
    void refineFinalTablePoses(const cv::Vec4f &tablePlane, const cv::Mat &testBgrImage, const cv::Mat &testEdges, const cv::Mat &silhouetteEdges,
                      std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities) const;

    EdgeModel edgeModel;
    std::vector<Silhouette> silhouettes;
    std::vector<cv::Mat> canonicScales;
    cv::Ptr<GHTable> ghTable;
    PoseEstimatorParams params;
    PinholeCamera kinectCamera;
  };
}

#endif /* POSEESTIMATOR_HPP_ */

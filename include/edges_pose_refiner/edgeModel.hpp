/*
 * edgeModel.hpp
 *
 *  Created on: Mar 14, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef EDGEMODEL_HPP_
#define EDGEMODEL_HPP_

#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/utils.hpp"
#include "edges_pose_refiner/silhouette.hpp"

/** \brief The 3D model that represents edges of an object in 3D
 *
 * The model is a collection of 3d points.
 * Point is added to the model if it projected on an edge frequently.
 * Also the model contains information about the object coordinate system.
 * The origin of this system is the centroid of the object points and axes are computed from the object points poing cloud by PCA
 */
struct EdgeModel
{
  /** \brief 3d coordinates of points
   *
   * These points are constructed by global registration of train point clouds
   */
  std::vector<cv::Point3f> points;

  /** \brief 3d coordinates of edgels
   *
   * These edgels are stable that is they are projected on an edge frequently
   */
  std::vector<cv::Point3f> stableEdgels;

  /** \brief normals of the corresponding points */
  std::vector<cv::Point3f> normals;

  /** \brief orientations of corresponding stable edgels
   *
   * Orientation is direction of a tangent vector at the corresponding 3d edgel to a 3d contour
   */
  std::vector<cv::Point3f> orientations;

  /** \brief 4x4 projection matrix [R, t; 0, 1]
   *
   * It brings points from the object coordinate system to the camera coordinate system
   */
  cv::Mat Rt_obj2cam;

  /** \brief The rotation axis of the object */
  cv::Point3d rotationAxis;

  /** \brief Point on the rotation axis and a table */
  cv::Point3d tableAnchor;


  /** \brief Empty constructor */
  EdgeModel();

  EdgeModel(const std::vector<cv::Point3f> &points, bool centralize);

  /** \brief Create deep copy of the edgeModel object */
  EdgeModel(const EdgeModel &edgeModel);

  /** \brief Create deep copy of the edgeModel object */
  EdgeModel *operator=(const EdgeModel &edgeModel);

  //TODO: remove imageSize from parameters
  /** \brief Compute a footprint (that is a silhouette) of a point set
   * \param points point set
   * \param imageSize image size for possible footprints
   * \param footprintPoints computed footprint
   * \param downFactor scale factor for the footprint
   * \param closingIterationsCount number of closing operations in morphology 
   */
  static void computeFootprint(const std::vector<cv::Point2f> &points, const cv::Size &imageSize, cv::Mat &footprintPoints, float downFactor, int closingIterationsCount);
  
  //TODO: remove imageSize from parameters
  /** \brief Compute a mask (that is a filled footprint) of a point set
   * \param points point set
   * \param imageSize image size for possible footprints
   * \param downFactor scale factor for the footprint
   * \param closingIterationsCount number of closing operations in morphology 
   * \param image computed mask
   * \param tl coordinates of the top-left corner of the computed mask
   */  
  static void computePointsMask(const std::vector<cv::Point2f> &points, const cv::Size &imageSize, float downFactor, int closingIterationsCount, cv::Mat &image, cv::Point &tl);
  
  /** \brief Get a silhouette of the edge model projected on an image
   *
   * \param pinholeCamera camera for the model projection
   * \param pose_cam pose of the model to be projected
   * \param silhouette computed silhouette
   * \param downFactor scale factor for the silhouette
   * \param closingIterationsCount number of closing operations in morphology
   */
  void getSilhouette(const cv::Ptr<const PinholeCamera> &pinholeCamera, const PoseRT &pose_cam, Silhouette &silhouette, float downFactor, int closingIterationsCount) const;

  /** \brief Generate several silhouettes of the edge model projected on an image
   *
   * \param pinholeCamera camera for the model projection
   * \param silhouetteCount Number of silhouettes to generate
   * \param silhouettes generated silhouettes which are distributed uniformly in pose space
   * \param downFactor scale factor for silhouettes
   * \param closingIterationsCount number of closing operations in morphology
   */  
  void generateSilhouettes(const cv::Ptr<const PinholeCamera> &pinholeCamera, int silhouetteCount, std::vector<Silhouette> &silhouettes, float downFactor, int closingIterationsCount) const;

  /** \brief Rotate the edge model using the camera coordinate system
   *
   * \param transformation_cam the transformation in the camera coordinate system
   * \param rotatedEdgeModel rotated edge model
   */
  void rotate_cam(const PoseRT &transformation_cam, EdgeModel &rotatedEdgeModel) const;

  /** \brief Rotate the edge model using the object coordinate system
   *
   * \param transformation_obj the transformation in the object coordinate system
   * \param rotatedEdgeModel rotated edge model
   * \return the 4x4 projection matrix for the camera coordinate system which is used to rotate the edge model
   */
  cv::Mat rotate_obj(const PoseRT &transformation_obj, EdgeModel &rotatedEdgeModel) const;


  /** \brief Get the center of the object model
   *
   * \return the center of the object model, that is mean of all object points
   */
  cv::Point3f getObjectCenter() const;

  /** \brief Clear all data in the edge model */
  void clear();

  /** \brief Visualize the edge model in rviz
   *
   *  Currently the point cloud of edges is displayed
   * \param points_pub A publisher of a point cloud
   */
  void visualize(const ros::Publisher &points_pub);

  void visualize();

  /** Write a model to a file */
  void write(const std::string &filename);
  void write(cv::FileStorage &fs) const;

  /** Read a model from a file */
  void read(const std::string &filename);
  void read(const cv::FileNode &fn);

private:
  //TODO: remove the default parameter
  void rotateToCanonicalPose(const PinholeCamera &camera, PoseRT &model2canonicalPose, float distance = 1.0f);
  static void projectPointsOnAxis(const EdgeModel &edgeModel, cv::Point3d axis, std::vector<float> &projections, cv::Point3d &center_d);
  //TODO: remove the default parameter
  static void setTableAnchor(EdgeModel &edgeModel, float belowTableRatio = 0.01f);
  //TODO: remove the default parameter
  static void setStableEdgels(EdgeModel &edgeModel, float stableEdgelsRatio = 0.9f);
};

/** \brief Parameters of the edge model creating */
struct EdgeModelCreatorParams
{
  /** \brief Threshold 1 for Canny edge detector */
  double cannyThreshold1;

  /** \brief Threshold 2 for Canny edge detector */
  double cannyThreshold2;

  /** \brief k in k-Nearest Neighbors graph
   *
   * tested for knn = 1 only
   */
  int knn;

  /** \brief This parameter is similar to h from Minimum Covariance Determinant estimator.
   *
   * We use partsCount*neighborsRatio vertices to get estimate for centroid and scatter.
   * Decrease this parameter if you want to include instable edges to the model.
   */
  float neighborsRatio;

  /**
   * This parameter is used in removing of close adjacent vertices after centroid computing.
   * Decreasing of this parameter will cause more aggressive vertices removal and the created model will have less points
   */
  float inliersRatio;

  /** \brief This parameter is similar to c-steps count from Fast Minimum Covariance Determinant estimator.
   *
   * Decreasing of this parameter accelerates the edge model creating but it also may decrease accuracy.
   */
  size_t cstepsCount;

  /** \brief assumed outliers ratio in the created edge model
   *
   *  We are removing finalModelOutliersRatio*|EdgeModel.stableEdgels| of the latest added centroids because they are likely to be outliers
   */
  float finalModelOutliersRatio;

  /** \brief Ratio of points to keep after the model downsampling */
  float downsamplingRatio;

  /** \brief Create the edge model or the model of the whole object */
  bool useOnlyEdges;

  EdgeModelCreatorParams()
  {
    cannyThreshold1 = 180;
    cannyThreshold2 = 100;

    knn = 1;
    neighborsRatio = 0.5f;
    inliersRatio = 0.33f;
    cstepsCount = 5;
    finalModelOutliersRatio = 0.1f;
    downsamplingRatio = 0.1f;

    useOnlyEdges = true;
  }
};

/** \brief The creator of the 3D edge model
 *
 * This class is used to create the edge model which is used for further pose refinement
 */
class EdgeModelCreator
{
public:
  /** \brief The data for one view of the object */
  struct TrainSample
  {
    /** \brief A grayscale image of the object */
    cv::Mat image;

    /** \brief A mask of the object
     *
     * It is an image in which background pixels are colored by black
     */
    cv::Mat mask;

    /** \brief A rotation vector of the transformation which brings pointCloud to the common frame */
    cv::Mat rvec;

    /** \brief A translation vector of the transformation which brings pointCloud to the common frame */
    cv::Mat tvec;

    /** \brief Point cloud of the scene */
    std::vector<cv::Point3f> pointCloud;

    bool isValid() const;
  };

  /**
   * \param cameraMatrix intrinsic parameters of the train camera matrix
   * \param distCoeffs distortion coefficients of the train camera matrix
   * \param pointsPublisher pass non-null pointer if you want to see visualization of the algorithm work in rviz (used for debugging)
   * \param params Parameters of the edge model creating
   */
  EdgeModelCreator(const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const ros::Publisher *pointsPublisher = 0,  const EdgeModelCreatorParams &params =
      EdgeModelCreatorParams());

  /** \brief Set parameters of the edge model creating
   *
   * \param params Parameters of the edge model creating
   */
  void setParams(const EdgeModelCreatorParams &params);

  /** \brief Create the edge model
   *
   * \param trainSamples data associated with several views of the object
   * \param edgeModel the created edge model
   * \param computeOrientations use or not edgels orientation in the model
   */
  void createEdgeModel(const std::vector<TrainSample> &trainSamples, EdgeModel &edgeModel, bool computeOrientations =
      false);

  /** \brief Align the edge model with the train data
   *
   * \param trainSamples data associated with several views of the object
   * \param edgeModel the aligned edge model
   * \param numberOfICPs number of ICPs to align edge model with train point clouds
   * \param numberOfIterationsInICP maximum number of iterations in each ICP
   */
  void alignModel(const std::vector<TrainSample> &trainSamples, EdgeModel &edgeModel, int numberOfICPs = 10, int numberOfIterationsInICP = 100);

  /** \brief Reduce number of points in the model 
   *  
   *  \param edgeModel the edge model to downsample
   *  \param ratio ratio between a new number of points in the model and the old number
   */ 
  void downsampleEdgeModel(EdgeModel &edgeModel, float ratio);
  
  /** \brief Get all point clouds corresponding to edges
   * 
   *  \param trainSamples data associated with several views of the object
   *  \param edgePointClouds point clouds corresponding to edges for each train sample
   */
  void getEdgePointClouds(const std::vector<TrainSample> &trainSamples,
                          std::vector<std::vector<cv::Point3f> > &edgePointClouds);

  /** \brief Run k-partite matching on input point clouds
   * 
   *  \param pointClouds Registered input point clouds
   *  \param matchedPointCloud Output point cloud after k-partite matching applied to pointClouds
   */
  void matchPointClouds(const std::vector<std::vector<cv::Point3f> > &pointClouds, std::vector<cv::Point3f> &matchedPointCloud);

  /** \brief Compute stable edgels of the model
   * 
   *  \param trainSamples data associated with several views of the object
   *  \param edgeModel Edge model for which stable edgels will be computed by using its points
   *  \param dilationsIterations Number of iterations to dilate samples' masks
   *  \param maxDistanceToEdge If distance between a projected point and the nearest edge is less then this param then the point is considered as edgel
   *  \param minRepeatability If a point is considered as edgel in (minRepeatability * number of train samples) cases then it is stable edgel
   */
  void computeStableEdgels(const std::vector<TrainSample> &trainSamples, EdgeModel &edgeModel, int dilationsIterations = 3, float maxDistanceToEdge = 3.99f, float minRepeatability = 0.8f);

  static void computeObjectSystem(const std::vector<cv::Point3f> &points, cv::Mat &Rt_obj2cam);
private:
  void computeModelEdgels(const std::vector<TrainSample> &trainSamples, std::vector<cv::Point3f> &edgels);
  void downsamplePointCloud(const std::vector<cv::Point3f> &pointCloud, std::vector<cv::Point3f> &downampledPointCloud,
                            std::vector<int> &srcIndices);

  cv::Mat cameraMatrix, distCoeffs;
  EdgeModelCreatorParams params;
  const ros::Publisher *pointsPublisher;
};

#endif /* EDGEMODEL_HPP_ */

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

struct EdgeModelCreationParams
{
  /** \brief Index of a neighbor to compute a median distance of all points */
  int neighborIndex;

  /** \brief maximum ratio between the largest Chamfer distance and the median distance to consider an object as symmetric */
  float distanceFactor;

  /** \brief number of rotations to compute Chamfer distance */
  int rotationCount;

  /** \brief ratio of object points below table
   *
   *  It is used to be robust to outliers
   */
  float belowTableRatio;

  /** \brief ratio of object points which are considered as unstable edgels
   *
   *  Stable edgels are located on the top of an object because they produces edges often.
   */
  float stableEdgelsRatio;

  EdgeModelCreationParams()
  {
    neighborIndex = 1;
    distanceFactor = 2.0f;
    rotationCount = 60;

    belowTableRatio = 0.01f;
    stableEdgelsRatio = 0.9f;
  }
};

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

 /** \brief Direction of a normal to a table when the object is placed on this table */
  cv::Point3d upStraightDirection;

 /** \brief Does the object have rotation symmetry aroung upStraightDirection or not */
  bool hasRotationSymmetry;

  /** \brief Point on the rotation axis and a table */
  cv::Point3d tableAnchor;

  /** \brief Empty constructor */
  EdgeModel();

  //TODO: remove this constructor
  EdgeModel(const std::vector<cv::Point3f> &points, bool isModelUpsideDown, bool centralize, const EdgeModelCreationParams &params = EdgeModelCreationParams());

  EdgeModel(const std::vector<cv::Point3f> &points, const std::vector<cv::Point3f> &normals, bool isModelUpsideDown, bool centralize, const EdgeModelCreationParams &params = EdgeModelCreationParams());

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

  void computeWeights(const PoseRT &pose_cam, double decayConstant, double maxWeight, cv::Mat &weights, cv::Mat *jacobian = 0) const;

  //TODO: remove imageSize from parameters
  /** \brief Compute a mask (that is a filled footprint) of a point set
   * \param points point set
   * \param imageSize image size for possible footprints
   * \param downFactor scale factor for the footprint
   * \param closingIterationsCount number of closing operations in morphology 
   * \param image computed mask
   * \param tl coordinates of the top-left corner of the computed mask
   */  
  static void computePointsMask(const std::vector<cv::Point2f> &points, const cv::Size &imageSize, float downFactor, int closingIterationsCount, cv::Mat &image, cv::Point &tl, bool cropMask = true);
  
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

 // cv::Vec3f getBoundingBox() const;

  std::vector<std::pair<float, float> > getObjectRanges() const;


  /** \brief Clear all data in the edge model */
  void clear();

  /** \brief Visualize the edge model in a PCL viewer */
  void visualize();

  /** \brief Write a model to a file
   *
   *  \param filename Name of the file to write the model
   */
  void write(const std::string &filename) const;

  /** \brief Write a model to a file storage
   *
   *  \param fs file storage to write the model
   */
  void write(cv::FileStorage &fs) const;

  /** \brief Read a model from a file
   *
   *  \param filename Name of the file to read the model
   */
  void read(const std::string &filename);

  /** \brief Read a model from a file node
   *
   *  \param fn file node to read the model
   */
  void read(const cv::FileNode &fn);

  //TODO: move to private
  void rotateToCanonicalPose(const PinholeCamera &camera, PoseRT &model2canonicalPose, float distance = 1.0f);
  static void computeSurfaceEdgelsOrientations(EdgeModel &edgeModel);
private:
  EdgeModelCreationParams params;

  static bool isAxisCorrect(const std::vector<cv::Point3f> &points, cv::Point3f rotationAxis, int neighborIndex, float distanceFactor, int rotationCount);
  //TODO: remove the default parameter
  static void projectPointsOnAxis(const EdgeModel &edgeModel, cv::Point3d axis, std::vector<float> &projections, cv::Point3d &center_d);
  static void setTableAnchor(EdgeModel &edgeModel, float belowTableRatio);
  static void setStableEdgels(EdgeModel &edgeModel, float stableEdgelsRatio);
};

void computeObjectSystem(const std::vector<cv::Point3f> &points, cv::Mat &Rt_obj2cam);

#endif /* EDGEMODEL_HPP_ */

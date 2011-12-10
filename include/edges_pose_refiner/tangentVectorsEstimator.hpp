/*
 * tangentVectorsEstimator.hpp
 *
 *  Created on: Apr 5, 2011
 *      Author: ilysenkov
 */

#ifndef TANGENTVECTORSESTIMATOR_HPP_
#define TANGENTVECTORSESTIMATOR_HPP_

#include <opencv2/core/core.hpp>
#include <list>

struct Graph
{
  std::vector<std::vector<int> > nearestNeighborsOut;
  std::vector<std::vector<int> > nearestNeighborsIn;
};

/** \brief Parameters of tangent vectors estimation */
struct TangentVectorEstimatorParams
{
  /** \brief knn for KNN-graph */
  int knn;

  /** \brief minimum number of points to estimate orientation when following a contour */
  int minEstimationPointsCount;

  /** \brief maximum number of points to estimate orientation */
  int maxEstimationPointsCount;

  /** \brief p-quantile of knn distances */
  float distanceQuantile;

  /** \brief maximum allowed ratio of a neighbor distance to p-quantile of knn distances */
  float distanceQuantileFactor;

  /** \brief h for robust centroid */
  float robustCentroidH;

  /** \brief maximum number of points to estimate orientation in final estimation */
  int maxHalfContourPoints;

  /** \brief minimum number of points to estimate orientation in final estimation */
  int minHalfContourPoints;

  /** \brief default value for cases when orientation cann't be computed */
  cv::Point3f nanOrientation;

  TangentVectorEstimatorParams()
  {
    knn = 10;
    minEstimationPointsCount = 6;
    maxEstimationPointsCount = 24;

    distanceQuantile = 0.5f;
    distanceQuantileFactor = 2.0f;

    robustCentroidH = 0.5f;

    maxHalfContourPoints = 16;
    minHalfContourPoints = 4;

    nanOrientation = cv::Point3f(0, 0, 0);
  }

};


/** \brief Estimate tangent vectors to a point cloud */
class TangentVectorEstimator
{
public:
  /**
   *  \param params Parameters of tangent vectors estimation
   */
  TangentVectorEstimator(const TangentVectorEstimatorParams &params = TangentVectorEstimatorParams());

  /** \brief Estimate tangent vectors to every point of a point cloud
   *
   *  \param pointCloud A point cloud for which tangent vectors will be estimated
   *  \param tangentVectors Estimated tangent vectors to corresponding points of the point cloud
   */
  void estimate(const std::vector<cv::Point3f> &pointCloud, std::vector<cv::Point3f> &tangentVectors);
private:
  void constructKNNGraph(const std::vector<cv::Point3f> &pointCloud, Graph &graph, float &distanceQuantile);
  void followContourForward(const std::vector<cv::Point3f> &pointCloud, const std::vector<std::vector<int> >  &edges, const std::vector<bool> &isTangentVectorEstimated, float distanceQuantile, std::vector<int> &contour, std::list<cv::Point3f> &orientations);

  //void publishContours(const std::vector<cv::Point3f> &model, const std::vector<std::vector<int> > &contours, const std::vector<std::list<cv::Point3f> > &orientations, int numberOfContours);
  cv::Point3f estimateOrientation(const std::vector<cv::Point3f> &pointCloud, const std::vector<int> &contour);
  void estimateFinalOrientations(const std::vector<cv::Point3f> &contour, std::vector<cv::Point3f> &orientations);

  cv::Point3f getRobustCentroid(const std::vector<cv::Point3f> &points);

  TangentVectorEstimatorParams params;
};

#endif /* TANGENTVECTORSESTIMATOR_HPP_ */

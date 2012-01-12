/*
 * registerPointClouds.hpp
 *
 *  Created on: Mar 15, 2011
 *      Author: ilysenkov
 */

#ifndef REGISTERPOINTCLOUDS_HPP_
#define REGISTERPOINTCLOUDS_HPP_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/core/core.hpp>

/** \brief Parameters of multi-view registration */
struct MultiViewRegistratorParams
{
  /** \brief searches count used to find nearest neighbor in filtering of a point cloud */
  int flannSearchesCount;

  /** \brief ratio of inliers in filtering of a point cloud */
  float inliersRatio;

  /** \brief factor of the maximum inlier distance
   *
   * max allowed distance to nearest neighbor for inlier is product of this factor and distance of the most distant inlier
   */
  float maxInlierDistFactor;

  /** \brief count of rounds in multi-view registration
   *
   * Round is alignment of each point cloud to a target point cloud
   */
  int roundsCount;

  /** \brief maximum number of iteration in each run of ICP */
  int maxIterations;

  MultiViewRegistratorParams()
  {
    flannSearchesCount = 256;
    inliersRatio = 0.8;
    maxInlierDistFactor = 1.5;
    roundsCount = 10;
    maxIterations = 10;
  }
};

/** \brief Register multiple point clouds
 *
 *  This is a very simple multi-view registration class. It uses pairwise ICP in similar way to Bergevin et al.
 */
class MultiViewRegistrator
{
public:
  /**
   * \param params Parameters of multi-view registration
   */
  MultiViewRegistrator(const MultiViewRegistratorParams &params = MultiViewRegistratorParams());

  /**
   * \brief Set parameters of multi-view registration
   * \param params Parameters of multi-view registration
   */
  void setParams(const MultiViewRegistratorParams &params);

  /** \brief Register multiple PCL point clouds
   *
   *  \param inputPointClouds PCL point clouds to register
   *  \param outputPointClouds registered PCL point clouds
   */
  void align(const std::vector<pcl::PointCloud<pcl::PointXYZ> > &inputPointClouds, std::vector<pcl::PointCloud<pcl::PointXYZ> > &outputPointClouds) const;

  /** \brief Register multiple OpenCV point clouds
   *
   *  \param inputPointClouds OpenCV point clouds to register
   *  \param outputPointClouds registered OpenCV point clouds
   */
  void align(const std::vector<std::vector<cv::Point3f> > &inputPointClouds, std::vector<std::vector<cv::Point3f> > &outputPointClouds) const;

private:
  //void getPCLPointClouds(Object *object, vector<pcl::PointCloud<pcl::PointXYZ> > &pointClouds);

  void filterPointCloudByNearestNeighborDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr &targetCloudPtr, const pcl::PointCloud<pcl::PointXYZ> &inputCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &filteredCloudPtr) const;

  MultiViewRegistratorParams params;
};

//void getPointClouds(Object *object, std::vector<std::vector<cv::Point3f> > &pointClouds);

#endif /* REGISTERPOINTCLOUDS_HPP_ */

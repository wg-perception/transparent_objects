/*
 * pcl.cpp
 *
 *  Created on: 12/21/2012
 *      Author: ilysenkov
 */

#include "edges_pose_refiner/pcl.hpp"

void pcl2cv(const pcl::PointCloud<pcl::PointXYZ> &pclCloud, std::vector<cv::Point3f> &cvCloud)
{
  cvCloud.resize(pclCloud.size());

  for(size_t i=0; i<pclCloud.size(); i++)
  {
    cvCloud[i] = cv::Point3f(pclCloud.points[i].x, pclCloud.points[i].y, pclCloud.points[i].z);
  }
}

void cv2pcl(const std::vector<cv::Point3f> &cvCloud, pcl::PointCloud<pcl::PointXYZ> &pclCloud)
{
  pclCloud.points.resize(cvCloud.size());
  for(size_t i=0; i<cvCloud.size(); i++)
  {
    cv::Point3f pt = cvCloud[i];
    pclCloud.points[i] = pcl::PointXYZ(pt.x, pt.y, pt.z);
  }
}

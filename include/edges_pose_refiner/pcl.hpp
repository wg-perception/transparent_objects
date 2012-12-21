/*
 * pcl.hpp
 *
 *  Created on: 12/21/2012
 *      Author: ilysenkov
 */

#ifndef PCL_HPP
#define PCL_HPP

#include <opencv2/core/core.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

void pcl2cv(const pcl::PointCloud<pcl::PointXYZ> &pclCloud, std::vector<cv::Point3f> &cvCloud);
void cv2pcl(const std::vector<cv::Point3f> &cvCloud, pcl::PointCloud<pcl::PointXYZ> &pclCloud);

#endif // PCL_HPP

/*
 * pclVisualization.hpp
 *
 *  Created on: 12/24/2012
 *      Author: ilysenkov
 */

#ifndef PCLVISUALIZATION_HPP
#define PCLVISUALIZATION_HPP

#define USE_3D_VISUALIZATION

#ifdef USE_3D_VISUALIZATION
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/utils.hpp"

void publishPoints(const std::vector<cv::Point3f>& points, const boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, cv::Scalar color = cv::Scalar(0, 0, 255), const std::string &title = "", const PoseRT &pose = PoseRT());
#endif


#endif // PCLVISUALIZATION_HPP

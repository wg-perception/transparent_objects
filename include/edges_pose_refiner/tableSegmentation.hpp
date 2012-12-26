/*
 * tableSegmentation.hpp
 *
 *  Created on: 12/21/2012
 *      Author: ilysenkov
 */

#ifndef TABLESEGMENTATION_HPP
#define TABLESEGMENTATION_HPP

#include "edges_pose_refiner/pinholeCamera.hpp"

bool computeTableOrientationByPCL(float downLeafSize, int kSearch, float distanceThreshold, const std::vector<cv::Point3f> &fullSceneCloud,
                                  cv::Vec4f &tablePlane,  const PinholeCamera *camera = 0, std::vector<cv::Point2f> *tableHull = 0,
                                  float clusterTolerance = 0.05f, cv::Point3f verticalDirection = cv::Point3f(0.0f, -1.0f, 0.0f));

bool computeTableOrientationByRGBD(const cv::Mat &depth, const PinholeCamera &camera,
                                   cv::Vec4f &tablePlane, std::vector<cv::Point> *tableHull = 0,
                                   cv::Point3f verticalDirection = cv::Point3f(0.0f, -1.0f, 0.0f));

int computeTableOrientationByFiducials(const PinholeCamera &camera, const cv::Mat &bgrImage,
                                        cv::Vec4f &tablePlane);

void drawTable(const std::vector<cv::Point2f> &tableHull, cv::Mat &image,
               cv::Scalar color = cv::Scalar(0, 255, 0), int thickness = 2);

#endif // TABLESEGMENTATION_HPP

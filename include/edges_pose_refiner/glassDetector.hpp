/*
 * glassDetector.hpp
 *
 *  Created on: Sep 26, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef GLASSDETECTOR_HPP_
#define GLASSDETECTOR_HPP_

#include <opencv2/core/core.hpp>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "edges_pose_refiner/pinholeCamera.hpp"

struct GlassSegmentationParams
{
  /** \brief Number of closing iterations in morphology */
  int closingIterations;

  /** \brief openingIterations Number of opening iterations in morphology */
  int openingIterations;

  /** \brief finalClosingIterations Number of final closing iterations in morphology */
  int finalClosingIterations;

  /** \brief Use grab cut to refine a segmented mask or not */
  bool useGrabCut;

  /** \brief Number of grab cut iteration to refine a segmented mask */
  int grabCutIterations;

  /** \brief Width of the region where grab cut will be used */
  int grabCutErosionsIterations;

  //TODO: do you need this parameter?
  /** \brief Additional width of grub cut ROI */
  int grabCutMargin;

  /** \brief Refine a segmentation mask with convexity assumption or not */
  bool fillConvex;

  GlassSegmentationParams()
  {
    closingIterations = 12;
    openingIterations = 6;
    finalClosingIterations = 15;

    useGrabCut = true;
    grabCutIterations = 2;
    grabCutErosionsIterations = 6;
    grabCutMargin = 10;

    fillConvex = false;
  }
};

class GlassSegmentator
{
public:
  GlassSegmentator(const GlassSegmentationParams &params = GlassSegmentationParams());

  /** \brief Segment glass on an image using a Kinect
   * \param bgrImage A color image returned by a Kinect
   * \param depthMat A depth map returned by a Kinect
   * \param registrationMask A mask of invalid values in a depth map when a Kinect returns the registered depth
   * \param numberOfComponents Number of connected components in glass segmentation
   * \param glassMask Mask with computed segmentation: white is glass, black is background
   * \param camera Pinhole camera to project 3d points on an image
   * \param tablePlane Coefficients of the test table plane
   * \param tableHull Convex hull of the test table plane
   */
  void segment(const cv::Mat &bgrImage, const cv::Mat &depthMat, const cv::Mat &registrationMask, int &numberOfComponents, cv::Mat &glassMask, const PinholeCamera *camera = 0, const cv::Vec4f *tablePlane = 0, const pcl::PointCloud<pcl::PointXYZ> *tableHull = 0);

private:
  GlassSegmentationParams params;
};

#endif

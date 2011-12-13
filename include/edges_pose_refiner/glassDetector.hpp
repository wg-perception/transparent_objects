/*
 * glassDetector.hpp
 *
 *  Created on: Sep 26, 2011
 *      Author: Ilya Lysenkov
 */

#include <opencv2/core/core.hpp>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "edges_pose_refiner/pinholeCamera.hpp"

struct GlassSegmentationParams
{
  int closingIterations, openingIterations, finalClosingIterations;

  bool useGrabCut;
  int grabCutIterations;
  int grabCutErosionsIterations;
  int grabCutMargin;

  bool fillConvex;

  GlassSegmentationParams()
  {
    closingIterations = 12;
    openingIterations = 6;
    finalClosingIterations = 15;

    useGrabCut = true;
    grabCutIterations = 2;
    grabCutErosionsIterations = 10;
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
   * \param numberOfComponents Number of connected components in glass segmentation
   * \param glassMask Mask with computed segmentation: white is glass, black is background
   * \param closingIterations Number of closing iterations in morphology
   * \param openingIterations Number of opening iterations in morphology
   * \param finalClosingIterations Number of final closing iterations in morphology
   */
  void segment(const cv::Mat &bgrImage, const cv::Mat &depthMat, int &numberOfComponents, cv::Mat &glassMask, const PinholeCamera *camera = 0, const cv::Vec4f *tablePlane = 0, const pcl::PointCloud<pcl::PointXYZ> *tableHull = 0);

private:
  GlassSegmentationParams params;
};

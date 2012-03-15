#ifndef GEODESIC_ACTIVE_CONTOURS_
#define GEODESIC_ACTIVE_CONTOURS_

#include <opencv2/core/core.hpp>

struct GeodesicActiveContourParams
{
  float beta;
  float alpha;

  float propagationScaling;
  float curvatureScaling;
  float advectionScaling;

  float maximumRMSError;
  int numberOfIterations;

  GeodesicActiveContourParams()
  {
    beta = 7.0f;
    alpha = -0.86f;

    //propagationScaling = -0.3f;
    propagationScaling = -0.02f;
    curvatureScaling = 1.0f;
    advectionScaling = 1.0f;

    maximumRMSError = 0.001f;
    numberOfIterations = 60000;
  }
};

void geodesicActiveContour(const cv::Mat &bgrImage, const cv::Mat &edges, cv::Mat &segmentation,
                           const GeodesicActiveContourParams &params = GeodesicActiveContourParams());

#endif

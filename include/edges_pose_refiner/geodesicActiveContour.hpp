#ifndef GEODESIC_ACTIVE_CONTOURS_
#define GEODESIC_ACTIVE_CONTOURS_

#include <opencv2/core/core.hpp>

void geodesicActiveContour(const cv::Mat &edges, cv::Mat &segmentation);

#endif

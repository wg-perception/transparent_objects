/*
 * glassDetector.hpp
 *
 *  Created on: Sep 26, 2011
 *      Author: Ilya Lysenkov
 */

#include <opencv2/core/core.hpp>

/** \brief Segment glass on an image using a Kinect
 * \param bgrImage A color image returned by a Kinect
 * \param depthMat A depth map returned by a Kinect
 * \param numberOfComponents Number of connected components in glass segmentation
 * \param glassMask Mask with computed segmentation: white is glass, black is background
 * \param closingIterations Number of closing iterations in morphology
 * \param openingIterations Number of opening iterations in morphology
 * \param finalClosingIterations Number of final closing iterations in morphology
 */
void findGlassMask(const cv::Mat &bgrImage, const cv::Mat &depthMat, int &numberOfComponents, cv::Mat &glassMask, int closingIterations = 8, int openingIterations = 12, int finalClosingIterations = 5 );

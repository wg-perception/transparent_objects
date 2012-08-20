#include "fdcm/fdcm.hpp"
#include "fdcm/image/Image.h"
#include "fdcm/fitline/LFLineFitter.h"
#include "fdcm/fdcm/LMDistanceImage.h"

#include <opencv2/opencv.hpp>

using namespace cv;

void cv2fdcm(const Mat &cvImage, Ptr<Image<uchar> > &fdcmImage)
{
  CV_Assert(cvImage.type() == CV_8UC1);

  fdcmImage = new Image<uchar>(cvImage.cols, cvImage.rows, false);

  CV_Assert(cvImage.isContinuous());
  memcpy(fdcmImage->data, cvImage.data, cvImage.total());

  //TODO: remove
  for (int i = 0; i < cvImage.rows; ++i)
  {
    for (int j = 0; j < cvImage.cols; ++j)
    {
      CV_Assert(cvImage.at<uchar>(i, j) == fdcmImage->Access(j, i));
    }
  }
}

//TODO: add const
void fdcm2cv(Image<uchar> &fdcmImage, Mat &cvImage)
{
//  cvImage = Mat(fdcmImage.height(), fdcmImage.width(), CV_8UC1, fdcmImage.data);
  cvImage.create(fdcmImage.height(), fdcmImage.width(), CV_8UC1);
  memcpy(cvImage.data, fdcmImage.data, cvImage.total());

  //TODO: remove
  for (int i = 0; i < cvImage.rows; ++i)
  {
    for (int j = 0; j < cvImage.cols; ++j)
    {
      CV_Assert(cvImage.at<uchar>(i, j) == fdcmImage.Access(j, i));
    }
  }
}

void fdcm2cv(Image<float> &fdcmImage, Mat &cvImage)
{
  cvImage.create(fdcmImage.height(), fdcmImage.width(), CV_32FC1);
  memcpy(cvImage.data, fdcmImage.data, cvImage.total() * sizeof(float));

  //TODO: remove
  for (int i = 0; i < cvImage.rows; ++i)
  {
    for (int j = 0; j < cvImage.cols; ++j)
    {
      CV_Assert(cvImage.at<float>(i, j) == fdcmImage.Access(j, i));
    }
  }
}

void fitLines(const cv::Mat &edges, LFLineFitter &lineFitter)
{
  Ptr<Image<uchar> > fdcmEdges;
  cv2fdcm(edges, fdcmEdges);
  lineFitter.Init();
  lineFitter.FitLine(fdcmEdges);
}

//TODO: remove code duplication with EIEdgeImage::Theta2Index
int theta2Index(float theta, int directionsCount)
{
  int orIndex = (int) floor ((theta * directionsCount) / (M_PI+1e-5));
  if (orIndex < 0 || orIndex >= directionsCount)
  {
    std::stringstream errorMessage;
    errorMessage << theta << " has invalid orIndex: " << orIndex << " / " << directionsCount;
    CV_Error(CV_StsBadArg, errorMessage.str());
  }
  return orIndex;
}

void computeOrientationIndices(const std::vector<cv::Point2f> &points, const cv::Mat &dx, const cv::Mat &dy,
                               std::vector<int> &orientationIndices)
{
  CV_Assert(dx.size() == dy.size());
  CV_Assert(dx.type() == CV_32FC1);
  CV_Assert(dy.type() == CV_32FC1);

  //TODO: move up
  const int directionsCount = 60;

  orientationIndices.clear();
  for (size_t i = 0; i < points.size(); ++i)
  {
    cv::Point pt = points[i];

    //TODO: use isPointInside
    if (0 <= pt.x && pt.x < dx.cols && 0 <= pt.y && pt.y < dx.rows)
    {
      //TODO: use LFLineSegment::Theta()
      //TODO: do you need inverse y-axis?
      double theta = atan2(dy.at<float>(pt), dx.at<float>(pt));
      if (theta<0)
      {
          theta += M_PI;
      }
      int orIndex = theta2Index(theta, directionsCount);
      orientationIndices.push_back(orIndex);
    }
    else
    {
      //TODO: move up
      const int defaultOrIndex = 0;
      orientationIndices.push_back(defaultOrIndex);
    }
  }
}

void computeNormals(const cv::Mat &edges, cv::Mat &normals, cv::Mat &orientationIndices)
{
  //TODO: move up
  const int directionsCount = 60;
  LFLineFitter lineFitter;
  fitLines(edges, lineFitter);

  Mat linearMap(edges.size(), CV_8UC1, Scalar(0));
  Mat linearMapNormals(edges.size(), CV_32FC2, Scalar::all(std::numeric_limits<float>::quiet_NaN()));
  Mat linearMapOrientationIndices(edges.size(), CV_32SC1, Scalar(-1));
  for (int i = 0; i < lineFitter.rNLineSegments(); ++i)
  {
    cv::Point start(lineFitter.outEdgeMap_[i].sx_, lineFitter.outEdgeMap_[i].sy_);
    cv::Point end(lineFitter.outEdgeMap_[i].ex_, lineFitter.outEdgeMap_[i].ey_);

    LineIterator edgelsIterator(linearMap, start, end);
    for(int j = 0; j < edgelsIterator.count; ++j, ++edgelsIterator)
    {
      **edgelsIterator = 255;
      Vec2f normal(lineFitter.outEdgeMap_[i].normal_.x, lineFitter.outEdgeMap_[i].normal_.y);
      linearMapNormals.at<Vec2f>(edgelsIterator.pos()) = normal;
      linearMapOrientationIndices.at<int>(edgelsIterator.pos()) = theta2Index(lineFitter.outEdgeMap_[i].Theta(), directionsCount);
    }
  }
//  imshow("edges", edges);
//  imshow("linearMap", linearMap);
//  waitKey();

  Mat dt, labels;
  distanceTransform(~linearMap, dt, labels, CV_DIST_L2, CV_DIST_MASK_PRECISE, DIST_LABEL_PIXEL);

  CV_Assert(linearMap.type() == CV_8UC1);
  CV_Assert(labels.type() == CV_32SC1);
  std::map<int, cv::Point> label2position;
  for (int i = 0; i < linearMap.rows; ++i)
  {
    for (int j = 0; j < linearMap.cols; ++j)
    {
      if (linearMap.at<uchar>(i, j) != 0)
      {
        //TODO: singal error if the label already exists
        label2position[labels.at<int>(i, j)] = cv::Point(j, i);
      }
    }
  }

  orientationIndices.create(edges.size(), CV_32SC1);
  orientationIndices = -1;

  normals.create(edges.size(), CV_32FC2);
  normals = Scalar::all(std::numeric_limits<float>::quiet_NaN());
  for (int i = 0; i < edges.rows; ++i)
  {
    for (int j = 0; j < edges.cols; ++j)
    {
//      if (edges.at<uchar>(i, j) != 0)
      cv::Point nearestEdgelPosition = label2position[labels.at<int>(i, j)];
      normals.at<Vec2f>(i, j) = linearMapNormals.at<Vec2f>(nearestEdgelPosition);
      orientationIndices.at<int>(i, j) = linearMapOrientationIndices.at<int>(nearestEdgelPosition);
      CV_Assert(orientationIndices.at<int>(i, j) >= 0 && orientationIndices.at<int>(i, j) < directionsCount);
    }
  }

/*
  Mat vis(orientations.size(), CV_8UC1, Scalar(0));
  for (int i = 0; i < orientations.rows; ++i)
  {
    for (int j = 0; j < orientations.cols; ++j)
    {
      Vec2f elem = orientations.at<Vec2f>(i, j);
      vis.at<uchar>(i, j) = (cvIsNaN(elem[0]) || cvIsNaN(elem[1])) ? 0 : 255;
    }
  }
  imshow("final or", vis);
  waitKey();
*/
}

void computeDistanceTransform3D(const cv::Mat &edges,
                                std::vector<cv::Mat> &dtImages)
{
  //TODO: move up
  const float directionCost = 0.5f;
  const double maxCost = 30.0;
  const int nDirections = 60;
  const double scale = 1.0;

  LFLineFitter lineFitter;
  fitLines(edges, lineFitter);

  EIEdgeImage linearEdges;
  linearEdges.SetNumDirections(nDirections);
  linearEdges.Read(lineFitter);
  linearEdges.Scale(scale);

  LMDistanceImage distanceImage;
  distanceImage.Configure(directionCost, maxCost);
  distanceImage.SetImage(linearEdges);

  vector<Image<float> > &fdcmDtImages = distanceImage.getDtImages();
  dtImages.resize(fdcmDtImages.size());

  for (size_t i = 0; i < fdcmDtImages.size(); ++i)
  {
    fdcm2cv(fdcmDtImages[i], dtImages[i]);
  }
}

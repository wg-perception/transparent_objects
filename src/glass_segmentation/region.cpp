#include <numeric>
#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/region.hpp"

using namespace cv;

Region::Region(const cv::Mat &_image, const cv::Mat &_textonLabels, const cv::Mat &_mask)
{
  //TODO: move up
  const int erosionCount = 2;

  image = _image;
  textonLabels = _textonLabels;
  mask = _mask;
//  erode(mask, erodedMask, Mat(), Point(-1, -1), erosionCount);
  erodedMask = mask.clone();
  CV_Assert(countNonZero(erodedMask) != 0);

  computeColorHistogram();
  computeTextonHistogram();
  clusterIntensities();
  computeCenter();
}

cv::Point2f Region::getCenter() const
{
  return center;
}

const cv::Mat& Region::getColorHistogram() const
{
  return hist;
}

const cv::Mat& Region::getTextonHistogram() const
{
  return textonHistogram;
}

const cv::Mat& Region::getIntensityClusters() const
{
  return intensityClusterCenters;
}

const cv::Mat& Region::getMask() const
{
  return mask;
}

void Region::computeCenter()
{
  Point2d sum(0.0, 0.0);
  int pointCount = 0;
  for (int i = 0; i < mask.rows; ++i)
  {
    for (int j = 0; j < mask.cols; ++j)
    {
      if (mask.at<uchar>(i, j) != 0)
      {
        sum += Point2d(j, i);
        ++pointCount;
      }
    }
  }
  sum *= 1.0 / pointCount;
  center = sum;
}

void Region::computeColorHistogram()
{
  //TODO: move up
  const int hbins = 20;
  const int sbins = 20;

  CV_Assert(image.type() == CV_8UC3);
  CV_Assert(erodedMask.type() == CV_8UC1);
  Mat hsv;
  cvtColor(image, hsv, CV_BGR2HSV);

  int histSize[] = {hbins, sbins};
  float hranges[] = {0, 180};
  float sranges[] = {0, 256};
  const float* ranges[] = {hranges, sranges};
  int channels[] = {0, 1};
  calcHist(&hsv, 1, channels, erodedMask, hist, 2, histSize, ranges, true, false);
  hist /= countNonZero(erodedMask);
}

void Region::computeTextonHistogram()
{
  //TODO: move up
  const int textonCount = 36;
  Mat textonLabels_8U;
  textonLabels.convertTo(textonLabels_8U, CV_8UC1);

  int histSize[] = {textonCount};
  float labelRanges[] = {0, textonCount};
  const float* ranges[] = {labelRanges};
  int channels[] = {0};
  int narrays = 1;
  int dims = 1;
  bool isUniform = true;
  bool accumulate = false;
  calcHist(&textonLabels_8U, narrays, channels, erodedMask, textonHistogram, dims, histSize, ranges, isUniform, accumulate);
  textonHistogram /= countNonZero(erodedMask);
}

void Region::clusterIntensities()
{
  //TODO: move up
  const int clusterCount = 10;

  if (grayscaleImage.empty())
  {
    if (image.channels() == 3)
    {
      cvtColor(image, grayscaleImage, CV_BGR2GRAY);
    }
    else
    {
      grayscaleImage = image;
    }
  }
  CV_Assert(grayscaleImage.type() == CV_8UC1);
  CV_Assert(erodedMask.type() == CV_8UC1);
  CV_Assert(grayscaleImage.size() == erodedMask.size());

  vector<int> intensities;
  for (int i = 0; i < erodedMask.rows; ++i)
  {
    for (int j = 0; j < erodedMask.cols; ++j)
    {
      if (erodedMask.at<uchar>(i, j) == 255)
      {
        intensities.push_back(grayscaleImage.at<uchar>(i, j));
      }
    }
  }

  std::sort(intensities.begin(), intensities.end());
  vector<int> boundaries;
  int currentBoundaryIndex = 0;
  int step = intensities.size() / clusterCount;
  int residual = intensities.size() % clusterCount;
  while (currentBoundaryIndex <= intensities.size())
  {
    boundaries.push_back(currentBoundaryIndex);
    currentBoundaryIndex += step;
    if (residual != 0)
    {
      ++currentBoundaryIndex;
      --residual;
    }
  }
  CV_Assert(boundaries.size() == clusterCount + 1);

  vector<float> clusterCenters(clusterCount);
  for (int i = 0; i < clusterCount; ++i)
  {
    int intensitiesSum = std::accumulate(intensities.begin() + boundaries[i], intensities.begin() + boundaries[i + 1], 0);
    clusterCenters[i] = static_cast<float>(intensitiesSum) / (boundaries[i + 1] - boundaries[i]);
  }

  intensityClusterCenters = cv::Mat(clusterCenters).clone();
  CV_Assert(intensityClusterCenters.cols == 1);
}

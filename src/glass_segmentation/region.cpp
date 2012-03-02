#include <numeric>
#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/region.hpp"

using namespace cv;
using std::cout;
using std::endl;

Region::Region()
{
}

Region::Region(const cv::Mat &_image, const cv::Mat &_textonLabels, const cv::Mat &_mask)
{
  image = _image;
  textonLabels = _textonLabels;
  mask = _mask;
  computeErodedMask(mask, erodedMask);
  erodedArea = countNonZero(erodedMask);

  computeColorHistogram();
  computeTextonHistogram();
  clusterIntensities();
  computeCenter();
  computeMedianColor();
}

void Region::computeErodedMask(const cv::Mat &mask, cv::Mat &erodedMask)
{
  //TODO: move up
  const int erosionCount = 3;

  erode(mask, erodedMask, Mat(), Point(-1, -1), erosionCount);
//  erodedMask = mask.clone();
}

bool Region::isEmpty() const
{
  return (erodedArea == 0);
}

cv::Point2f Region::getCenter() const
{
  return center;
}

cv::Vec3b Region::getMedianColor() const
{
  return medianColor;
}

const cv::Mat& Region::getColorHistogram() const
{
  return colorHistogram;
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

const cv::Mat& Region::getErodedMask() const
{
  return erodedMask;
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
  if (pointCount != 0)
  {
    sum *= 1.0 / pointCount;
  }
  center = sum;
}

void Region::computeMedianColor()
{
  const int channelsCount = 3;
  vector<vector<uchar> > channels(channelsCount);
  for (int i = 0; i < erodedMask.rows; ++i)
  {
    for (int j = 0; j < erodedMask.cols; ++j)
    {
      if (erodedMask.at<uchar>(i, j) == 0)
      {
        continue;
      }

      Vec3b currentColor = image.at<Vec3b>(i, j);
      for (int k = 0; k < channelsCount; ++k)
      {
        channels[k].push_back(currentColor[k]);
      }
    }
  }

  if (channels[0].empty())
  {
    //TODO: move up
    medianColor = Vec3b(128, 128, 128);
    return;
  }

  for (int i = 0; i < channelsCount; ++i)
  {
    size_t index = channels[i].size() / 2;
    std::nth_element(channels[i].begin(), channels[i].begin() + index, channels[i].end());
    medianColor[i] = channels[i][index];
  }
}

void Region::computeColorHistogram()
{
/*
  //TODO: move up
  const int bins = 20;

  CV_Assert(image.type() == CV_8UC3);
  CV_Assert(erodedMask.type() == CV_8UC1);

  int histSize[] = {bins, bins, bins};
  float rRanges[] = {0, 256};
  float bRanges[] = {0, 256};
  float gRanges[] = {0, 256};

  const float* ranges[] = {rRanges, bRanges, gRanges};
  int channels[] = {0, 1, 2};

  if (isEmpty())
  {
    colorHistogram = Mat(3, histSize, CV_32FC1, Scalar(0));
    return;
  }

  calcHist(&image, 1, channels, erodedMask, colorHistogram, 3, histSize, ranges, true, false);
*/


  //TODO: move up
  const int hbins = 20;
  const int sbins = 20;

  if (isEmpty())
  {
    colorHistogram = Mat(hbins, sbins, CV_32FC1, Scalar(0));
    return;
  }

  CV_Assert(image.type() == CV_8UC3);
  CV_Assert(erodedMask.type() == CV_8UC1);
  Mat hsv;
  cvtColor(image, hsv, CV_BGR2HSV);

  int histSize[] = {hbins, sbins};
  float hranges[] = {0, 180};
  float sranges[] = {0, 256};
  const float* ranges[] = {hranges, sranges};
  int channels[] = {0, 1};
  calcHist(&hsv, 1, channels, erodedMask, colorHistogram, 2, histSize, ranges, true, false);

  colorHistogram /= erodedArea;

  CV_Assert(colorHistogram.type() == CV_32FC1);
}

void Region::computeTextonHistogram()
{
  //TODO: move up
  const int textonCount = 36;

  if (isEmpty())
  {
    textonHistogram = Mat(1, textonCount, CV_32FC1, Scalar(0));
    return;
  }


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
  textonHistogram /= erodedArea;
  CV_Assert(textonHistogram.type() == CV_32FC1);
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
  if (intensities.size() < clusterCount)
  {
    //TODO: move up
    intensityClusterCenters = Mat(clusterCount, 1, CV_32FC1, Scalar(128));
    return;
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

void Region::write(cv::FileStorage &fs) const
{
  fs << "colorHistogram" << colorHistogram;
  fs << "textonHistogram" << textonHistogram;
  fs << "intensityClusters" << intensityClusterCenters;
  fs << "center" << Mat(center);
  fs << "medianColor" << Mat(medianColor);
}

void Region::read(const Mat &_image, const Mat &_mask, const FileNode &fn)
{
  image = _image;
  mask = _mask;
  computeErodedMask(mask, erodedMask);

  fn["colorHistogram"] >> colorHistogram;
  fn["textonHistogram"] >> textonHistogram;
  fn["intensityClusters"] >> intensityClusterCenters;
  Mat centerMat;
  fn["center"] >> centerMat;
  center = Point2f(centerMat);

  Mat medianColorMat;
  fn["medianColor"] >> medianColorMat;
  medianColor = medianColorMat;
}

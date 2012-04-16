#include <numeric>
#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/region.hpp"

using namespace cv;
using std::cout;
using std::endl;

Region::Region(const RegionParams &_params)
{
  params = _params;
}

Region::Region(const cv::Mat &_image, const cv::Mat &_textonLabels, const cv::Mat &_mask, const RegionParams &_params)
{
  params = _params;
  image = _image;
  textonLabels = _textonLabels;
  mask = _mask;
  regionArea = countNonZero(mask);
  computeErodedMask(mask, erodedMask);
  erodedArea = countNonZero(erodedMask);

  computeIntensities();
  computeColorHistogram();
  computeTextonHistogram();
  clusterIntensities();
  computeCenter();
  computeMedianColor();
  computeRMSContrast();
  computeMichelsonContrast();
}

void Region::setDepth(const cv::Mat &invalidDepthMask)
{
  int invalidDepthArea = countNonZero(invalidDepthMask & mask);
  int allArea = countNonZero(mask);

  depthRatio = static_cast<float>(invalidDepthArea) / allArea;
}

float Region::getDepthRatio() const
{
  return depthRatio;
}

void Region::computeErodedMask(const cv::Mat &mask, cv::Mat &erodedMask)
{
  erode(mask, erodedMask, Mat(), Point(-1, -1), params.erosionCount);
//  erodedMask = mask.clone();
}

bool Region::isEmpty() const
{
  return (erodedArea == 0);
}

float Region::getRMSContrast() const
{
  return rmsContrast;
}

float Region::getMichelsonContrast() const
{
  return michelsonContrast;
}

float Region::getRobustMichelsonContrast() const
{
  return robustMichelsonContrast;
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

void Region::computeRMSContrast()
{
  if (intensities.empty())
  {
    rmsContrast = std::numeric_limits<float>::quiet_NaN();
    return;
  }

  CV_Assert(!intensities.empty());
  Mat intensitiesMat(intensities);
  Scalar mean, stdDev;
  meanStdDev(intensitiesMat, mean, stdDev);
  rmsContrast = stdDev[0];
}

void Region::computeMichelsonContrast()
{
  if (intensities.size() <= 1)
  {
    michelsonContrast = std::numeric_limits<float>::quiet_NaN();
    return;
  }

  const float eps = 1e-4;
  CV_Assert(intensities.size() > 1);
  int lastIndex = static_cast<int>(intensities.size()) - 1;
  michelsonContrast = (intensities[lastIndex] - intensities[0]) / (eps + intensities[lastIndex] + intensities[0]);
}

void Region::computeRobustMichelsonContrast()
{
  if (intensities.size() <= 1)
  {
    robustMichelsonContrast = std::numeric_limits<float>::quiet_NaN();
    return;
  }

  const float eps = 1e-4;
  CV_Assert(intensities.size() > 1);
  int firstIndex = floor(params.outliersRatio * intensities.size());
  int lastIndex = floor((1.0f - params.outliersRatio) * intensities.size());

  int firstMedianIndex = (static_cast<int>(intensities.size()) - 1) / 2;
  int secondMedianIndex = intensities.size() / 2;
  robustMichelsonContrast = (intensities[lastIndex] - intensities[firstIndex]) / (eps + intensities[firstMedianIndex] + intensities[secondMedianIndex]);
}

void Region::computeIntensities()
{
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

  intensities.clear();
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
  int params.bins = 20;

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

  if (isEmpty())
  {
    colorHistogram = Mat(params.hbins, params.sbins, CV_32FC1, Scalar(0));
    return;
  }

  CV_Assert(image.type() == CV_8UC3);
  CV_Assert(erodedMask.type() == CV_8UC1);
  Mat hsv;
  cvtColor(image, hsv, CV_BGR2HSV);

  int histSize[] = {params.hbins, params.sbins};
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
  if (isEmpty())
  {
    textonHistogram = Mat(params.textonCount, 1, CV_32FC1, Scalar(0));
    return;
  }

  Mat textonLabels_8U;
  textonLabels.convertTo(textonLabels_8U, CV_8UC1);

  int histSize[] = {params.textonCount};
  float labelRanges[] = {0, params.textonCount};
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
  if (intensities.size() < params.clusterCount)
  {
    intensityClusterCenters = Mat(params.clusterCount, 1, CV_32FC1, Scalar(128));
    return;
  }

  vector<int> boundaries;
  int currentBoundaryIndex = 0;
  int step = intensities.size() / params.clusterCount;
  int residual = intensities.size() % params.clusterCount;
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
  CV_Assert(boundaries.size() == params.clusterCount + 1);

  vector<float> clusterCenters(params.clusterCount);
  for (int i = 0; i < params.clusterCount; ++i)
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
  regionArea = countNonZero(mask);
  computeErodedMask(mask, erodedMask);
  erodedArea = countNonZero(erodedMask);

  computeIntensities();
  computeRMSContrast();
  computeMichelsonContrast();
  computeRobustMichelsonContrast();

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

bool Region::isLabeled(const cv::Mat &labelMask, float confidentLabelArea) const
{
  int labelArea = countNonZero(getMask() & labelMask);
  float areaRatio = static_cast<float>(labelArea) / regionArea;
  return (areaRatio > confidentLabelArea);
}

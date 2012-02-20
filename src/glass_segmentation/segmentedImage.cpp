#include <fstream>
#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/segmentedImage.hpp"

using namespace cv;

std::vector<cv::Mat> SegmentedImage::filterBank;

SegmentedImage::SegmentedImage(const cv::Mat &_image)
{
  image = _image;
  //TODO: move up
  const string filterBankFilename = "textureFilters.xml";

  if (filterBank.empty())
  {
    loadFilterBank(filterBankFilename, filterBank);
  }
  oversegmentImage(image, segmentation);
  segmentation2regions(image, segmentation, filterBank, regions);
}

const std::vector<Region>& SegmentedImage::getRegions() const
{
  return regions;
}

const cv::Mat& SegmentedImage::getSegmentation() const
{
  return segmentation;
}

const cv::Mat& SegmentedImage::getOriginalImage() const
{
  return image;
}

void SegmentedImage::oversegmentImage(const cv::Mat &image, cv::Mat &segmentation)
{
  //TODO: move up
//  const float sigma = 0.2f;

  const float sigma = 0.4f;
  const float k = 300.0f;
//  const float k = 500.0f;
  const int min_size = 200;

  const string sourceFilename = "imageForSegmenation.ppm";
  const string outputImageFilename = "segmentedImage.ppm";
  const string outputTxtFilename = "segmentation.txt";

  //TODO: re-implement
  imwrite(sourceFilename, image);
  std::stringstream command;
  command << "./segment " << sigma << " " << k << " " << min_size << " " << sourceFilename << " " << outputImageFilename;

  std::system(command.str().c_str());
  sleep(2);

  segmentation.create(image.size(), CV_32SC1);
  std::ifstream segmentationTxt(outputTxtFilename.c_str());
  CV_Assert(segmentationTxt.is_open());
  for (int i = 0; i < image.rows; ++i)
  {
    for (int j = 0; j < image.cols; ++j)
    {
      segmentationTxt >> segmentation.at<int>(i, j);
    }
  }
  segmentationTxt.close();
}

void SegmentedImage::mergeThinRegions(cv::Mat &segmentation, vector<int> &labels)
{
  //TODO: move up
  const int erosionIterations = 6;
  vector<bool> isThin(labels.size(), false);
  vector<Mat> masks(labels.size());
  for (size_t i = 0; i < labels.size(); ++i)
  {
    masks[i] = (segmentation == labels[i]);
    Mat erodedMask;
    erode(masks[i], erodedMask, Mat(), Point(-1, -1), erosionIterations);

    if (countNonZero(erodedMask) == 0)
    {
      isThin[i] = true;
    }
  }

  Mat finalMask(segmentation.size(), CV_8UC1, Scalar(0));
  for (size_t i = 0; i < labels.size(); ++i)
  {
    if (isThin[i])
    {
      finalMask.setTo(255, masks[i]);
    }
  }

  vector<int> newLabels;
  for (size_t i = 0; i < labels.size(); ++i)
  {
    if (!isThin[i])
    {
      newLabels.push_back(labels[i]);
      continue;
    }
    Mat dilatedMask;
    dilate(masks[i], dilatedMask, Mat());

    int largestAdjacentRegionIndex;
    int largestArea = 0;
    for (size_t j = 0; j < labels.size(); ++j)
    {
      if (i == j || countNonZero(dilatedMask & masks[j]) == 0)
      {
        continue;
      }
      int currentArea = countNonZero(masks[j]);
      if (currentArea > largestArea)
      {
        largestArea = currentArea;
        largestAdjacentRegionIndex = j;
      }
    }
    segmentation.setTo(labels[largestAdjacentRegionIndex], masks[i]);
  }
  std::swap(labels, newLabels);
}

void SegmentedImage::showSegmentation(const std::string &title) const
{
  CV_Assert(segmentation.type() == CV_32SC1);
  Vec3b *colors = new Vec3b[segmentation.total()];
  for (size_t i = 0; i < segmentation.total(); ++i)
  {
    colors[i] = Vec3b(56 + rand() % 200, 56 + rand() % 200, 56 + rand() % 200);
  }

  Mat visualization(segmentation.size(), CV_8UC3);
  for (int i = 0; i < visualization.rows; ++i)
  {
    for (int j = 0; j < visualization.cols; ++j)
    {
      visualization.at<Vec3b>(i, j) = colors[segmentation.at<int>(i, j)];
    }
  }

  imshow(title, visualization);
  waitKey();
}

void SegmentedImage::segmentation2regions(const cv::Mat &image, cv::Mat &segmentation, const std::vector<cv::Mat> &filterBank, std::vector<Region> &regions)
{
  //TODO: move up
  const int textonCount = 36;
  const int iterationCount = 10;
  const int attempts = 3;

//  const int iterationCount = 20;
//  const int attempts = 1;

  Mat responses;
  convolveImage(image, filterBank, responses);
  Mat responsesMLData = responses.reshape(1, image.total());
  CV_Assert(responsesMLData.cols == filterBank.size());
  Mat textonLabels;
  TermCriteria termCriteria = cvTermCriteria(TermCriteria::MAX_ITER, iterationCount, 0.0);
  Mat responsesMLDataFloat;
  responsesMLData.convertTo(responsesMLDataFloat, CV_32FC1);
  kmeans(responsesMLDataFloat, textonCount, textonLabels, termCriteria, attempts, KMEANS_PP_CENTERS);
  textonLabels = textonLabels.reshape(1, image.rows);
  CV_Assert(textonLabels.size() == image.size());
  CV_Assert(textonLabels.type() == CV_32SC1);

  CV_Assert(segmentation.type() == CV_32SC1);
  vector<int> labels = segmentation.reshape(1, 1);
  std::sort(labels.begin(), labels.end());
  vector<int>::iterator endIt = std::unique(labels.begin(), labels.end());
  labels.resize(endIt - labels.begin());

  int firstFreeLabel = 1 + *std::max_element(labels.begin(), labels.end());

  mergeThinRegions(segmentation, labels);
  regions.clear();
  for (size_t i = 0; i < labels.size(); ++i)
  {
    Mat mask = (segmentation == labels[i]);
    segmentation.setTo(firstFreeLabel + i, mask);
    Region currentRegion(image, textonLabels, mask);
    regions.push_back(currentRegion);
  }
  segmentation -= firstFreeLabel;
}

void convolveImage(const cv::Mat &image, const std::vector<cv::Mat> &filterBank, cv::Mat &responses)
{
  CV_Assert(image.channels() == 3);
  Mat grayscaleImage;
  cvtColor(image, grayscaleImage, CV_BGR2GRAY);
  Mat imageDouble;
  grayscaleImage.convertTo(imageDouble, CV_64FC1);
  vector<Mat> allResponses;
  for (size_t i = 0; i < filterBank.size(); ++i)
  {
    Mat currentResponses;
    filter2D(imageDouble, currentResponses, -1, filterBank[i]);
    allResponses.push_back(currentResponses);
  }
  merge(allResponses, responses);
  CV_Assert(responses.size() == image.size());
  CV_Assert(responses.channels() == filterBank.size());
}

void loadFilterBank(const std::string &filename, std::vector<cv::Mat> &filterBank)
{
  filterBank.clear();
  FileStorage fs(filename, FileStorage::READ);
  CV_Assert(fs.isOpened());
  FileNode filtersNode = fs["filters"];
  for (FileNodeIterator it = filtersNode.begin(); it != filtersNode.end(); ++it)
  {
    Mat currentFilter;
    *it >> currentFilter;
    filterBank.push_back(currentFilter);
  }
}


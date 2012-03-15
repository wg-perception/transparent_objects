#include <fstream>
#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/segmentedImage.hpp"
#include "edges_pose_refiner/utils.hpp"

using namespace cv;
using std::cout;
using std::endl;

std::vector<cv::Mat> SegmentedImage::filterBank;

SegmentedImage::SegmentedImage(const SegmentedImageParams &_params)
{
  params = _params;
}

SegmentedImage::SegmentedImage(const cv::Mat &_image, const std::string &segmentationFilename, const SegmentedImageParams &_params)
{
  params = _params;

  image = _image;

  if (filterBank.empty())
  {
    loadFilterBank(params.filterBankFilename, filterBank);
  }
  oversegmentImage(image, segmentationFilename, segmentation);
  cout << "image is oversegmented" << endl;
  segmentation2regions(image, segmentation, textonLabelsMap, filterBank, regions);
  regionAdjacencyMat.create(regions.size(), regions.size(), CV_8UC1);

#pragma omp parallel for schedule(dynamic, 4)
  for (size_t i = 0; i < regions.size(); ++i)
  {
    Mat dilatedMask;
    dilate(regions[i].getMask(), dilatedMask, Mat());
    for (size_t j = 0; j < regions.size(); ++j)
    {
      regionAdjacencyMat.at<uchar>(i, j) = (countNonZero(dilatedMask & regions[j].getMask()) == 0) ? 0 : 255;
    }
  }
  cout << "adjacency mat is computed" << endl;
}

const std::vector<Region>& SegmentedImage::getRegions() const
{
  return regions;
}

const Region& SegmentedImage::getRegion(int regionIndex) const
{
  return regions[regionIndex];
}

const cv::Mat& SegmentedImage::getSegmentation() const
{
  return segmentation;
}

const cv::Mat& SegmentedImage::getOriginalImage() const
{
  return image;
}

void SegmentedImage::oversegmentImage(const cv::Mat &image, const std::string &segmentationFilename, cv::Mat &segmentation)
{
/*
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
*/

  segmentation.create(image.size(), CV_32SC1);
  std::ifstream segmentationTxt(segmentationFilename.c_str());
  CV_Assert(segmentationTxt.is_open());
  for (int i = 0; i < image.rows; ++i)
  {
    for (int j = 0; j < image.cols; ++j)
    {
      double currentIndex;
      segmentationTxt >> currentIndex;
      segmentation.at<int>(i, j) = cvRound(currentIndex);
    }
  }
  segmentationTxt.close();
}

void SegmentedImage::mergeThinRegions(cv::Mat &segmentation, vector<int> &labels)
{
  vector<bool> isThin(labels.size(), false);
  vector<Mat> masks(labels.size());
  for (size_t i = 0; i < labels.size(); ++i)
  {
    masks[i] = (segmentation == labels[i]);
    Mat erodedMask;
    erode(masks[i], erodedMask, Mat(), Point(-1, -1), params.erosionIterations);

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

void SegmentedImage::showTextonLabelsMap(const std::string &title) const
{
  vector<Vec3b> colors(params.textonCount);
  for (int i = 0; i < params.textonCount; ++i)
  {
    const int minColor = 56;
    const int rndWidth = 200;
    colors[i] = Vec3b(minColor + rand() % rndWidth, minColor + rand() % rndWidth, minColor + rand() % rndWidth);
  }

  CV_Assert(textonLabelsMap.type() == CV_32SC1);
  Mat image(textonLabelsMap.size(), CV_8UC3);
  for (int i = 0; i < image.rows; ++i)
  {
    for (int j = 0; j < image.cols; ++j)
    {
      image.at<Vec3b>(i, j) = colors[textonLabelsMap.at<int>(i, j)];
    }
  }
  imshow(title, image);
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
  waitKey(500);
}

void SegmentedImage::showBoundaries(const std::string &title, const cv::Scalar &color) const
{
  Mat visualization = image.clone();
  for (size_t i = 0; i < regions.size(); ++i)
  {
    Mat gradientMask;
    morphologyEx(regions[i].getMask(), gradientMask, MORPH_GRADIENT, Mat());
    visualization.setTo(color, gradientMask);
  }

  imshow(title, visualization);
  waitKey(500);
}

void SegmentedImage::computeTextonLabels(const cv::Mat &image, cv::Mat &textonLabels)
{
  Mat responses;
  convolveImage(image, filterBank, responses);
  cout << "image is convolved" << endl;
  Mat responsesMLData = responses.reshape(1, image.total());
  CV_Assert(responsesMLData.cols == filterBank.size());
  TermCriteria termCriteria = cvTermCriteria(TermCriteria::MAX_ITER, params.iterationCount, 0.0);
  Mat responsesMLDataFloat;
  responsesMLData.convertTo(responsesMLDataFloat, CV_32FC1);
  kmeans(responsesMLDataFloat, params.textonCount, textonLabels, termCriteria, params.attempts, KMEANS_PP_CENTERS);
//  kmeans(responsesMLDataFloat, textonCount, textonLabels, termCriteria, attempts, KMEANS_RANDOM_CENTERS);
  textonLabels = textonLabels.reshape(1, image.rows);
  CV_Assert(textonLabels.size() == image.size());
  CV_Assert(textonLabels.type() == CV_32SC1);
  cout << "texton labels are computed" << endl;
}

void SegmentedImage::segmentation2regions(const cv::Mat &image, cv::Mat &segmentation, cv::Mat &textonLabels, const std::vector<cv::Mat> &filterBank, std::vector<Region> &regions)
{
//  textonLabels = getFromCache("textonLabels.xml");
//  if (textonLabels.empty())
  {
    computeTextonLabels(image, textonLabels);
//    saveToCache("textonLabels.xml", textonLabels);
  }

  CV_Assert(segmentation.type() == CV_32SC1);
  vector<int> labels = segmentation.reshape(1, 1).clone();
  std::sort(labels.begin(), labels.end());
  vector<int>::iterator endIt = std::unique(labels.begin(), labels.end());
  labels.resize(endIt - labels.begin());

  int firstFreeLabel = 1 + *std::max_element(labels.begin(), labels.end());
  cout << "FirstFreeLabel: " << firstFreeLabel << endl;
  cout << "size: " << labels.size() << endl;

  mergeThinRegions(segmentation, labels);
  cout << "regions are merged" << endl;
  regions.clear();
  for (size_t i = 0; i < labels.size(); ++i)
  {
    Mat mask = (segmentation == labels[i]);
    segmentation.setTo(firstFreeLabel + i, mask);
    Region currentRegion(image, textonLabels, mask);
    regions.push_back(currentRegion);
  }
  segmentation -= firstFreeLabel;
  cout << "all regions are created" << endl;
}

bool SegmentedImage::areRegionsAdjacent(int i, int j) const
{
  CV_Assert(!regionAdjacencyMat.empty());
  return regionAdjacencyMat.at<uchar>(i, j);
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
  if (!fs.isOpened())
  {
    CV_Error(CV_StsBadArg, "cannot open the file " + filename);
  }
  FileNode filtersNode = fs["filters"];
  for (FileNodeIterator it = filtersNode.begin(); it != filtersNode.end(); ++it)
  {
    Mat currentFilter;
    *it >> currentFilter;
    filterBank.push_back(currentFilter);
  }
}

void SegmentedImage::write(const std::string &filename) const
{
  FileStorage fs(filename, FileStorage::WRITE);
  CV_Assert(fs.isOpened());
  fs << "rgbImage" << image;
  fs << "segmentation" << segmentation;
  fs << "textonLabelsMap" << textonLabelsMap;
  fs << "regionAdjacencyMat" << regionAdjacencyMat;
  fs << "regions" << "[";
  for (size_t i = 0; i < regions.size(); ++i)
  {
    fs << "{";
    regions[i].write(fs);
    fs << "}";
  }
  fs << "]";
  fs.release();
}

void SegmentedImage::read(const std::string &filename)
{
  FileStorage fs(filename, FileStorage::READ);
  if (!fs.isOpened())
  {
    CV_Error(CV_StsBadArg, "Cannot open the file " + filename);
  }
  fs["rgbImage"] >> image;
  fs["segmentation"] >> segmentation;
  fs["textonLabelsMap"] >> textonLabelsMap;
  fs["regionAdjacencyMat"] >> regionAdjacencyMat;
  regions.clear();
  FileNode regionsFN = fs["regions"];
  FileNodeIterator it = regionsFN.begin(), it_end = regionsFN.end();
  int regionIndex = 0;
  for ( ; it != it_end; ++it)
  {
    Region currentRegion;
    currentRegion.read(image, segmentation == regionIndex, *it);
    regions.push_back(currentRegion);
    ++regionIndex;
  }
  fs.release();
}

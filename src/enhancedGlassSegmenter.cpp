#include <opencv2/opencv.hpp>
#include <fstream>
#include <numeric>

#include "edges_pose_refiner/utils.hpp"

using namespace cv;
using std::cout;
using std::endl;

class Region
{
  public:
    Region(const cv::Mat &image, const cv::Mat &mask);

    const cv::Mat& getMask() const;
    const cv::Mat& getColorHistogram() const;
    const cv::Mat& getIntensityClusters() const;
  private:
    void computeColorHistogram();
    void clusterIntensities();

    cv::Mat image, mask;
    cv::Mat grayscaleImage;

    cv::Mat hist;
    cv::Mat intensityClusterCenters;
};

Region::Region(const cv::Mat &_image, const cv::Mat &_mask)
{
  image = _image;
  mask = _mask;

  computeColorHistogram();
  clusterIntensities();
}

const cv::Mat& Region::getColorHistogram() const
{
  return hist;
}

const cv::Mat& Region::getIntensityClusters() const
{
  return intensityClusterCenters;
}

const cv::Mat& Region::getMask() const
{
  return mask;
}

void Region::computeColorHistogram()
{
  //TODO: move up
  const int hbins = 20;
  const int sbins = 20;

  CV_Assert(image.type() == CV_8UC3);
  CV_Assert(mask.type() == CV_8UC1);
  Mat hsv;
  cvtColor(image, hsv, CV_BGR2HSV);

  int histSize[] = {hbins, sbins};
  float hranges[] = {0, 180};
  float sranges[] = {0, 256};
  const float* ranges[] = {hranges, sranges};
  int channels[] = {0, 1};
  calcHist(&hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false);
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
  CV_Assert(mask.type() == CV_8UC1);
  CV_Assert(grayscaleImage.size() == mask.size());

  vector<int> intensities;
  for (int i = 0; i < mask.rows; ++i)
  {
    for (int j = 0; j < mask.cols; ++j)
    {
      if (mask.at<uchar>(i, j) == 255)
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

void computeColorSimilarity(const Region &region_1, const Region &region_2, float &distance)
{
  Mat hist_1 = region_1.getColorHistogram();
  Mat hist_2 = region_2.getColorHistogram();

  //TODO: experiment with different distances
  distance = norm(hist_1 - hist_2);
}

void computeOverlayConsistency(const Region &region_1, const Region &region_2, float &slope, float &intercept)
{
  //TODO: move up
  const float minAlpha = 0.0f;
  const float maxAlpha = 1.001f;


  Mat clusters_1 = region_1.getIntensityClusters();
  Mat clusters_2 = region_2.getIntensityClusters();

  Mat b = clusters_1;
  const int dim = 2;
  Mat A = Mat(b.rows, dim, b.type());
  Mat clusters_2_col = clusters_2;
  Mat col_0 = A.col(0);
  clusters_2_col.copyTo(col_0);
  A.col(1).setTo(1.0);

  Mat model;
  solve(A, b, model, DECOMP_SVD);
  CV_Assert(model.type() == CV_32FC1);
  CV_Assert(model.total() == dim);
  if (model.at<float>(0) < minAlpha || model.at<float>(0) > maxAlpha)
  {
    //TODO: test this
    std::swap(col_0, b);
    solve(A, b, model, DECOMP_SVD);
  }
  CV_Assert(model.at<float>(0) >= minAlpha && model.at<float>(0) <= maxAlpha);

  slope = model.at<float>(0);
  intercept = model.at<float>(1);
}

void oversegmentImage(const cv::Mat &image, cv::Mat &segmentation)
{
  //TODO: move up
  const float sigma = 0.5f;
  const float k = 500.0f;
  const int min_size = 50;

  const string sourceFilename = "imageForSegmenation.ppm";
  const string outputImageFilename = "segmentedImage.ppm";
  const string outputTxtFilename = "segmentation.txt";

  //TODO: re-implement
  imwrite(sourceFilename, image);
  std::stringstream command;
  command << "./segment " << sigma << " " << k << " " << min_size << " " << sourceFilename << " " << outputImageFilename;

  std::system(command.str().c_str());
  sleep(3);

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

void showSegmentation(const cv::Mat &segmentation)
{
  CV_Assert(segmentation.type() == CV_32SC1);
  Vec3b *colors = new Vec3b[segmentation.total()];
  for (size_t i = 0; i < segmentation.total(); ++i)
  {
    colors[i] = Vec3b(56 + rand() % 200, 56 + rand() % 200, 56 + rand() % 200);
  }

  Mat image(segmentation.size(), CV_8UC3);
  for (int i = 0; i < image.rows; ++i)
  {
    for (int j = 0; j < image.cols; ++j)
    {
      image.at<Vec3b>(i, j) = colors[segmentation.at<int>(i, j)];
    }
  }

  imshow("oversegmentation", image);
  waitKey();
}

void segmentation2regions(const cv::Mat &image, const cv::Mat &segmentation, vector<Region> &regions)
{
  CV_Assert(segmentation.type() == CV_32SC1);
  vector<int> labels = segmentation.reshape(1, 1);
  std::sort(labels.begin(), labels.end());
  vector<int>::iterator endIt = std::unique(labels.begin(), labels.end());
  labels.resize(endIt - labels.begin());

  regions.clear();
  for (size_t i = 0; i < labels.size(); ++i)
  {
    Mat mask = (segmentation == labels[i]);
    Region currentRegion(image, mask);
    regions.push_back(currentRegion);
  }
}

enum {DIFFERENT, GLASS_COVERED} TrainingLabels;

void train(CvSVM &svm)
{
  //TODO: move up
  const string trainingFilesList = "/media/2Tb/transparentBases/rgbGlassData/trainingImages.txt";
  const string groundTruthFilesList = "/media/2Tb/transparentBases/rgbGlassData/trainingImagesGroundTruth.txt";

  vector<string> trainingGroundTruhFiles;
  readLinesInFile(groundTruthFilesList, trainingGroundTruhFiles);

  vector<string> trainingFiles;
  readLinesInFile(trainingFilesList, trainingFiles);

  const size_t imageCount = trainingGroundTruhFiles.size();
  CV_Assert(trainingFiles.size() == imageCount);

  Mat trainingData;
  vector<int> trainingLabelsVec;
  for (size_t imageIndex = 0; imageIndex < imageCount; ++imageIndex)
  {
    Mat trainingImage = imread(trainingFiles[imageIndex]);
    CV_Assert(!trainingImage.empty());

    Mat groundTruthMask = imread(trainingGroundTruhFiles[imageIndex], CV_LOAD_IMAGE_GRAYSCALE);
    CV_Assert(!groundTruthMask.empty());
    CV_Assert(trainingImage.size() == groundTruthMask.size());

    Mat segmentation;
    oversegmentImage(trainingImage, segmentation);

    vector<Region> regions;
    segmentation2regions(trainingImage, segmentation, regions);
    vector<bool> isGlass(regions.size());
    for (size_t i = 0; i < regions.size(); ++i)
    {
      int regionArea = countNonZero(regions[i].getMask() != 0);
      int glassArea = countNonZero(regions[i].getMask() & groundTruthMask);
      const int glassFactor = 2;
      isGlass[i] = (glassFactor * glassArea > regionArea);
    }

    for (size_t i = 0; i < regions.size(); ++i)
    {
      for (size_t j = 0; j < regions.size(); ++j)
      {
        if (i == j)
        {
          continue;
        }

        float colorDistance;
        computeColorSimilarity(regions[i], regions[j], colorDistance);
        float slope, intercept;
        computeOverlayConsistency(regions[i], regions[j], slope, intercept);

        const int dim = 3;
        Mat sample = (Mat_<float>(1, dim) << colorDistance, slope, intercept);
        trainingData.push_back(sample);
        if (isGlass[i] ^ isGlass[j])
        {
          trainingLabelsVec.push_back(GLASS_COVERED);
        }
        else
        {
          trainingLabelsVec.push_back(DIFFERENT);
        }
      }
    }
  }
  Mat trainingLabels = Mat(trainingLabelsVec).reshape(1, trainingLabelsVec.size());
  CV_Assert(trainingLabels.rows == trainingData.rows);

  CvSVMParams svmParams;
  //TODO: move up
  svmParams.svm_type = CvSVM::C_SVC;
  cout << "trainingData size: " << trainingData.rows << " x " << trainingData.cols << endl;

  svm.train(trainingData, trainingLabels, Mat(), Mat(), svmParams);
}

int main()
{
//  Mat testImage = imread("tea_table2.jpg");
//  Mat oversegmentation;
//  oversegmentImage(testImage, oversegmentation);
//  showSegmentation(oversegmentation);

  CvSVM svm;
  train(svm);


  return 0;
}

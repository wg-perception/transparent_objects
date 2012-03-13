#include "edges_pose_refiner/enhancedGlassSegmenter.hpp"
#include "edges_pose_refiner/segmentedImage.hpp"
#include <omp.h>

using namespace cv;
using std::cout;
using std::endl;

void drawHistogram1D(const cv::Mat &histogram, cv::Mat &image)
{
  CV_Assert(histogram.type() == CV_32FC1);
  int dim = std::max(histogram.rows, histogram.cols);
  const int binWidth = 5;
  const int binHeight = 100;
  image.create(binHeight, binWidth * dim, CV_8UC3);
  image.setTo(Scalar::all(255));

  for (int i = 0; i < dim; ++i)
  {
    int currentHeight = cvRound(histogram.at<float>(i) * binHeight);
    rectangle(image, Point(i * binWidth, binHeight - currentHeight), Point((i+1) * binWidth, binHeight), Scalar(255, 0, 0), -1);
  }
}

void computeHistogram1D(const cv::Mat &grayscaleImage, const cv::Mat &mask, cv::Mat &histogram)
{
  const int binCount = 256;

  int histSize[] = {binCount};
  float intensityRanges[] = {0, 256};
  const float* ranges[] = {intensityRanges};
  int channels[] = {0};
  calcHist(&grayscaleImage, 1, channels, mask, histogram, 1, histSize, ranges, true, false);
  histogram /= countNonZero(mask);
}

struct InteractiveClassificationData
{
  Mat segmentation;
  Mat grayscaleImage;
  vector<Region> regions;

  bool isFirstClick;
  int firstRegionIndex;
};

void visualizeClassification(const vector<Region> &regions, const vector<float> &labels, cv::Mat &visualization)
{
  if (visualization.empty())
  {
    visualization.create(regions[0].getMask().size(), CV_8UC1);
    visualization.setTo(0);
  }

  for (size_t i = 0; i < regions.size(); ++i)
  {
    int currentLabel = cvRound(labels[i]);
    if (currentLabel == GLASS_COVERED)
    {
      visualization.setTo(255, regions[i].getMask());
    }
  }
}

void onMouse(int event, int x, int y, int, void *rawData)
{
  //TODO: move up
  const float regionEdgeLength = 1e6;

  if (event != CV_EVENT_LBUTTONDOWN)
  {
    return;
  }

  InteractiveClassificationData *data = static_cast<InteractiveClassificationData*>(rawData);

  int regionIndex = data->segmentation.at<int>(y, x);
  if (data->isFirstClick)
  {
    data->firstRegionIndex = regionIndex;
  }
  else
  {
    Region firstRegion = data->regions[data->firstRegionIndex];
    Region secondRegion = data->regions[regionIndex];
    Mat firstIntensityClusters = firstRegion.getIntensityClusters();
    Mat secondIntensityClusters = secondRegion.getIntensityClusters();
    cout << firstIntensityClusters << endl;
    cout << secondIntensityClusters << endl;
    float slope, intercept;
    computeOverlayConsistency(firstRegion, secondRegion, slope, intercept);
    Mat fullSample;
    GlassClassifier::regions2samples(firstRegion, secondRegion, fullSample);
    cout << fullSample << endl;
    cout << slope << " " << intercept << endl;
    cout << endl << endl;

    Mat firstHistogram, secondHistogram;
    computeHistogram1D(data->grayscaleImage, firstRegion.getErodedMask(), firstHistogram);
    computeHistogram1D(data->grayscaleImage, secondRegion.getErodedMask(), secondHistogram);
    Mat firstHistogramImage, secondHistogramImage;
    drawHistogram1D(firstHistogram, firstHistogramImage);
    drawHistogram1D(secondHistogram, secondHistogramImage);
    imshow("first histogram", firstHistogramImage);
    imshow("second histogram", secondHistogramImage);


    Mat closeRegionsImage(data->grayscaleImage.size(), CV_8UC1, Scalar(0));
    for (size_t i = 0; i < data->regions.size(); ++i)
    {
      Mat fullSample;
      GlassClassifier::regions2samples(firstRegion, data->regions[i], fullSample);
      //TODO: move up
      const float maxDistance = 10.0f;
      if (fullSample.at<float>(4) < maxDistance)
      {
        closeRegionsImage.setTo(255, data->regions[i].getMask());
      }
      else
      {
        if (fullSample.at<float>(4) < 30.0f)
        {
          closeRegionsImage.setTo(128, data->regions[i].getMask());
        }
      }
    }
    imshow("close regions", closeRegionsImage);
  }

  data->isFirstClick = !data->isFirstClick;

#if 0
  vector<float> labels(data->regions.size());
  for (size_t i = 0; i < data->regions.size(); ++i)
  {
    Mat ecaSample, dcaSample, fullSample;
    regions2samples(data->regions[regionIndex], data->regions[i], ecaSample, dcaSample, fullSample);
    labels[i] = cvRound(data->svm->predict(ecaSample));
//    labels[i] = cvRound(data->svm->predict(dcaSample));
  }

  Mat visualization;
  visualizeClassification(data->regions, labels, visualization);
  imshow("classification", visualization);

  /*

  vector<float> affinities;
  estimateAffinities(data->graph, data->regions.size(), data->segmentation.size(), regionIndex, affinities);
  CV_Assert(data->regions.size() == affinities.size());

  Mat regionAffinities(data->segmentation.size(), CV_32FC1, Scalar(0));
  for (size_t i = 0; i < affinities.size(); ++i)
  {
    regionAffinities.setTo(affinities[i], data->regions[i].getMask());
  }
  static Mat boundaryPresences;
  if (boundaryPresences.empty())
  {
    computeBoundaryPresences(data->regions, data->edges, boundaryPresences);
  }
  regionAffinities.setTo(0, boundaryPresences);
//  regionAffinities -= 2 * regionEdgeLength;
// regionAffinities.setTo(0, regionAffinities > regionEdgeLength);

  Mat affinityImage(regionAffinities.size(), CV_8UC1, Scalar(0));

  regionAffinities.convertTo(affinityImage, CV_8UC1, 255.0);
//  cout << regionAffinities << endl;
//  affinityImage.setTo(255, regionAffinities == 0.0);
 // regionAffinities.convertTo(affinityImage, CV_8UC1, 10.0);
  imshow("affinities", affinityImage);
  */
#endif
}

void createContour(const cv::Size &imageSize, cv::Mat &contourEdges)
{
  contourEdges.create(imageSize, CV_8UC1);
  contourEdges.setTo(0);

  Point tl(210, 120);
  Point br(420, 400);
  Rect contourRect(tl, br);
  rectangle(contourEdges, contourRect, Scalar(255), 2);
}

#define CLASSIFY

int main(int argc, char *argv[])
{
  std::system("date");
  omp_set_num_threads(4);

  if (argc != 5 && argc != 6)
  {
    cout << argv[0] << " <testFolder> <fullTestIndex> <classifierFilename> <segmentationFilename> [--visualize]" << endl;
    return -1;
  }

  const string testFolder = argv[1];
  const string fullTestIndex = argv[2];
  const string classifierFilename = argv[3];
  const string segmentationFilename = argv[4];

  bool visualize;
  if (argc == 5)
  {
    visualize = false;
  }
  else
  {
    visualize = true;
    CV_Assert(string(argv[argc - 1]) == "--visualize");
  }

#ifdef CLASSIFY
  GlassClassifier classifier;
  bool doesExist = classifier.read(classifierFilename);
  if (!doesExist)
  {
    classifier.train();
    classifier.write(classifierFilename);
  }
  cout << "classifier is ready" << endl;
#endif

  Mat testGlassMask = imread(testFolder + "/glassMask_" + fullTestIndex + ".png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat testImage = imread(testFolder + "/image_" + fullTestIndex + ".png");
  CV_Assert(!testImage.empty());
  CV_Assert(!testGlassMask.empty());

  SegmentedImage segmentedImage;
  segmentedImage.read(testFolder + "/segmentedImage_" + fullTestIndex + ".xml");

  if (visualize)
  {
    segmentedImage.showSegmentation("segmentation");
    segmentedImage.showBoundaries("boundaries", Scalar(255, 0, 255));
    segmentedImage.showTextonLabelsMap("textons");

    vector<Region> regions = segmentedImage.getRegions();
    Mat maskImage = Mat(testImage.size(), CV_8UC1, Scalar(0));
    for (size_t i = 0; i < regions.size(); ++i)
    {
      Mat mask = regions[i].getMask();
      const int erosionCount = 2;
      Mat erodedMask;
      erode(mask, erodedMask, Mat(), Point(-1, -1), erosionCount);

      testImage.copyTo(maskImage, erodedMask);
    }
    imshow("maskImage", maskImage);
    waitKey(500);
  }

#ifdef CLASSIFY
  Mat boundaryStrength;
  classifier.test(segmentedImage, testGlassMask, boundaryStrength);

  Mat finalSegmentation;
  geodesicActiveContour(segmentedImage.getOriginalImage(), boundaryStrength, finalSegmentation);
//  Mat gac = drawSegmentation(segmentedImage.getOriginalImage(), finalSegmentation, 2);
  imwrite(segmentationFilename, finalSegmentation);
#endif

  if (visualize)
  {
    imshow("finalSegmentation", finalSegmentation);
    waitKey();

    const string testSegmentationTitle = "test segmentation";
    namedWindow(testSegmentationTitle);
    InteractiveClassificationData data;
    data.segmentation = segmentedImage.getSegmentation();
    data.regions = segmentedImage.getRegions();
    cvtColor(testImage, data.grayscaleImage, CV_BGR2GRAY);
    data.isFirstClick = true;

    setMouseCallback(testSegmentationTitle, onMouse, &data);
  //  segmentedImage.showSegmentation(testSegmentationTitle);

    segmentedImage.showBoundaries(testSegmentationTitle);
    waitKey();
  }

  std::system("date");

  return 0;
}

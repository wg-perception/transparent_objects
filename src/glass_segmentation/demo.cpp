#include "edges_pose_refiner/enhancedGlassSegmenter.hpp"
#include "edges_pose_refiner/segmentedImage.hpp"
#include "edges_pose_refiner/glassDetector.hpp"
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

bool isArgumentSet(int argc, char *argv[], const std::string &argument)
{
  for (int i = 0; i < argc; ++i)
  {
    string currentArgument = argv[i];
    if (currentArgument == argument)
    {
      return true;
    }
  }

  return false;
}


int main(int argc, char *argv[])
{
  std::system("date");
  omp_set_num_threads(5);

  if (argc != 7 && argc != 8)
  {
    cout << argv[0] << " <testFolder> <fullTestIndex> <classifierFilename> <segmentationFilesList> <propagationScalingsList> <algorithmName> [--visualize]" << endl;
    return -1;
  }

  const string trainingFilesList = "/media/2Tb/transparentBases/fixedOnTable/base/trainingImages.txt";
  const string depthFilesList = "/media/2Tb/transparentBases/fixedOnTable/base/trainingDepths.txt";
  const string groundTruthFilesList = "/media/2Tb/transparentBases/fixedOnTable/base/trainingImagesGroundTruth.txt";

  const string testFolder = argv[1];
  const string fullTestIndex = argv[2];
  const string classifierFilename = argv[3];
  const string segmentationFilesList = argv[4];
  const string propagationScalingsList = argv[5];
  const string algorithmName = argv[6];
  vector<string> segmentationFilenames;
  readLinesInFile(segmentationFilesList, segmentationFilenames);
  CV_Assert(!segmentationFilenames.empty());
  vector<string> propagationScalingsStrings;
  readLinesInFile(propagationScalingsList, propagationScalingsStrings);
  CV_Assert(propagationScalingsStrings.size() == segmentationFilenames.size());
  vector<float> propagationScalings(propagationScalingsStrings.size());
  for (size_t i = 0; i < propagationScalingsStrings.size(); ++i)
  {
    propagationScalings[i] = atof(propagationScalingsStrings[i].c_str());
  }

  bool useGAC = isArgumentSet(argc, argv, "--use_GAC");
  bool visualize = isArgumentSet(argc, argv, "--visualize");

  string testImageFilename = testFolder + "/image_" + fullTestIndex + ".png";
  Mat testImage = imread(testImageFilename);
  if (testImage.empty())
  {
    CV_Error(CV_StsBadArg, "Cannot read " + testImageFilename);
  }

  string testGlassMaskFilename = testFolder + "/glassMask_" + fullTestIndex + ".png";
  Mat testGlassMask = imread(testGlassMaskFilename, CV_LOAD_IMAGE_GRAYSCALE);
  if (testGlassMask.empty())
  {
    CV_Error(CV_StsBadArg, "Cannot read" + testGlassMaskFilename);
  }

  //TODO: move up
  const std::string registrationMaskFilename = "/media/2Tb/transparentBases/fixedOnTable/base/registrationMask.png";
  Mat registrationMask = imread(registrationMaskFilename, CV_LOAD_IMAGE_GRAYSCALE);
  CV_Assert(!registrationMask.empty());

  //TODO: use TODBaseImporter
  string depthMatFilename = testFolder + "/depth_image_" + fullTestIndex + ".xml.gz";
  FileStorage depthFs(depthMatFilename, FileStorage::READ);
  Mat testDepthMat;
  depthFs["depth_image"] >> testDepthMat;
  depthFs.release();
  CV_Assert(!testDepthMat.empty());

  if (algorithmName == "Depth")
  {
    for (size_t i = 0; i < segmentationFilenames.size(); ++i)
    {
      GlassSegmentationParams params;
      params.finalClosingIterations = propagationScalings[i];
      GlassSegmentator glassSegmentator(params);
      int numberOfComponents;
      Mat currentGlassMask;
      glassSegmentator.segment(testImage, testDepthMat, registrationMask, numberOfComponents, currentGlassMask);
      imwrite(segmentationFilenames[i], currentGlassMask);
    }

    return 0;
  }

#ifdef CLASSIFY
  GlassClassifier classifier;
  bool doesExist = classifier.read(classifierFilename);
  if (!doesExist)
  {
    classifier.train(trainingFilesList, groundTruthFilesList, depthFilesList, registrationMask);
    classifier.write(classifierFilename);
  }
  cout << "classifier is ready" << endl;
#endif

  SegmentedImage segmentedImage;
  segmentedImage.read(testFolder + "/segmentedImage_" + fullTestIndex + ".xml");

  Mat invalidDepthMask = getInvalidDepthMask(testDepthMat, registrationMask);
  segmentedImage.setDepth(invalidDepthMask);

  if (visualize)
  {
    imshow("invalid depth mask", invalidDepthMask);
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


  if (algorithmName == "Voting")
  {
    for (size_t i = 0; i < segmentationFilenames.size(); ++i)
    {
      Mat currentMask = boundaryStrength > propagationScalings[i];
      imwrite(segmentationFilenames[i], currentMask);
    }
  }

  if (algorithmName == "GAC")
  {
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < segmentationFilenames.size(); ++i)
    {
      Mat finalSegmentation;
      GeodesicActiveContourParams params;
      params.propagationScaling = propagationScalings[i];
      geodesicActiveContour(segmentedImage.getOriginalImage(), boundaryStrength, finalSegmentation, params);
    //  Mat gac = drawSegmentation(segmentedImage.getOriginalImage(), finalSegmentation, 2);
      CV_Assert(!finalSegmentation.empty());
      imwrite(segmentationFilenames[i], finalSegmentation);

      if (visualize)
      {
        std::stringstream indexStr;
        indexStr << i;
        imshow("finalSegmentation_" + indexStr.str(), finalSegmentation);
        Mat drawedSegmentation = drawSegmentation(testImage, finalSegmentation);
        imshow("drawedSegmentation_" + indexStr.str(), drawedSegmentation);

        const string testSegmentationTitle = "test segmentation " + indexStr.str();
        namedWindow(testSegmentationTitle);
        InteractiveClassificationData data;
        data.segmentation = segmentedImage.getSegmentation();
        data.regions = segmentedImage.getRegions();
        cvtColor(testImage, data.grayscaleImage, CV_BGR2GRAY);
        data.isFirstClick = true;

        if (i == 0)
        {
          setMouseCallback(testSegmentationTitle, onMouse, &data);
          segmentedImage.showBoundaries(testSegmentationTitle);
        }
      }
    }
    if (visualize)
    {
      waitKey();
    }
  }
#endif


  std::system("date");

  return 0;
}

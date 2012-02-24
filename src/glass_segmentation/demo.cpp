#include "edges_pose_refiner/enhancedGlassSegmenter.hpp"
#include "edges_pose_refiner/segmentedImage.hpp"

using namespace cv;
using std::cout;
using std::endl;

struct InteractiveClassificationData
{
  CvSVM *svm;
  Mat edges, segmentation;
  vector<Region> regions;

  Graph graph;
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

#if 0
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
}
#endif

void createContour(const cv::Size &imageSize, cv::Mat &contourEdges)
{
  contourEdges.create(imageSize, CV_8UC1);
  contourEdges.setTo(0);

  Point tl(210, 120);
  Point br(420, 400);
  Rect contourRect(tl, br);
  rectangle(contourEdges, contourRect, Scalar(255), 2);
}
int main(int argc, char *argv[])
{
  std::system("date");

  GlassClassifier classifier;
  classifier.train();

  const string testSegmentationTitle = "test segmentation";

//  Mat testImage = imread("/media/2Tb/transparentBases/rgbGlassData/Test/plate_building.jpg");
//  Mat testImage = imread("/media/2Tb/transparentBases/rgbGlassData/Test/tea_table2.jpg");
//  Mat testImage = imread("/media/2Tb/transparentBases/rgbGlassData/Test/table_misc3.jpg");


//  Mat testGlassMask = imread("/media/2Tb/transparentBases/fixedOnTable/base/glass/mask_image_00050_croped.png", CV_LOAD_IMAGE_GRAYSCALE);
//  Mat testImage = imread("/media/2Tb/transparentBases/fixedOnTable/base/glass/image_00050_croped.png");

//  Mat testGlassMask = imread("/media/2Tb/transparentBases/segmentation/rgb/mask_17.png", CV_LOAD_IMAGE_GRAYSCALE);

  Mat testGlassMask = imread("/media/2Tb/transparentBases/segmentation/rgb/mask_00017.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat testImage = imread("/media/2Tb/transparentBases/segmentation/rgb/image_00017.png");
//  imshow("testImage", testImage);
//  waitKey(500);

//  Mat testImage = imread("/media/2Tb/transparentBases/rgbGlassData/Training/teaB1f.jpg");
//  Mat testImage = imread("/media/2Tb/transparentBases/rgbGlassData/Training/teaB2f.jpg");
//  Mat testImage = imread("/media/2Tb/transparentBases/rgbGlassData/Training/plateB1fc.jpg");
//  Mat testImage = imread("/media/2Tb/transparentBases/rgbGlassData/Training/wineB1f.jpg");

//  SegmentedImage segmentedImage(testImage);
//  segmentedImage.write("image_00017.xml");

  SegmentedImage segmentedImage;
//  segmentedImage.read("segmentedImage.xml");
  segmentedImage.read("image_00017.xml");
  segmentedImage.showSegmentation("test segmentation");

  Mat boundaryStrength;
  classifier.test(segmentedImage, testGlassMask, boundaryStrength);

  Mat finalSegmentation;

//  Mat contour;
//  createContour(testImage.size(), contour);
//  geodesicActiveContour(contour, finalSegmentation);
  geodesicActiveContour(segmentedImage.getOriginalImage(), boundaryStrength, finalSegmentation);
  imshow("finalSegmentation", finalSegmentation);
  Mat gac = drawSegmentation(segmentedImage.getOriginalImage(), finalSegmentation, 2);
  imwrite("final.png", gac);
  waitKey();

/*

  namedWindow(testSegmentationTitle);
  InteractiveClassificationData data;
  data.svm = &svm;
  data.edges = edges;
  data.segmentation = segmentedImage.getSegmentation();
  data.regions = regions;
  data.graph = graph;

  setMouseCallback(testSegmentationTitle, onMouse, &data);
  segmentedImage.showSegmentation(testSegmentationTitle);
*/
  std::system("date");

  return 0;
}

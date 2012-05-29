#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/detector.hpp"
#include "edges_pose_refiner/TODBaseImporter.hpp"

using namespace cv;
using namespace transpod;
using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
  std::system("date");

  srand(42);
  RNG &rng = theRNG();
  rng.state = 0xffffffff;

  CV_Assert(argc == 3);
  string baseFolder = argv[1];
  string testObjectName = argv[2];

  const string modelsPath = "/media/2Tb/transparentBases/trainedModels/";
  const string testFolder = baseFolder + "/" + testObjectName + "/";
  const string kinectCameraFilename = baseFolder + "/center.yml";
  const string registrationMaskFilename = baseFolder + "/registrationMask.png";
  const string imageFilename = baseFolder + "/image.png";
  const string depthFilename = baseFolder + "/depth.xml.gz";
  const string pointCloudFilename = baseFolder + "/pointCloud.pcd";

  const vector<string> objectNames = {testObjectName};
//  const vector<string> objectNames = {"bank", "bottle", "glass", "sourCream", "wineglass"};
//  const vector<string> objectNames = {"bank", "bottle", "sourCream", "wineglass"};
//  const vector<string> objectNames = {"bank", "bottle", "wineglass"};
//  const vector<string> objectNames = {"bank", "wineglass"};
//  const vector<string> objectNames = {"wineglass"};
//, "bottle", "glass", "sourCream", "wineglass"};

  DetectorParams params;
//  params.glassSegmentationParams.closingIterations = 12;
//  params.glassSegmentationParams.openingIterations = 8;
//  params.glassSegmentationParams.finalClosingIterations = 8;
 // params.glassSegmentationParams.finalClosingIterations = 12;

  //good clutter
  params.glassSegmentationParams.openingIterations = 15;
  params.glassSegmentationParams.closingIterations = 12;
  params.glassSegmentationParams.finalClosingIterations = 32;
  params.glassSegmentationParams.grabCutErosionsIterations = 4;

  TODBaseImporter dataImporter(testFolder);

  PinholeCamera kinectCamera;
  dataImporter.readCameraParams(kinectCameraFilename, kinectCamera, false);
  CV_Assert(kinectCamera.imageSize == Size(640, 480));

  vector<EdgeModel> edgeModels(objectNames.size());
  for (size_t i = 0; i < objectNames.size(); ++i)
  {
    dataImporter.importEdgeModel(modelsPath, objectNames[i], edgeModels[i]);
    cout << "All points in the model: " << edgeModels[i].points.size() << endl;
    cout << "Surface points in the model: " << edgeModels[i].stableEdgels.size() << endl;
  }

  Detector detector(kinectCamera, params);
  for (size_t i = 0; i < edgeModels.size(); ++i)
  {
    detector.addTrainObject(objectNames[i], edgeModels[i]);
  }

  Mat registrationMask = imread(registrationMaskFilename, CV_LOAD_IMAGE_GRAYSCALE);
  CV_Assert(!registrationMask.empty());

  Mat kinectDepth, kinectBgrImage;
  dataImporter.importBGRImage(imageFilename, kinectBgrImage);
  dataImporter.importDepth(depthFilename, kinectDepth);

  pcl::PointCloud<pcl::PointXYZ> testPointCloud;
  dataImporter.importPointCloud(pointCloudFilename, testPointCloud);

  vector<PoseRT> poses_cam;
  vector<float> posesQualities;
  vector<string> detectedObjectsNames;

  TickMeter recognitionTime;
  recognitionTime.start();
  Detector::DebugInfo debugInfo;
  try
  {
    detector.detect(kinectBgrImage, kinectDepth, registrationMask, testPointCloud, poses_cam, posesQualities, detectedObjectsNames, &debugInfo);
  }
  catch(const cv::Exception &)
  {
  }
  recognitionTime.stop();
  cout << "Recognition time: " << recognitionTime.getTimeSec() << "s" << endl;

  Mat glassMask = debugInfo.glassMask;
  imshow("glassMask", glassMask);
  showSegmentation(kinectBgrImage, glassMask, "segmentation");

/*
  Mat detectionResults = kinectBgrImage.clone();
  detector.visualize(poses_cam, detectedObjectsNames, detectionResults);
  imshow("detection", detectionResults);
  waitKey();
*/
  cout << "number of detected poses: " << poses_cam.size() << endl;
  cout << "number of qualities: " << posesQualities.size() << endl;

  if (!posesQualities.empty())
  {
    std::vector<float>::iterator bestDetection = std::min_element(posesQualities.begin(), posesQualities.end());
    int bestDetectionIndex = std::distance(posesQualities.begin(), bestDetection);
    int detectedObjectIndex = detector.getTrainObjectIndex(detectedObjectsNames[bestDetectionIndex]);
    cout << "Recognized object: " << detectedObjectsNames[bestDetectionIndex] << endl;

    Mat detectionResults = kinectBgrImage.clone();
    vector<PoseRT> bestPose(1, poses_cam[bestDetectionIndex]);
    vector<string> bestName(1, detectedObjectsNames[bestDetectionIndex]);
    detector.visualize(bestPose, bestName, detectionResults);
    imshow("detection", detectionResults);
    imwrite("detection_" + bestName[0] + ".png", detectionResults);
    imwrite("testImage_" + bestName[0] + ".png", kinectBgrImage);
    waitKey();
  }


  std::system("date");
  return 0;
}

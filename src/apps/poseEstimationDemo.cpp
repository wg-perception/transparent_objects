#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/detector.hpp"
#include "edges_pose_refiner/TODBaseImporter.hpp"

//#define USE_INITIAL_GUESS

#ifdef USE_INITIAL_GUESS
#include "edges_pose_refiner/pclProcessing.hpp"
#endif

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

  if (argc != 3)
  {
    cout << argv[0] << " <baseFolder> <testObjectName>" << endl;
    return -1;
  }
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
//  params.glassSegmentationParams.finalClosingIterations = 12;

  //good clutter
  params.glassSegmentationParams.openingIterations = 15;
  params.glassSegmentationParams.closingIterations = 12;
  params.glassSegmentationParams.finalClosingIterations = 32;
  params.glassSegmentationParams.grabCutErosionsIterations = 4;
  params.planeSegmentationMethod = FIDUCIALS;

  TODBaseImporter dataImporter(testFolder);

  Mat kinectDepth, kinectBgrImage;
  dataImporter.importBGRImage(imageFilename, kinectBgrImage);
  dataImporter.importDepth(depthFilename, kinectDepth);
  imshow("rgb image", kinectBgrImage);
  imshow("depth", kinectDepth);
  waitKey(500);

  PinholeCamera kinectCamera;
  dataImporter.readCameraParams(kinectCameraFilename, kinectCamera, false);
  CV_Assert(kinectCamera.imageSize == Size(640, 480));

  vector<EdgeModel> edgeModels(objectNames.size());
  for (size_t i = 0; i < objectNames.size(); ++i)
  {
    dataImporter.importEdgeModel(modelsPath, objectNames[i], edgeModels[i]);
    cout << "All points in the model: " << edgeModels[i].points.size() << endl;
    cout << "Surface points in the model: " << edgeModels[i].stableEdgels.size() << endl;
    EdgeModel::computeSurfaceEdgelsOrientations(edgeModels[i]);
  }

  Detector detector(kinectCamera, params);
  for (size_t i = 0; i < edgeModels.size(); ++i)
  {
    detector.addTrainObject(objectNames[i], edgeModels[i]);
  }

  Mat registrationMask = imread(registrationMaskFilename, CV_LOAD_IMAGE_GRAYSCALE);
  CV_Assert(!registrationMask.empty());


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

#ifdef USE_INITIAL_GUESS
  {
    PoseRT initialPose;
    //3
    initialPose.rvec = (Mat_<double>(3, 1) << -0.8356714356174999, 0.08672943393358865, 0.1875608929524414);
    initialPose.tvec = (Mat_<double>(3, 1) << -0.0308572565967134, 0.1872369696442459, 0.8105566363422957);

    poses_cam.push_back(initialPose);
    //TODO: move up
    posesQualities.push_back(1.0f);

    GlassSegmentator glassSegmentator(params.glassSegmentationParams);
    Mat glassMask;
    int numberOfComponents;
    glassSegmentator.segment(kinectBgrImage, kinectDepth, registrationMask, numberOfComponents, glassMask);
    showSegmentation(kinectBgrImage, glassMask);

    transpod::PoseEstimator poseEstimator(kinectCamera);
    poseEstimator.setModel(edgeModels[0]);

    Vec4f tablePlane;
    computeTableOrientationByFiducials(kinectCamera, kinectBgrImage, tablePlane);
    poseEstimator.refinePosesBySupportPlane(kinectBgrImage, glassMask, tablePlane, poses_cam, posesQualities);

    Mat finalVisualization = kinectBgrImage.clone();
    poseEstimator.visualize(poses_cam[0], finalVisualization);

    imshow("estimated poses", finalVisualization);
    waitKey();
  }
#endif

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
    cout << "Recognized object: " << detectedObjectsNames[bestDetectionIndex] << endl;

    Mat detectionResults = kinectBgrImage.clone();
//    vector<PoseRT> bestPose(1, poses_cam[bestDetectionIndex]);
    vector<string> bestName(1, detectedObjectsNames[bestDetectionIndex]);
//    detector.visualize(bestPose, bestName, detectionResults);
    detector.visualize(poses_cam, detectedObjectsNames, detectionResults);
    imshow("detection", detectionResults);
    imwrite("detection_" + bestName[0] + ".png", detectionResults);
    imwrite("testImage_" + bestName[0] + ".png", kinectBgrImage);
    waitKey();
  }


  std::system("date");
  return 0;
}

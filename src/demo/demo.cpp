#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/detector.hpp"
#include "edges_pose_refiner/TODBaseImporter.hpp"

using namespace cv;
using namespace transpod;

void readData(const string &pathToDemoData, PinholeCamera &camera, Mat &objectPointCloud,
              Mat &registrationMask, Mat &image, Mat &depth, Mat &testPointCloud);

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << argv[0] << " <path_to_demo_data>" << std::endl;
    return -1;
  }

  string pathToDemoData = argv[1];
  const string objectName = "middle_cup";

  // 1. Get the data
  PinholeCamera camera;
  Mat objectPointCloud, registrationMask, image, depth, testPointCloud;
  readData(pathToDemoData, camera, objectPointCloud, registrationMask, image, depth, testPointCloud);

  // 2. Initialize the detector
  Detector detector(camera);
  detector.addTrainObject(objectName, objectPointCloud);

  // 3. Detect transparent objects
  vector<PoseRT> poses;
  vector<float> errors;
  vector<string> detectedObjectsNames;
  Detector::DebugInfo debugInfo;
  detector.detect(image, depth, registrationMask, testPointCloud,
                  poses, errors, detectedObjectsNames, &debugInfo);

  // 4. Visualize results
  showSegmentation(debugInfo.glassMask, image);
  detector.showResults(poses, detectedObjectsNames, image);
  waitKey();

  return 0;
}

void readData(const string &pathToDemoData, PinholeCamera &camera, Mat &objectPointCloud,
              Mat &registrationMask, Mat &image, Mat &depth, Mat &testPointCloud)
{
  const string objectPointCloudFilename = pathToDemoData + "/trainObject.xml.gz";
  const string cameraFilename = pathToDemoData + "/camera.yml";
  const string registrationMaskFilename = pathToDemoData + "/registrationMask.png";
  const string imageFilename = pathToDemoData + "/image.png";
  const string depthFilename = pathToDemoData + "/depth.xml.gz";
  const string testPointCloudFilename = pathToDemoData + "/testPointCloud.xml.gz";

  TODBaseImporter dataImporter;
  dataImporter.importCamera(cameraFilename, camera);
  dataImporter.importPointCloud(objectPointCloudFilename, objectPointCloud);
  dataImporter.importRegistrationMask(registrationMaskFilename, registrationMask);
  dataImporter.importBGRImage(imageFilename, image);
  dataImporter.importDepth(depthFilename, depth);
  dataImporter.importPointCloud(testPointCloudFilename, testPointCloud);
}

#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/detector.hpp"
#include "edges_pose_refiner/TODBaseImporter.hpp"

using namespace cv;
using namespace transpod;

void readData(const string &pathToDemoData, PinholeCamera &camera,
              Mat &objectPointCloud_1, Mat &objectNormals_1, Mat &objectPointCloud_2, Mat &objectNormals_2,
              Mat &registrationMask, Mat &image, Mat &depth);

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << argv[0] << " <path_to_sample_data>" << std::endl;
    return -1;
  }

  string pathToDemoData = argv[1];
  const string objectName_1 = "glass";
  const string objectName_2 = "middle_cup";

  // 1. Get the data
  std::cout << "Reading data...  " << std::flush;
  PinholeCamera camera;
  Mat objectPointCloud_1, objectNormals_1, objectPointCloud_2, objectNormals_2, registrationMask, image, depth;
  readData(pathToDemoData, camera, objectPointCloud_1, objectNormals_1, objectPointCloud_2, objectNormals_2, registrationMask, image, depth);
  std::cout << "done." << std::endl;

  // 2. Initialize the detector
  std::cout << "Training...  " << std::flush;
  //    A. set morphology parameters of glass segmentation
  DetectorParams params;
  params.glassSegmentationParams.closingIterations = 6;
  params.glassSegmentationParams.openingIterations = 10;
  //    B. add train objects into the detector
  Detector detector(camera, params);
  detector.addTrainObject(objectName_1, objectPointCloud_1, objectNormals_1);
  detector.addTrainObject(objectName_2, objectPointCloud_2, objectNormals_2);
  std::cout << "done." << std::endl;

  // 3. Detect transparent objects
  std::cout << "Detecting...  " << std::flush;
  vector<PoseRT> poses;
  vector<float> errors;
  vector<string> detectedObjectsNames;
  Detector::DebugInfo debugInfo;
  detector.detect(image, depth, registrationMask,
                  poses, errors, detectedObjectsNames, &debugInfo);
  std::cout << "done." << std::endl;

  // 4. Visualize results
  imshow("input rgb image", image);
  imshow("input depth image", depth);
  showSegmentation(image, debugInfo.glassMask);
  detector.showResults(poses, detectedObjectsNames, image);
  waitKey();

  return 0;
}

void readData(const string &pathToDemoData, PinholeCamera &camera,
              Mat &objectPointCloud_1, Mat &objectNormals_1, Mat &objectPointCloud_2, Mat &objectNormals_2,
              Mat &registrationMask, Mat &image, Mat &depth)
{
  const string objectPointCloudFilename_1 = pathToDemoData + "/trainObject_1.ply";
  const string objectPointCloudFilename_2 = pathToDemoData + "/trainObject_2.ply";
  const string cameraFilename = pathToDemoData + "/camera.yml";
  const string registrationMaskFilename = pathToDemoData + "/registrationMask.png";
  const string imageFilename = pathToDemoData + "/image.png";
  const string depthFilename = pathToDemoData + "/depth.xml.gz";

  TODBaseImporter dataImporter;
  dataImporter.importCamera(cameraFilename, camera);
  dataImporter.importPointCloud(objectPointCloudFilename_1, objectPointCloud_1, objectNormals_1);
  dataImporter.importPointCloud(objectPointCloudFilename_2, objectPointCloud_2, objectNormals_2);
  dataImporter.importRegistrationMask(registrationMaskFilename, registrationMask);
  dataImporter.importBGRImage(imageFilename, image);
  dataImporter.importDepth(depthFilename, depth);
}

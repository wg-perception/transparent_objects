#include <opencv2/opencv.hpp>

#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/glassSegmentator.hpp"
#include "modelCapture/modelCapturer.hpp"

#include <omp.h>

using namespace cv;
using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
  omp_set_num_threads(7);
//  omp_set_num_threads(1);

  std::system("date");

  if (argc != 4)
  {
    cout << argv[0] << " <baseFoldler> <modelsPath> <objectName>" << endl;
    return -1;
  }

  const string baseFolder = argv[1];
  const string modelsPath = argv[2];
  const string objectName = argv[3];
  const string testFolder = baseFolder + "/" + objectName + "/";

  vector<string> trainObjectNames;
  trainObjectNames.push_back(objectName);

  PinholeCamera kinectCamera;
  vector<int> testIndices;
  Mat registrationMask;
  vector<EdgeModel> edgeModels;
  TODBaseImporter dataImporter(baseFolder, testFolder);
  dataImporter.importAllData(&modelsPath, &trainObjectNames, &kinectCamera, &registrationMask, &edgeModels, &testIndices);

  GlassSegmentatorParams glassSegmentationParams;
  glassSegmentationParams.openingIterations = 15;
  glassSegmentationParams.closingIterations = 12;
//  glassSegmentationParams.finalClosingIterations = 22;
  glassSegmentationParams.finalClosingIterations = 25;
//  glassSegmentationParams.grabCutErosionsIterations = 4;
  glassSegmentationParams.grabCutErosionsIterations = 3;
  GlassSegmentator glassSegmentator(glassSegmentationParams);

  ModelCapturer modelCapturer(kinectCamera);
  vector<ModelCapturer::Observation> observations(testIndices.size());

#pragma omp parallel for
  for(size_t testIdx = 0; testIdx < testIndices.size(); testIdx++)
  {
    int testImageIdx = testIndices[ testIdx ];
    cout << "Test: " << testIdx << " " << testImageIdx << endl;

    Mat bgrImage, depthImage;
    dataImporter.importBGRImage(testImageIdx, bgrImage);
    dataImporter.importDepth(testImageIdx, depthImage);

//    imshow("bgr", bgrImage);
//    imshow("depth", depthImage);

    PoseRT fiducialPose;
    dataImporter.importGroundTruth(testImageIdx, fiducialPose, false);

    int numberOfComponens;
    Mat glassMask;
    glassSegmentator.segment(bgrImage, depthImage, registrationMask, numberOfComponens, glassMask);

//    showSegmentation(bgrImage, glassMask);
//    waitKey(200);
    observations[testIdx].bgrImage = bgrImage;
    observations[testIdx].mask = glassMask;
    observations[testIdx].pose = fiducialPose;
  }

  modelCapturer.setObservations(observations);


  vector<Point3f> modelPoints;
  modelCapturer.createModel(modelPoints);
  writePointCloud("model.asc", modelPoints);
  EdgeModel createdEdgeModel(modelPoints, true, true);

  vector<vector<Point3f> > allModels;
  allModels.push_back(createdEdgeModel.points);
  allModels.push_back(edgeModels[0].points);
  publishPoints(allModels);
  return 0;
}

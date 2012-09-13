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
  omp_set_num_threads(5);

  std::system("date");

  if (argc != 3)
  {
    cout << argv[0] << " <baseFoldler> <objectName>" << endl;
    return -1;
  }

  const string baseFolder = argv[1];
  const string objectName = argv[2];
  const string testFolder = baseFolder + "/" + objectName + "/";

  PinholeCamera kinectCamera;
  vector<int> testIndices;
  Mat registrationMask;
  TODBaseImporter dataImporter(baseFolder, testFolder);
  dataImporter.importAllData(0, 0, &kinectCamera, &registrationMask, 0, &testIndices);

  GlassSegmentatorParams glassSegmentationParams;
  glassSegmentationParams.openingIterations = 15;
  glassSegmentationParams.closingIterations = 12;
  glassSegmentationParams.finalClosingIterations = 22;
  glassSegmentationParams.grabCutErosionsIterations = 4;
  GlassSegmentator glassSegmentator(glassSegmentationParams);

  ModelCapturer modelCapturer(kinectCamera);
  vector<Mat> bgrImages(testIndices.size());
  vector<Mat> glassMasks(testIndices.size());
  vector<PoseRT> fiducialPoses(testIndices.size());

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

    dataImporter.importGroundTruth(testImageIdx, fiducialPoses[testIdx], false);

    int numberOfComponens;
    glassSegmentator.segment(bgrImage, depthImage, registrationMask, numberOfComponens, glassMasks[testIdx]);

//    showSegmentation(bgrImage, glassMask);
//    waitKey();
  }

  CV_Assert(glassMasks.size() == fiducialPoses.size());
  for (size_t i = 0; i < glassMasks.size(); ++i)
  {
    modelCapturer.addObservation(glassMasks[i], fiducialPoses[i]);
  }

  modelCapturer.createModel();

  return 0;
}

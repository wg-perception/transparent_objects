#include <opencv2/opencv.hpp>

#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/glassSegmentator.hpp"
#include "modelCapture/modelCapturer.hpp"

#include <omp.h>

using namespace cv;
using std::cout;
using std::endl;


float computePartialDirectionalHausdorffDistance(const std::vector<cv::Point3f> &baseCloud, const std::vector<cv::Point3f> &testCloud,
                                                 float percentile, int neighbourIndex)
{
  flann::LinearIndexParams flannIndexParams;
  flann::Index flannIndex(Mat(baseCloud).reshape(1), flannIndexParams);

  vector<float> knnDists(testCloud.size());
  for (size_t i = 0; i < testCloud.size(); ++i)
  {
    Mat query;
    point2row(testCloud[i], query);
    Mat indices, dists;
    const int knn = neighbourIndex;
    flannIndex.knnSearch(query, indices, dists, knn);
    CV_Assert(dists.type() == CV_32FC1);
    knnDists[i] = sqrt(dists.at<float>(knn - 1));
  }

  int percentileIndex = floor(percentile * (static_cast<int>(testCloud.size()) - 1));
  CV_Assert(percentileIndex >= 0 && percentileIndex < testCloud.size());
  std::nth_element(knnDists.begin(), knnDists.begin() + percentileIndex, knnDists.end());
  return knnDists[percentileIndex];
}

Vec3f computeModelDimensions(const std::vector<cv::Point3f> &modelPoints)
{
  Mat modelPointsMat = Mat(modelPoints).reshape(1);
  CV_Assert(modelPointsMat.cols == 3);
  Vec3f dimensions;
  for (int axisIndex = 0; axisIndex < modelPointsMat.cols; ++axisIndex)
  {
    double minVal, maxVal;
    minMaxLoc(modelPointsMat.col(axisIndex), &minVal, &maxVal);
    dimensions[axisIndex] = maxVal - minVal;
  }
  return dimensions;
}

void createGroundTruthModel(vector<Point3f> &model)
{
  model.clear();

  //TODO: move up
  const double b = 0.0650 / 2.0;
  const double a = 0.0533 / 2.0;
  const double H = 0.1262;

  double tanAlpha = (b - a) / H;
  for (double x = -a; x <= a; x += 0.0005)
  {
    for (double y = -a; y <= a; y += 0.0005)
    {
      if (x*x + y*y <= a*a)
      {
        model.push_back(Point3f(x, y, 0.0f));
      }
    }
  }

  for (double h = 0.0; h >= -H; h -= 0.001)
  {
    double r = a + (-h) * tanAlpha;
//    for (double phi = 0.0; phi < 2 * CV_PI; phi += CV_PI / 100)
    for (double phi = 0.0; phi < 2 * CV_PI; phi += CV_PI / 500)
    {
      double x = r * cos(phi);
      double y = r * sin(phi);

      model.push_back(Point3f(x, y, h));
    }
  }

  cout << model.size() << endl;
}

int main(int argc, char *argv[])
{
/*
  vector<Point3f> model;
  createGroundTruthModel(model);
  EdgeModel edgeModel(model, true, false);
  edgeModel.write("idealModel.xml");
  exit(-1);
*/

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
  bool compareWithKinFu = true;

  vector<string> trainObjectNames;
  trainObjectNames.push_back(objectName);

  PinholeCamera kinectCamera;
  vector<int> testIndices;
  Mat registrationMask;
  vector<EdgeModel> edgeModels;
  TODBaseImporter dataImporter(baseFolder, testFolder);
  if (compareWithKinFu)
  {
    dataImporter.importAllData(&modelsPath, &trainObjectNames, &kinectCamera, &registrationMask, &edgeModels, &testIndices);
  }
  else
  {
    dataImporter.importAllData(0, 0, &kinectCamera, &registrationMask, 0, &testIndices);
  }

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
//    fiducialPose = fiducialPose.inv() * zeroPose;

    int numberOfComponens;
    Mat glassMask;
    glassSegmentator.segment(bgrImage, depthImage, registrationMask, numberOfComponens, glassMask);
//    dataImporter.importRawMask(testImageIdx, glassMask);

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
  if (compareWithKinFu)
  {
    allModels.push_back(edgeModels[0].points);

    cout << "Quantitavie comparison with the KinFu model" << endl;
    cout << "Percentile\t SfS->KinFu\t KinFu->SfS" << endl;
    for (float percentile = 1.0f; percentile > 0.1f; percentile -= 0.2f)
    {
      cout << percentile << "\t\t ";
      cout << computePartialDirectionalHausdorffDistance(allModels[1], allModels[0], percentile, 1) << "\t ";
      cout << computePartialDirectionalHausdorffDistance(allModels[0], allModels[1], percentile, 1) << endl;
    }
    cout << endl;
  }

  cout << "Dimensions:" << endl;
  cout << Mat(computeModelDimensions(allModels[0])) << endl;
  if (compareWithKinFu)
  {
    cout << Mat(computeModelDimensions(allModels[1])) << endl;
  }

  publishPoints(allModels);
  return 0;
}

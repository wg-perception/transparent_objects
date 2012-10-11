#include <opencv2/opencv.hpp>

#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/glassSegmentator.hpp"
#include "modelCapture/modelCapturer.hpp"

#include <omp.h>

//#define CHECK_QUALITY_OF_POSES

using namespace cv;
using std::cout;
using std::endl;


void computePartialDirectionalHausdorffDistances(const std::vector<cv::Point3f> &baseCloud, const std::vector<cv::Point3f> &testCloud,
                                                 const std::vector<float> &percentiles, int neighbourIndex,
                                                 std::vector<float> &distances)
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

  distances.clear();
  for (size_t i = 0; i < percentiles.size(); ++i)
  {
    int percentileIndex = floor(percentiles[i] * (static_cast<int>(testCloud.size()) - 1));
    CV_Assert(percentileIndex >= 0 && percentileIndex < testCloud.size());
    std::nth_element(knnDists.begin(), knnDists.begin() + percentileIndex, knnDists.end());
    distances.push_back(knnDists[percentileIndex]);
  }
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
//  const bool isDense = true;
  const bool isDense = false;

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

    if (isDense)
    {
      for (double x = -r; x <= r; x += 0.0005)
      {
        for (double y = -r; y <= r; y += 0.0005)
        {
          if (x*x + y*y <= r*r)
          {
            model.push_back(Point3f(x, y, h));
          }
        }
      }
    }
    else
    {

  //    for (double phi = 0.0; phi < 2 * CV_PI; phi += CV_PI / 100)
      for (double phi = 0.0; phi < 2 * CV_PI; phi += CV_PI / 500)
      {
        double x = r * cos(phi);
        double y = r * sin(phi);

        model.push_back(Point3f(x, y, h));
      }
    }
  }

  cout << model.size() << endl;
}

int main(int argc, char *argv[])
{
  vector<Point3f> groundTruthModel;
  createGroundTruthModel(groundTruthModel);
/*
  EdgeModel edgeModel(model, true, false);
  edgeModel.write("idealModel.xml");
  exit(-1);
*/

  int omp_num_threads = 7;
#ifdef CHECK_QUALITY_OF_POSES
  omp_num_threads = 1;
#endif

  omp_set_num_threads(omp_num_threads);

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
  const bool compareWithKinFu = true;
  const bool useOdometryPoses = false;
  const bool useKeyFrames = false;

  vector<string> trainObjectNames;
  trainObjectNames.push_back(objectName);

  PinholeCamera kinectCamera;
  vector<int> testIndices;
  Mat registrationMask;
  vector<EdgeModel> edgeModels;
  TODBaseImporter dataImporter(baseFolder, testFolder);
  PoseRT objectOffset;
  if (compareWithKinFu)
  {
    dataImporter.importAllData(&modelsPath, &trainObjectNames, &kinectCamera, &registrationMask, &edgeModels, &testIndices, 0, 0, &objectOffset);
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

//fullModelCapture
//  glassSegmentationParams.grabCutErosionsIterations = 3;
//  glassSegmentationParams.grabCutDilationsIterations = 9;
//  glassSegmentationParams.grabCutMargin = 20;

//textureWithCircles
  glassSegmentationParams.grabCutErosionsIterations = 4;
  glassSegmentationParams.grabCutDilationsIterations = 4;

//textureWithCircles_hand
  glassSegmentationParams.openingIterations = 10;

  GlassSegmentator glassSegmentator(glassSegmentationParams);

  ModelCapturer modelCapturer(kinectCamera);
  if (compareWithKinFu)
  {
    vector<Point3f> rotatedGroundTruthModel;
//    project3dPoints(edgeModels[0].points, objectOffset, rotatedGroundTruthModel);
    project3dPoints(groundTruthModel, objectOffset, rotatedGroundTruthModel);
    modelCapturer.setGroundTruthModel(rotatedGroundTruthModel);
  }
  vector<ModelCapturer::Observation> observations(testIndices.size());
  vector<bool> isObservationValid(testIndices.size(), true);

  PoseRT zeroPose;
  if (useOdometryPoses)
  {
    const int zeroIndex = 99999;
    dataImporter.importGroundTruth(zeroIndex, zeroPose, false);
  }

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
    try
    {
      dataImporter.importGroundTruth(testImageIdx, fiducialPose, false, 0, useKeyFrames);
    }
    catch (cv::Exception ex)
    {
      isObservationValid[testIdx] = false;
      continue;
    }

    if (useOdometryPoses)
    {
      fiducialPose = fiducialPose.inv() * zeroPose;
    }



    int numberOfComponens;
    Mat glassMask;
    glassSegmentator.segment(bgrImage, depthImage, registrationMask, numberOfComponens, glassMask);
//    dataImporter.importRawMask(testImageIdx, glassMask);

//    showSegmentation(bgrImage, glassMask);
//    imshow("mask", glassMask);
//    waitKey(200);
//    waitKey();
    observations[testIdx].bgrImage = bgrImage;
    observations[testIdx].mask = glassMask;
    observations[testIdx].pose = fiducialPose;

#ifdef CHECK_QUALITY_OF_POSES
    fiducialPose = fiducialPose * objectOffset;

    Mat mask;
    Point tl;
    vector<Point2f> projectedGroundTruthModel;
    kinectCamera.projectPoints(edgeModels[0].points, fiducialPose, projectedGroundTruthModel);
    EdgeModel::computePointsMask(projectedGroundTruthModel, bgrImage.size(), 1.0, 3, mask, tl, false);
    showSegmentation(bgrImage, mask, "rgb");
    showSegmentation(depthImage, mask, "depth");

    showSegmentation(bgrImage, glassMask, "grab cut");
    waitKey();
#endif
  }

  modelCapturer.setObservations(observations, &isObservationValid);


  vector<Point3f> modelPoints;
  modelCapturer.createModel(modelPoints);

#if 1
for (size_t iter = 0; iter < 1; ++iter)
{
  cout << "final iteration #" << iter << endl;
#pragma omp parallel for
  for (size_t i = 0; i < observations.size(); ++i)
  {
    if (!isObservationValid[i])
    {
      continue;
    }

    vector<Point2f> projectedPoints;
    kinectCamera.projectPoints(modelPoints, observations[i].pose, projectedPoints);
    Mat newMask;
    Point tl;
    EdgeModel::computePointsMask(projectedPoints, observations[i].mask.size(), 1.0, 7, newMask, tl, false);
    observations[i].initMask = newMask;

//    showEdgels(observations[i].bgrImage, modelPoints, observations[i].pose, kinectCamera);
//    showSegmentation(observations[i].bgrImage, observations[i].mask);
//    waitKey();
    Mat refinedMask;
    glassSegmentationParams.grabCutErosionsIterations = 2;
    glassSegmentationParams.grabCutDilationsIterations = 2;
//    glassSegmentationParams.grabCutDilationsIterations = 6;
    glassSegmentationParams.grabCutIterations = 1;
//    glassSegmentationParams.grabCutIterations = 6;
    Mat bgdProbabilities, fgdProbabilities;
    refineSegmentationByGrabCut(observations[i].bgrImage, newMask, refinedMask, glassSegmentationParams,
                                  0, 0,
                                &bgdProbabilities, &fgdProbabilities);
    observations[i].mask = refinedMask;
    observations[i].bgdProbabilities = bgdProbabilities;
    observations[i].fgdProbabilities = fgdProbabilities;

//    showSegmentation(observations[i].bgrImage, newMask, "new mask");
//    showSegmentation(observations[i].bgrImage, observations[i].mask, "refined mask");
//    waitKey();
  }

/*
  vector<Mat> images, initMasks;
  for (size_t i = 0; i < 1; ++i)
  {
    CV_Assert(isObservationValid[i]);
    images.push_back(observations[i].bgrImage);
    initMasks.push_back(observations[i].initMask);
  }
  Mat bgdModel, fgdModel;
  glassSegmentationParams.grabCutErosionsIterations = 3;
  glassSegmentationParams.grabCutDilationsIterations = 6;
  createModels(images, initMasks, bgdModel, fgdModel, glassSegmentationParams);

  for (size_t i = 0; i < 50; ++i)
  {
    glassSegmentationParams.grabCutErosionsIterations = 2;
    glassSegmentationParams.grabCutDilationsIterations = 4;
    Mat refinedMask;
    refineSegmentationByGrabCut(images[i], initMasks[i], refinedMask, glassSegmentationParams, false, &bgdModel, &fgdModel);

    imshow("nm", refinedMask);
    showSegmentation(images[i], refinedMask, "new mask");
    imshow("sm", observations[i].mask);
    showSegmentation(images[i], observations[i].mask, "simple mask");
    waitKey();
  }
*/




  modelCapturer.setObservations(observations, &isObservationValid);
  vector<Point3f> confidentModelPoints;
  std::swap(confidentModelPoints, modelPoints);
  modelCapturer.createModel(modelPoints, &confidentModelPoints);

  vector<Point3f> newPoints;
  for (size_t i = 0; i < modelPoints.size(); ++i)
  {
    bool isNew = true;
    for (size_t j = 0; j < confidentModelPoints.size(); ++j)
    {
      if (modelPoints[i] == confidentModelPoints[j])
      {
          isNew = false;
          break;
      }
    }

    if (isNew)
    {
      newPoints.push_back(modelPoints[i]);
//      cout << modelPoints[i] << endl;
    }
  }

  vector<vector<Point3f> > allPoints;
  allPoints.push_back(confidentModelPoints);
  allPoints.push_back(newPoints);

  publishPoints(allPoints);
//  exit(-1);
}
#endif

  writePointCloud("model.asc", modelPoints);
  EdgeModel createdEdgeModel(modelPoints, true, true);

  //evaluation
  vector<vector<Point3f> > allModels;
  allModels.push_back(createdEdgeModel.points);
  if (compareWithKinFu)
  {
    allModels.push_back(edgeModels[0].points);

    vector<float> allPercentiles;
    vector<float> sfsToKinfuDitances, kinfuToSfsDistances;
    for (float percentile = 1.0f; percentile > 0.1f; percentile -= 0.2f)
    {
      allPercentiles.push_back(percentile);
    }
    computePartialDirectionalHausdorffDistances(allModels[1], allModels[0], allPercentiles, 1, sfsToKinfuDitances);
    computePartialDirectionalHausdorffDistances(allModels[0], allModels[1], allPercentiles, 1, kinfuToSfsDistances);

    cout << "Quantitavie comparison with the KinFu model" << endl;
    cout << "Percentile\t SfS->KinFu\t KinFu->SfS" << endl;
    for (size_t i = 0; i < allPercentiles.size(); ++i)
    {
      cout << allPercentiles[i] << "\t\t ";
      cout << sfsToKinfuDitances[i] << "\t ";
      cout << kinfuToSfsDistances[i] << endl;
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

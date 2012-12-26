/*
 * transparentExperiments.cpp
 *
 *  Created on: Aug 11, 2011
 *      Author: Ilya Lysenkov
 */

#include <opencv2/opencv.hpp>
#include <edges_pose_refiner/edgeModel.hpp>
#include <fstream>
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include <numeric>

#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/poseError.hpp"
#include <iomanip>
#include "edges_pose_refiner/pclProcessing.hpp"
#include <pcl/visualization/cloud_viewer.h>

#include "edges_pose_refiner/detector.hpp"


using namespace cv;
using namespace transpod;
using std::cout;
using std::endl;
using std::stringstream;

//#define VISUALIZE_TEST_DATA
//#define VISUALIZE_POSE_REFINEMENT
//#define WRITE_ERRORS
//#define PROFILE


#if defined(VISUALIZE_POSE_REFINEMENT) && defined(USE_3D_VISUALIZATION)
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>
#endif

void evaluatePose(const EdgeModel &testEdgeModel, const cv::Mat &rvec_est_cam, const cv::Mat &tvec_est_cam, const PoseRT &ground_cam, string prefix = "")
{
  double distance = 0;
  std::cout << prefix + "Hausdorff distance: " << distance << endl;

  Mat Rt_est_cam2test;
  createProjectiveMatrix(rvec_est_cam, tvec_est_cam, Rt_est_cam2test);

  Mat Rt_diff_cam = ground_cam.getProjectiveMatrix() * Rt_est_cam2test.inv(DECOMP_SVD);
  Mat Rt_obj2cam = testEdgeModel.Rt_obj2cam;
  Mat Rt_diff_obj = Rt_obj2cam.inv(DECOMP_SVD) * Rt_diff_cam * Rt_obj2cam;


  Mat rvec_diff_obj, tvec_diff_obj;
  getRvecTvec(Rt_diff_obj, rvec_diff_obj, tvec_diff_obj);
  double hartleyDiff = norm(rvec_diff_obj);


  const double radians2degrees = 180.0 / CV_PI;
  cout << "Hartley diff (deg): " << radians2degrees * hartleyDiff << endl;
  cout << "Angle diff (deg): " << 0 << endl;
  cout << "Normal diff (deg): " << 0 << endl;
  cout << "Tvec diff: " << norm(tvec_diff_obj) << endl;
}

void evaluatePoseWithRotation(const EdgeModel &originalEdgeModel, const PoseRT &est_cam, const PoseRT &ground_cam, PoseError &poseError)
{
  EdgeModel groundModel, estimatedModel;
  originalEdgeModel.rotate_cam(ground_cam, groundModel);
  originalEdgeModel.rotate_cam(est_cam, estimatedModel);

  const double eps = 1e-4;
  CV_Assert(groundModel.hasRotationSymmetry);
  CV_Assert(estimatedModel.hasRotationSymmetry);

  Point3d tvecPoint = groundModel.getObjectCenter() - estimatedModel.getObjectCenter();
  double tvecDiff = norm(tvecPoint);

  double cosAngle = groundModel.upStraightDirection.ddot(estimatedModel.upStraightDirection);
  cosAngle = std::min(cosAngle, 1.0);
  cosAngle = std::max(cosAngle, -1.0);
  double hartleyDiff = acos(cosAngle);

  PoseRT diff_cam = est_cam * ground_cam.inv();
  Mat Rt_diff_obj = groundModel.Rt_obj2cam.inv(DECOMP_SVD) * diff_cam.getProjectiveMatrix() * groundModel.Rt_obj2cam;
  Mat rvec_diff_obj, tvec_diff_obj;
  getRvecTvec(Rt_diff_obj, rvec_diff_obj, tvec_diff_obj);

/*
    point2col(tvecPoint, *tvec);
    Point3d rvecPoint = estimatedModel.rotationAxis.cross(groundModel.rotationAxis);
    rvecPoint *= hartleyDiff / norm(rvecPoint);
    point2col(rvecPoint, *rvec);
*/
  poseError.init(PoseRT(rvec_diff_obj, tvec_diff_obj), hartleyDiff, tvecDiff);

/*

  Point3d zRvecPoint_obj = estimatedModel.rotationAxis.cross(groundModel.rotationAxis);

  CV_Assert(norm(zRvecPoint_obj) > eps);
  zRvecPoint_obj *= hartleyDiff / norm(zRvecPoint_obj);
  Mat zRvec_obj;
  point2col(zRvecPoint_obj, zRvec_obj);
  const int dim = 3;
  Point3d tvecPoint_obj = groundModel.getObjectCenter() - estimatedModel.getObjectCenter();

  zRvec_obj = Mat::zeros(dim, 1, CV_64FC1);

  Mat zTvec_obj;
  point2col(tvecPoint_obj, zTvec_obj);

  //zTvec_obj = Mat::zeros(dim, 1, CV_64FC1);

  PoseRT zPose_obj = PoseRT(zRvec_obj, zTvec_obj);
  Mat withoutZRotation_Rt = estimatedModel.Rt_obj2cam * zPose_obj.getProjectiveMatrix() * estimatedModel.Rt_obj2cam.inv(DECOMP_SVD) * est_cam.getProjectiveMatrix();
  PoseRT withoutZRotationPose = PoseRT(withoutZRotation_Rt);


  double xyRotationDiff, xyTranslationDiff;
  PoseRT::computeDistance(ground_cam, withoutZRotationPose, xyRotationDiff, xyTranslationDiff, groundModel.Rt_obj2cam);
  //PoseRT::computeDistance(ground_cam, withoutZRotationPose, xyRotationDiff, xyTranslationDiff);
  cout << "xy: " << xyTranslationDiff << " " << xyRotationDiff * 180.0 / CV_PI<< endl;
*/
}

void writeImages(const vector<Mat> &images, const string &path, int testIdx, int translationIdx, int rotationIdx, const string &postfix)
{
  stringstream name;
  name << path << "/" << testIdx << "_" << translationIdx << "_" << rotationIdx << "_" << postfix << ".png" ;

  Mat commonImage;
  commonImage = images[0].t();
  for(size_t i=1; i<images.size(); i++)
  {
    Mat transposedImage = images[i].t();
    commonImage.push_back(transposedImage);
  }
  commonImage = commonImage.t();

  imwrite(name.str(), commonImage);
}

float computeOcclusionPercentage(const PinholeCamera &camera,
                                 const EdgeModel &testObject, const PoseRT &objectOffset,
                                 const std::vector<EdgeModel> &occlusionObjects, const std::vector<PoseRT> &occlusionOffsets,
                                 const PoseRT &model2test_fiducials)
{
  //TODO: move up
  const float downFactor = 1.0f;
  const int closingIterationsCount = 7;
  const int objectColor = 127;
  const int occlusionColor = 255;

  PoseRT objectPose = model2test_fiducials * objectOffset;
  Silhouette objectSilhouette;
  Ptr<PinholeCamera> cameraPtr = new PinholeCamera(camera);
  testObject.getSilhouette(cameraPtr, objectPose, objectSilhouette, downFactor, closingIterationsCount);

  Mat image(camera.imageSize, CV_8UC1, Scalar(0));
  objectSilhouette.draw(image, Scalar(objectColor), 0);

  for (size_t i = 0; i < occlusionObjects.size(); ++i)
  {
    PoseRT pose_cam = model2test_fiducials * occlusionOffsets[i];
    Silhouette silhouette;
    occlusionObjects[i].getSilhouette(cameraPtr, pose_cam, silhouette, downFactor, closingIterationsCount);
    silhouette.draw(image, Scalar(occlusionColor), -1);
  }

//  imshow("occlusions", image);
//  waitKey(200);

  Mat edgels;
  objectSilhouette.getEdgels(edgels);
  vector<Point2f> contour = edgels;

  bool isOccluded = true;
  int firstUnoccludedIndex = 0;
  double unoccludedLength = 0.0;
  for (int i = 0; i < edgels.rows; ++i)
  {
    Point pt = contour[i];
    if (image.at<uchar>(pt) == objectColor && isOccluded)
    {
      firstUnoccludedIndex = i;
      isOccluded = false;
    }

    if (image.at<uchar>(pt) == occlusionColor && !isOccluded)
    {
      isOccluded = true;

      unoccludedLength += arcLength(edgels.rowRange(firstUnoccludedIndex, i), false);
    }
  }
  if (!isOccluded)
  {
    bool isClosed = (firstUnoccludedIndex == 0);
    unoccludedLength += arcLength(edgels.rowRange(firstUnoccludedIndex, edgels.rows), isClosed);
  }

  double contourLength = arcLength(edgels, true);

  const float eps = 1e-2;
  CV_Assert(contourLength > eps);
  return 1.0 - (unoccludedLength / contourLength);
}

int main(int argc, char **argv)
{
  std::system("date");

  if (argc != 4)
  {
    cout << argv[0] << " <modelsPath> <baseFoldler> <testObjectName>" << endl;
    return -1;
  }

  const string trainedModelsPath = argv[1];
  const string baseFolder = argv[2];
  const string testObjectName = argv[3];

  const string testFolder = baseFolder + "/" + testObjectName + "/";
//  const string visualizationPath = "visualized_results/";
  const string errorsVisualizationPath = "/home/ilysenkov/errors/";
//  const vector<string> objectNames = {"bank", "bucket"};
//  const vector<string> objectNames = {"bank", "bottle", "bucket", "glass", "wineglass"};
  const vector<string> objectNames = {testObjectName};


  TODBaseImporter dataImporter(baseFolder, testFolder);

  PinholeCamera kinectCamera;
  vector<EdgeModel> edgeModels;
  vector<EdgeModel> occlusionObjects;
  vector<PoseRT> occlusionOffsets;
  vector<int> testIndices;
  Mat registrationMask;
  dataImporter.importAllData(&trainedModelsPath, &objectNames,
                             &kinectCamera, &registrationMask, &edgeModels, &testIndices, &occlusionObjects, &occlusionOffsets);

  DetectorParams params;
//  params.glassSegmentationParams.closingIterations = 8;
// bucket
//  params.glassSegmentationParams.openingIterations = 8;

  //good clutter
  params.glassSegmentationParams.openingIterations = 15;
  params.glassSegmentationParams.closingIterations = 12;
  params.glassSegmentationParams.finalClosingIterations = 32;
  params.glassSegmentationParams.grabCutErosionsIterations = 4;
  params.planeSegmentationMethod = FIDUCIALS;

  //fixedOnTable
  //params.glassSegmentationParams.finalClosingIterations = 8;

  //clutter
  //bucket
  //params.glassSegmentationParams.finalClosingIterations = 12;

  Detector detector(kinectCamera, params);
  for (size_t i = 0; i < edgeModels.size(); ++i)
  {
    detector.addTrainObject(objectNames[i], edgeModels[i]);
  }

  vector<size_t> initialPoseCount;
  vector<PoseError> bestPoses, bestInitialPoses;
  int segmentationFailuresCount = 0;
  int badSegmentationCount = 0;

  vector<float> allChamferDistances;
  vector<size_t> geometricHashingPoseCount;
//  vector<int> indicesOfRecognizedObjects;
  vector<double> allRecognitionTimes;
  for(size_t testIdx = 0; testIdx < testIndices.size(); testIdx++)
  {
    srand(42);
    RNG &rng = theRNG();
    rng.state = 0xffffffff;

#if defined(VISUALIZE_POSE_REFINEMENT) && defined(USE_3D_VISUALIZATION)
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("transparent experiments"));
#endif
    int testImageIdx = testIndices[ testIdx ];
    cout << "Test: " << testIdx << " " << testImageIdx << endl;

    Mat kinectBgrImage, kinectDepth;
    dataImporter.importBGRImage(testImageIdx, kinectBgrImage);
    dataImporter.importDepth(testImageIdx, kinectDepth);

/*
    Mat silhouetteImage(480, 640, CV_8UC1, Scalar(0));
    silhouettes[0].draw(silhouetteImage);
    imshow("silhouette", silhouetteImage);
    imshow("mask", centerMask);
    waitKey();



    vector<Point2f> glassContour;
    mask2contour(centerMask, glassContour);
    Mat silhouette2test;
    silhouettes[0].match(Mat(glassContour), silhouette2test);
    exit(-1);
*/

    PoseRT model2test_fiducials, objectOffset;
    dataImporter.importGroundTruth(testImageIdx, model2test_fiducials, false, &objectOffset);
    PoseRT model2test_ground = model2test_fiducials * objectOffset;
//    cout << "Ground truth: " << model2test_ground << endl;

    CV_Assert(edgeModels.size() == 1);
    float occlusionPercentage = computeOcclusionPercentage(kinectCamera, edgeModels[0], objectOffset, occlusionObjects, occlusionOffsets, model2test_fiducials);
    CV_Assert(0.0 <= occlusionPercentage && occlusionPercentage <= 1.0);
    cout << "occlusion percentage: " << occlusionPercentage << endl;

    pcl::PointCloud<pcl::PointXYZ> testPointCloud;
#ifdef USE_3D_VISUALIZATION
    dataImporter.importPointCloud(testImageIdx, testPointCloud);
#endif

#ifdef VISUALIZE_TEST_DATA
    imshow("rgb", kinectBgrImage);
    imshow("depth", kinectDepth * 20);
#endif

#ifdef VISUALIZE_POSE_REFINEMENT
#ifdef USE_3D_VISUALIZATION
    {
      vector<Point3f> cvTestPointCloud;
      pcl2cv(testPointCloud, cvTestPointCloud);
      cout << "test point cloud size: " << cvTestPointCloud.size() << endl;
      publishPoints(cvTestPointCloud, viewer, Scalar(0, 255, 0), "test point cloud");
    }

    publishPoints(edgeModels[0].points, viewer, Scalar(0, 0, 255), "ground object", model2test_ground);
#endif

    if(!kinectCameraFilename.empty())
    {
//      displayEdgels(glassMask, edgeModels[0].points, model2test_ground, kinectCamera, "kinect");
      showEdgels(kinectBgrImage, edgeModels[0].points, model2test_ground, kinectCamera, "ground truth");
      showEdgels(kinectBgrImage, edgeModels[0].stableEdgels, model2test_ground, kinectCamera, "ground truth surface");
    }
    namedWindow("ground truth");
#ifdef USE_3D_VISUALIZATION
    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    viewer->resetStoppedFlag();
#endif
    waitKey();
    destroyWindow("ground truth");
#endif

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
#ifdef VISUALIZE_POSE_REFINEMENT
    Mat glassMask = debugInfo.glassMask;
    imshow("glassMask", glassMask);
    showSegmentation(kinectBgrImage, glassMask, "segmentation");

    Mat detectionResults = kinectBgrImage.clone();
    detector.visualize(poses_cam, detectedObjectsNames, detectionResults);
    imshow("detection", detectionResults);
    waitKey();
#endif

#ifndef PROFILE

    if (edgeModels.size() == 1)
    {
      vector<Point2f> groundEdgels;
      kinectCamera.projectPoints(edgeModels[0].points, model2test_ground, groundEdgels);

      vector<float> chamferDistances;
      for (size_t silhouetteIndex = 0; silhouetteIndex < debugInfo.initialSilhouettes.size(); ++silhouetteIndex)
      {
        vector<Point2f> silhouette = debugInfo.initialSilhouettes[silhouetteIndex];

        double silhoutteDistance = 0.0;
        for (int i = 0; i < silhouette.size(); ++i)
        {
          float minDist = std::numeric_limits<float>::max();
          for (int j = 0; j < groundEdgels.size(); ++j)
          {
            float currentDist = norm(silhouette[i] - groundEdgels[j]);
            if (currentDist < minDist)
            {
              minDist = currentDist;
            }
          }
          silhoutteDistance += minDist;
        }
        silhoutteDistance /= silhouette.size();
        chamferDistances.push_back(silhoutteDistance);
      }
      std::sort(chamferDistances.begin(), chamferDistances.end());
      if (chamferDistances.empty())
      {
        chamferDistances.push_back(std::numeric_limits<float>::max());
      }
      cout << "Best geometric hashing pose (px): " << chamferDistances[0] << endl;
      cout << "Number of initial poses: " << chamferDistances.size() << endl;
      allChamferDistances.push_back(chamferDistances[0]);
      geometricHashingPoseCount.push_back(chamferDistances.size());
    }

    if (poses_cam.size() == 0)
    {
      ++segmentationFailuresCount;
      continue;
    }

    if (!posesQualities.empty())
    {
      std::vector<float>::iterator bestDetection = std::min_element(posesQualities.begin(), posesQualities.end());
      int bestDetectionIndex = std::distance(posesQualities.begin(), bestDetection);
//      int detectedObjectIndex = detector.getTrainObjectIndex(detectedObjectsNames[bestDetectionIndex]);
//      indicesOfRecognizedObjects.push_back(detectedObjectIndex);
      cout << "Recognized object: " << detectedObjectsNames[bestDetectionIndex] << endl;
    }

    cout << "Recognition time: " << recognitionTime.getTimeSec() << "s" << endl;
    allRecognitionTimes.push_back(recognitionTime.getTimeSec());

    if (objectNames.size() == 1)
    {
      cout << "initial poses: " << debugInfo.initialPoses.size() << endl;
      vector<PoseError> initialPoseErrors;
      for (size_t i = 0 ; i < debugInfo.initialPoses.size(); ++i)
      {
        PoseError poseError;
        evaluatePoseWithRotation(edgeModels[0], debugInfo.initialPoses[i], model2test_ground, poseError);
        cout << poseError << endl;
        initialPoseErrors.push_back(poseError);
//        showEdgels(kinectBgrImage, edgeModels[0].points, debugInfo.initialPoses[i], kinectCamera, "gh pose");
//        waitKey();
      }
      cout << "the end." << endl;
      PoseError currentBestInitialError;
      {
        vector<PoseError>::iterator bestPoseIt = std::min_element(initialPoseErrors.begin(), initialPoseErrors.end());
        int bestPoseIdx = std::distance(initialPoseErrors.begin(), bestPoseIt);
        currentBestInitialError = initialPoseErrors[bestPoseIdx];
        cout << "Best initial error: " << currentBestInitialError << endl;
        bestInitialPoses.push_back(currentBestInitialError);
      }


      CV_Assert(poses_cam.size() == 1);
      int objectIndex = 0;
      initialPoseCount.push_back(poses_cam.size());

      vector<PoseError> currentPoseErrors(poses_cam.size());
      for (size_t i = 0 ; i < poses_cam.size(); ++i)
      {
        evaluatePoseWithRotation(edgeModels[objectIndex], poses_cam[i], model2test_ground, currentPoseErrors[i]);
        cout << currentPoseErrors[i] << endl;

#ifdef VISUALIZE_POSE_REFINEMENT
        namedWindow("pose is ready");
        waitKey();
        destroyWindow("pose is ready");
//        displayEdgels(glassMask, edgeModels[objectIndex].points, initPoses_cam[objectIndex][i], kinectCamera, "initial");
#ifdef USE_3D_VISUALIZATION
        publishPoints(edgeModels[objectIndex].points, viewer, Scalar(255, 0, 0), "final object", poses_cam[i]);
#endif
        showEdgels(kinectBgrImage, edgeModels[objectIndex].points, poses_cam[i], kinectCamera, "final");
        showEdgels(kinectBgrImage, edgeModels[objectIndex].stableEdgels, poses_cam[i], kinectCamera, "final surface");
        namedWindow("initial pose");

#ifdef USE_3D_VISUALIZATION
        while (!viewer->wasStopped ())
        {
          viewer->spinOnce (100);
          boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
#endif
        waitKey();
        destroyWindow("initial pose");
  #endif
      }
      vector<PoseError>::iterator bestPoseIt = std::min_element(currentPoseErrors.begin(), currentPoseErrors.end());
      int bestPoseIdx = std::distance(currentPoseErrors.begin(), bestPoseIt);
      PoseError currentBestError = currentPoseErrors[bestPoseIdx];
      cout << "Best pose: " << currentBestError << endl;
      bestPoses.push_back(currentBestError);

      cout << "Result: " << occlusionPercentage << ", " << debugInfo.initialPoses.size() << ", " <<
              currentBestInitialError.getTranslationDifference() << ", " << currentBestInitialError.getRotationDifference(false) << ", " <<
              currentBestError.getTranslationDifference() << ", " << currentBestError.getRotationDifference(false) << endl;

#ifdef WRITE_ERRORS
      const float maxTrans = 0.02;
      if (currentPoseErrors[bestPoseIdx].getTranslationDifference() > maxTrans)
      {
        Mat glassMask = debugInfo.glassMask;
        std::stringstream str;
        str << testImageIdx;
        Mat segmentation = drawSegmentation(kinectBgrImage, glassMask);
        imwrite(errorsVisualizationPath + "/" + objectNames[0] + "_" + str.str() + "_mask.png", segmentation);

        Mat poseImage = kinectBgrImage.clone();
        detector.visualize(poses_cam, detectedObjectsNames, poseImage);
        imwrite(errorsVisualizationPath + "/" + objectNames[0] + "_" + str.str() + "_pose.png", poseImage);

        const float depthNormalizationFactor = 100;
        imwrite(errorsVisualizationPath + "/" + objectNames[0] + "_" + str.str() + "_depth.png", kinectDepth * depthNormalizationFactor);
      }
#endif
    }
#endif
    cout << endl;

#ifdef PROFILE
    return 0;
#endif
  }


  cout << "Segmentation failures: " << static_cast<float>(segmentationFailuresCount) / testIndices.size() << endl;
  cout << "Bad segmentation rate: " << static_cast<float>(badSegmentationCount) / testIndices.size() << endl;

/*
  cout << "Recognition statistics:" << endl;
  for (size_t i = 0; i < objectNames.size(); ++i)
  {
    cout << countNonZero(Mat(indicesOfRecognizedObjects) == i) << " ";
  }
  cout << endl;

  for (size_t i = 0; i < objectNames.size(); ++i)
  {
    cout << countNonZero(Mat(indicesOfRecognizedObjects) == i) / static_cast<float>(indicesOfRecognizedObjects.size())  << " ";
  }
  cout << endl;
*/

  if (objectNames.size() == 1)
  {

    float meanInitialPoseCount = std::accumulate(initialPoseCount.begin(), initialPoseCount.end(), 0);
    meanInitialPoseCount /= initialPoseCount.size();
    cout << "mean initial pose count: " << meanInitialPoseCount << endl;

    //TODO: move up
    const double cmThreshold = 2.0;

//    const double cmThreshold = 5.0;
    PoseError::evaluateErrors(bestPoses, cmThreshold);

    cout << "initial poses:" << endl;
    //TODO: move up
    PoseError::evaluateErrors(bestInitialPoses, 3.0 * cmThreshold);
  }

  cout << "Evaluation of geometric hashing" << endl;
  std::sort(allChamferDistances.begin(), allChamferDistances.end());
  const float successfulChamferDistance = 5.0f;
  int ghSuccessCount = 0;
  double meanChamferDistance = 0.0;
  for (size_t i = 0; i < allChamferDistances.size(); ++i)
  {
    cout << i << "\t: " << allChamferDistances[i] << endl;
    if (allChamferDistances[i] < successfulChamferDistance)
    {
      ++ghSuccessCount;
      meanChamferDistance += allChamferDistances[i];
    }
  }
  if (ghSuccessCount != 0)
  {
    meanChamferDistance /= ghSuccessCount;
  }
  int posesSum = std::accumulate(geometricHashingPoseCount.begin(), geometricHashingPoseCount.end(), 0);
  float meanInitialPoseCount = static_cast<float>(posesSum) / initialPoseCount.size();
  cout << "Mean number of initial poses: " << meanInitialPoseCount << endl;

  float ghSuccessRate = static_cast<float>(ghSuccessCount) / allChamferDistances.size();
  cout << "Success rate: " << ghSuccessRate << endl;
  cout << "Mean chamfer distance (px): " << meanChamferDistance << endl;

  double timesSum = std::accumulate(allRecognitionTimes.begin(), allRecognitionTimes.end(), 0.0);
  double meanRecognitionTime = timesSum / allRecognitionTimes.size();
  cout << "Mean recognition time (s): " << meanRecognitionTime << endl;

  std::system("date");
  return 0;
}

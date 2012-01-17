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

#include "TODBaseImporter.hpp"
#include "edges_pose_refiner/poseError.hpp"
#include <iomanip>
#include "edges_pose_refiner/pclProcessing.hpp"
#include <pcl/visualization/cloud_viewer.h>

#include "edges_pose_refiner/transparentDetector.hpp"


using namespace cv;
using std::cout;
using std::endl;
using std::stringstream;

//#define VISUALIZE_POSE_REFINEMENT
//#define VISUALIZE_INITIAL_POSE_REFINEMENT
//#define WRITE_RESULTS
//#define PROFILE
//#define WRITE_GLASS_SEGMENTATION


#ifdef VISUALIZE_POSE_REFINEMENT
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

int main(int argc, char **argv)
{
  std::system("date");

  CV_Assert(argc == 3);
  string baseFolder = argv[1];
  string testObjectName = argv[2];

  //const string modelFilename = "finalModels/" + objectName + ".xml";
  const string modelsPath = "/media/2Tb/transparentBases/trainedModels/";

  const string trainFolder ="/media/2Tb/transparentBases/base_with_ground_truth/base/wh_" + testObjectName + "/";
  const string testFolder = baseFolder + "/" + testObjectName + "/";

//  const string camerasListFilename = baseFolder + "/cameras.txt";
  const string kinectCameraFilename = baseFolder + "/center.yml";
//  const string visualizationPath = "visualized_results/";
  const string errorsVisualizationPath = "/home/ilysenkov/errors/";
  //const vector<string> objectNames = {"bank", "bucket"};
//  const vector<string> objectNames = {"bank", "bottle", "bucket", "glass", "wineglass"};
  const string registrationMaskFilename = baseFolder + "/registrationMask.png";

  const vector<string> objectNames = {testObjectName};



  TODBaseImporter dataImporter(trainFolder, testFolder);

  PinholeCamera kinectCamera;
  if(!kinectCameraFilename.empty())
  {
    dataImporter.readCameraParams(kinectCameraFilename, kinectCamera, false);
    CV_Assert(kinectCamera.imageSize == Size(640, 480));
  }

  vector<EdgeModel> edgeModels(objectNames.size());
  for (size_t i = 0; i < objectNames.size(); ++i)
  {
    dataImporter.importEdgeModel(modelsPath, objectNames[i], edgeModels[i]);
    cout << "All points in the model: " << edgeModels[i].points.size() << endl;
    cout << "Surface points in the model: " << edgeModels[i].stableEdgels.size() << endl;
  }

//#ifdef VISUALIZE_POSE_REFINEMENT
//  edgeModels[0].visualize();
//#endif

  TransparentDetectorParams params;
  params.glassSegmentationParams.closingIterations = 8;
  params.glassSegmentationParams.openingIterations = 15;
  params.glassSegmentationParams.finalClosingIterations = 8;

//  TransparentDetector detector(kinectCamera, params);
  TransparentDetector detector(kinectCamera);
  for (size_t i = 0; i < edgeModels.size(); ++i)
  {
    detector.addModel(objectNames[i], edgeModels[i]);
  }

  vector<int> testIndices;
  dataImporter.importTestIndices(testIndices);

  Mat registrationMask = imread(registrationMaskFilename, CV_LOAD_IMAGE_GRAYSCALE);
  CV_Assert(!registrationMask.empty());

  vector<size_t> initialPoseCount;
  vector<PoseError> bestPoses;
  int segmentationFailuresCount = 0;
  int badSegmentationCount = 0;

  vector<int> indicesOfRecognizedObjects;
  for(size_t testIdx = 0; testIdx < testIndices.size(); testIdx++)
  {
#ifdef VISUALIZE_POSE_REFINEMENT
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("transparent experiments"));
#endif
    int testImageIdx = testIndices[ testIdx ];
    cout << "Test: " << testIdx << " " << testImageIdx << endl;

    Mat kinectDepth, kinectBgrImage;
    if(!kinectCameraFilename.empty())
    {
      dataImporter.importBGRImage(testImageIdx, kinectBgrImage);
      dataImporter.importDepth(testImageIdx, kinectDepth);
    }


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

    PoseRT model2test_ground;
    dataImporter.importGroundTruth(testImageIdx, model2test_ground);
//    cout << "Ground truth: " << model2test_ground << endl;

    pcl::PointCloud<pcl::PointXYZ> testPointCloud;
    dataImporter.importPointCloud(testImageIdx, testPointCloud);

#ifdef VISUALIZE_POSE_REFINEMENT
    {
      vector<Point3f> cvTestPointCloud;
      pcl2cv(testPointCloud, cvTestPointCloud);
      cout << "test point cloud size: " << cvTestPointCloud.size() << endl;
      publishPoints(cvTestPointCloud, viewer, Scalar(0, 255, 0), "test point cloud");
    }

    if(!kinectCameraFilename.empty())
    {
//      displayEdgels(glassMask, edgeModels[0].points, model2test_ground, kinectCamera, "kinect");
      displayEdgels(kinectBgrImage, edgeModels[0].points, model2test_ground, kinectCamera, "ground truth");
      displayEdgels(kinectBgrImage, edgeModels[0].stableEdgels, model2test_ground, kinectCamera, "ground truth surface");
    }
    publishPoints(edgeModels[0].points, viewer, Scalar(0, 0, 255), "ground object", model2test_ground);
    namedWindow("ground truth");
    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
      boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    viewer->resetStoppedFlag();
    waitKey();
    destroyWindow("ground truth");
#endif

    vector<PoseRT> poses_cam;
    vector<float> posesQualities;
    vector<string> detectedObjectsNames;

    TickMeter recognitionTime;
    recognitionTime.start();

    Mat glassMask;
    try
    {
      glassMask = detector.detect(kinectBgrImage, kinectDepth, registrationMask, testPointCloud, poses_cam, posesQualities, detectedObjectsNames);
    }
    catch(const cv::Exception &)
    {
    }
    if (poses_cam.size() == 0)
    {
      ++segmentationFailuresCount;
      continue;
    }

    recognitionTime.stop();

    cout << poses_cam.size() << endl;

    if (!posesQualities.empty())
    {
      std::vector<float>::iterator bestDetection = std::min_element(posesQualities.begin(), posesQualities.end());
      int bestDetectionIndex = std::distance(posesQualities.begin(), bestDetection);
      int detectedObjectIndex = detector.getObjectIndex(detectedObjectsNames[bestDetectionIndex]);
      indicesOfRecognizedObjects.push_back(detectedObjectIndex);
      cout << "Recognized object: " << detectedObjectsNames[bestDetectionIndex] << endl;
    }

    cout << "Recognition time: " << recognitionTime.getTimeSec() << "s" << endl;

    if (objectNames.size() == 1)
    {
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
#endif

  #ifdef VISUALIZE_POSE_REFINEMENT
//        displayEdgels(glassMask, edgeModels[objectIndex].points, initPoses_cam[objectIndex][i], kinectCamera, "initial");
        publishPoints(edgeModels[objectIndex].points, viewer, Scalar(255, 0, 0), "final object", poses_cam[i]);
        displayEdgels(kinectBgrImage, edgeModels[objectIndex].points, poses_cam[i], kinectCamera, "final");
        displayEdgels(kinectBgrImage, edgeModels[objectIndex].stableEdgels, poses_cam[i], kinectCamera, "final surface");
        namedWindow("initial pose");

        while (!viewer->wasStopped ())
        {
          viewer->spinOnce (100);
          boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
        waitKey();
        destroyWindow("initial pose");
  #endif
      }
      vector<PoseError>::iterator bestPoseIt = std::min_element(currentPoseErrors.begin(), currentPoseErrors.end());
      int bestPoseIdx = std::distance(currentPoseErrors.begin(), bestPoseIt);
      cout << "Best pose: " << currentPoseErrors[bestPoseIdx] << endl;
      bestPoses.push_back(currentPoseErrors[bestPoseIdx]);

#ifdef WRITE_RESULTS
      const float maxTrans = 0.02;
      if (currentPoseErrors[bestPoseIdx].getTranslationDifference() > maxTrans)
      {
        std::stringstream str;
        str << testImageIdx;
        Mat segmentation = drawSegmentation(kinectBgrImage, glassMask);
        imwrite(errorsVisualizationPath + "/" + objectNames[0] + "_" + str.str() + "_mask.png", segmentation);

        Mat poseImage = displayEdgels(kinectBgrImage, edgeModels[objectIndex].points, poses_cam[bestPoseIdx], kinectCamera, "final");
        imwrite(errorsVisualizationPath + "/" + objectNames[0] + "_" + str.str() + "_pose.png", poseImage);

        const float depthNormalizationFactor = 100;
        imwrite(errorsVisualizationPath + "/" + objectNames[0] + "_" + str.str() + "_depth.png", kinectDepth * depthNormalizationFactor);
      }
#endif
    }

#ifdef PROFILE
    return 0;
#endif
  }


  cout << "Segmentation failures: " << static_cast<float>(segmentationFailuresCount) / testIndices.size() << endl;
  cout << "Bad segmentation rate: " << static_cast<float>(badSegmentationCount) / testIndices.size() << endl;

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


  if (objectNames.size() == 1)
  {

    float meanInitialPoseCount = std::accumulate(initialPoseCount.begin(), initialPoseCount.end(), 0);
    meanInitialPoseCount /= initialPoseCount.size();
    cout << "mean initial pose count: " << meanInitialPoseCount << endl;

    const double cmThreshold = 2.0;
    PoseError::evaluateErrors(bestPoses, cmThreshold);
  }

  std::system("date");
  return 0;
}

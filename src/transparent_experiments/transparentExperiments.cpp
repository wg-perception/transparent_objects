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

#ifdef USE_3D_VISUALIZATION
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>
#endif

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
void drawAxis(const Point3d &origin, const Point3d &direction, const ros::Publisher &points_pub, int id, Scalar color)
{
  vector<Point3f> points;
  const int pointsCount = 1000;
  for(int i=0; i<pointsCount; i++)
  {
    points.push_back(origin + direction * i * 0.001);
    points.push_back(origin - direction * i * 0.001);
  }

  publishPoints(points, points_pub, id, color);
}
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

void evaluatePoseWithRotation(const EdgeModel &originalEdgeModel, const PoseRT &est_cam, const PoseRT &ground_cam, PoseError &poseError, ros::Publisher *pointsPublisher = 0)
{
  EdgeModel groundModel, estimatedModel;
  originalEdgeModel.rotate_cam(ground_cam, groundModel);
  originalEdgeModel.rotate_cam(est_cam, estimatedModel);

  const double eps = 1e-4;
  CV_Assert(groundModel.hasRotationSymmetry);
  CV_Assert(estimatedModel.hasRotationSymmetry);

  double hartleyDiff, tvecDiff;
  Point3d tvecPoint = groundModel.getObjectCenter() - estimatedModel.getObjectCenter();
  tvecDiff = norm(tvecPoint);
  hartleyDiff = acos(groundModel.upStraightDirection.ddot(estimatedModel.upStraightDirection));

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

  const string camerasListFilename = baseFolder + "/cameras.txt";
  const string kinectCameraFilename = baseFolder + "/center.yml";
  const string visualizationPath = "visualized_results/";
  //const string errorsVisualizationPath = "errors/" + objectName;
  //const vector<string> objectNames = {"bank", "bucket"};
  //const vector<string> objectNames = {"bank", "bottle", "bucket", "glass", "wineglass"};
  const string registrationMaskFilename = baseFolder + "/registrationMask.png";

  const vector<string> objectNames = {testObjectName};


#ifdef VISUALIZE_POSE_REFINEMENT
//  ros::init(argc, argv, "transparent4");
//  ros::NodeHandle nh("~");
//  ros::Publisher pt_pub = nh.advertise<visualization_msgs::Marker> ("pose_points", 0);
//  ros::Publisher *publisher = &pt_pub;
#else
//  ros::Publisher *publisher = 0;
#endif

#ifdef USE_3D_VISUALIZATION
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("detected objects"));
#endif


  ros::Publisher *publisher = 0;
  TODBaseImporter dataImporter(trainFolder, testFolder, publisher);

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

#ifdef VISUALIZE_POSE_REFINEMENT
//  publishPoints(edgeModels[0].points, pt_pub, 0, Scalar(255, 255, 0));
//  publishPoints(edgeModels[0].stableEdgels, pt_pub, 1, Scalar(0, 0, 0));

  Scalar meanVal = mean(Mat(edgeModels[0].points));
  Point3d center(meanVal[0], meanVal[1], meanVal[2]);

//  drawAxis(center, Point3d(0, 0, 1), pt_pub, 40, Scalar(255, 0, 255));

//  publishPoints(vector<Point3f>(1, edgeModels[0].tableAnchor), pt_pub, 11203, Scalar(0, 255, 0));
//  publishPoints(vector<Point3f>(1, edgeModels[0].tableAnchor + 0.1 * edgeModels[0].rotationAxis), pt_pub, 11205, Scalar(0, 0, 0));
//  namedWindow("edge model is published");
//  waitKey();


/*
  //load test point cloud
  {
    vector<vector<Point3f> > publishedPointClouds(2);

    string testCloudFilename = testFolder + "/cloud_00000.pcd";
    pcl::PointCloud<pcl::PointXYZ> testPointCloud;
    pcl::io::loadPCDFile(testCloudFilename.c_str(), testPointCloud);
//    std::ofstream fout("testCloud.asc");
    for(size_t ptIdx=0; ptIdx < testPointCloud.points.size(); ptIdx++)
    {
      pcl::PointXYZ ptXYZ = testPointCloud.points[ptIdx];
      Point3f pt3f(ptXYZ.x, ptXYZ.y, ptXYZ.z);
      if(isNan(pt3f))
        continue;

      publishedPointClouds[0].push_back(pt3f);
//      fout << format(mat, "csv") << endl;
    }
//    fout.close();
    publishPoints(publishedPointClouds[0], pt_pub, 23552, Scalar(0, 255, 0));
//    namedWindow("stop");
//    waitKey();
  }
*/
#endif

//  TransparentDetectorParams params;
//  params.glassSegmentationParams.closingIterations = 8;
//  params.glassSegmentationParams.openingIterations = 15;
//  params.glassSegmentationParams.finalClosingIterations = 8;
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


#ifdef VISUALIZE_POSE_REFINEMENT
    {
      stringstream pointCloudFilename;
      pointCloudFilename << testFolder << "/cloud_" << std::setfill('0') << std::setw(5) << testImageIdx << ".pcd";
      pcl::PointCloud<pcl::PointXYZ> testPointCloudPCL;
      pcl::io::loadPCDFile(pointCloudFilename.str().c_str(), testPointCloudPCL);
      vector<Point3f> testPointCloud;
      pcl2cv(testPointCloudPCL, testPointCloud);
      vector<Point3f> filteredPointCloud;
      for(size_t i = 0; i < testPointCloud.size(); i++)
      {
        if(isNan(testPointCloud[i]))
          continue;

        filteredPointCloud.push_back(testPointCloud[i]);
      }
      std::swap(filteredPointCloud, testPointCloud);
      cout << "test point cloud size: " <<testPointCloud.size() << endl;
//      publishPoints(testPointCloud, pt_pub, 222, Scalar(0, 255, 0));
    }

/*
      Mat rvecZeros = Mat::zeros(3, 1, CV_64FC1);
      Mat tvecZeros = Mat::zeros(3, 1, CV_64FC1);
      vector<Point2f> imagePoints;
      //projectPoints(Mat(testPointCloud), rvecZeros, tvecZeros, centerCameraMatrix, centerDistCoeffs, imagePoints);


      Mat rvec, tvec;
      getRvecTvec(allExtrinsicsRt[1] * centerExtrinsicsRt.inv(DECOMP_SVD), rvec, tvec);
      projectPoints(Mat(testPointCloud), rvec, tvec, allCameraMatrices[1], allDistCoeffs[1], imagePoints);

//      getRvecTvec(centerExtrinsicsRt.inv(DECOMP_SVD), rvec, tvec);
//      projectPoints(Mat(testPointCloud), rvec, tvec, allCameraMatrices[0], allDistCoeffs[0], imagePoints);
      Mat drawImage;
      //cvtColor(centerMask, drawImage, CV_GRAY2BGR);
      cvtColor(images[1], drawImage, CV_GRAY2BGR);
      for(size_t i = 0; i < imagePoints.size(); ++i)
      {
        Point pt = imagePoints[i];
        if(pt.x < 0 || pt.x >= drawImage.cols || pt.y < 0 || pt.y >= drawImage.rows)
          continue;



        //circle(drawImage, imagePoints[i], 3, Scalar(255, 0, 255), -1);
        drawImage.at<Vec3b>(pt) = Vec3b(255, 0, 255);
      }

      imshow("kps", drawImage);
      waitKey();
    }
*/
#endif

    PoseRT model2test_ground;
    dataImporter.importGroundTruth(testImageIdx, model2test_ground);
//    cout << "Ground truth: " << model2test_ground << endl;

    pcl::PointCloud<pcl::PointXYZ> testPointCloud;
    dataImporter.importPointCloud(testImageIdx, testPointCloud);

#ifdef VISUALIZE_POSE_REFINEMENT
    if(!kinectCameraFilename.empty())
    {
//      displayEdgels(glassMask, edgeModels[0].points, model2test_ground, kinectCamera, "kinect");
      displayEdgels(kinectBgrImage, edgeModels[0].points, model2test_ground, kinectCamera, "ground truth");
      displayEdgels(kinectBgrImage, edgeModels[0].stableEdgels, model2test_ground, kinectCamera, "ground truth surface");
    }
//    publishPoints(edgeModels[0].points, model2test_ground.getRvec(), model2test_ground.getTvec(), pt_pub, 1, Scalar(0, 0, 255), kinectCamera.extrinsics.getProjectiveMatrix());
    namedWindow("ground truth");
    waitKey();
    destroyWindow("ground truth");
#endif

#ifdef WRITE_RESULTS
    if(testIdx == 0)
    {
      vector<Mat> groundTruthImages = displayEdgels(images, edgeModel.points, rvec_model2test_ground, tvec_model2test_ground, allCameraMatrices, allDistCoeffs, allExtrinsicsRt);;
      writeImages(groundTruthImages, errorsVisualizationPath, 0, 0, 0, "ground");
    }
#endif

    vector<PoseRT> poses_cam;
    vector<float> posesQualities;
    vector<string> detectedObjectsNames;

    TickMeter recognitionTime;
    recognitionTime.start();

    try
    {
      detector.detect(kinectBgrImage, kinectDepth, registrationMask, testPointCloud, poses_cam, posesQualities, detectedObjectsNames);
    }
    catch(const cv::Exception &)
    {
      ++segmentationFailuresCount;
      continue;
    }

    recognitionTime.stop();

    CV_Assert(poses_cam.size() == 1);

    int detectedObjectIndex = detector.getObjectIndex(detectedObjectsNames[0]);
    indicesOfRecognizedObjects.push_back(detectedObjectIndex);

    cout << "Recognized object: " << detectedObjectsNames[0] << endl;
    cout << "Recognition time: " << recognitionTime.getTimeSec() << "s" << endl;

    if (objectNames.size() == 1)
    {
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
//        publishPoints(edgeModels[objectIndex].points, initPoses_cam[objectIndex][i].rvec, initPoses_cam[objectIndex][i].tvec, pt_pub, 1, Scalar(0, 0, 255), kinectCamera.extrinsics.getProjectiveMatrix());
        displayEdgels(kinectBgrImage, edgeModels[objectIndex].points, poses_cam[i], kinectCamera, "final");
        displayEdgels(kinectBgrImage, edgeModels[objectIndex].stableEdgels, poses_cam[i], kinectCamera, "final surface");
        namedWindow("initial pose");
        waitKey();
        destroyWindow("initial pose");
  #endif
      }
      vector<PoseError>::iterator bestPoseIt = std::min_element(currentPoseErrors.begin(), currentPoseErrors.end());
      int bestPoseIdx = std::distance(currentPoseErrors.begin(), bestPoseIt);
      cout << "Best pose: " << currentPoseErrors[bestPoseIdx] << endl;
      bestPoses.push_back(currentPoseErrors[bestPoseIdx]);
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

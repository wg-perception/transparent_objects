/*
 * transparentExperiments.cpp
 *
 *  Created on: Aug 11, 2011
 *      Author: Ilya Lysenkov
 */

#include <opencv2/opencv.hpp>
#include <edges_pose_refiner/edgeModel.hpp>
#include <edges_pose_refiner/poseRT.hpp>
#include <edges_pose_refiner/pinholeCamera.hpp>
#include <fstream>
//#include <visualization_msgs/Marker.h>
//#include <posest/pnp_ransac.h>
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"
#include <numeric>


#include "TODBaseImporter.hpp"
#include "edges_pose_refiner/glassDetector.hpp"
#include "edges_pose_refiner/poseError.hpp"
#include <iomanip>

#include "edges_pose_refiner/pclProcessing.hpp"
#include <pcl_visualization/cloud_viewer.h>

#include "edges_pose_refiner/poseEstimator.hpp"

#ifdef USE_STEREO
#include <edges_pose_refiner/edgesPoseRefiner.hpp>
#endif

using namespace cv;
using std::cout;
using std::endl;
using std::stringstream;

//#define USE_STEREO
//#define VISUALIZE_POSE_REFINEMENT
//#define VISUALIZE_INITIAL_POSE_REFINEMENT
//#define WRITE_RESULTS
//#define PROFILE
//#define WRITE_GLASS_SEGMENTATION



#ifdef USE_STEREO
void getUpsideDownPose(const EdgeModel &edgeModel, const Mat &rvec_cam, const Mat &tvec_cam, Mat &usRvec_cam, Mat &usTvec_cam)
{
  EdgeModel rotatedEdgeModel;
  edgeModel.rotate_cam(rvec_cam, tvec_cam, rotatedEdgeModel);
  Point3d rotAxis = rotatedEdgeModel.rotationAxis;
  Point3d objectCenter = rotatedEdgeModel.getObjectCenter();
  Point3d rvecDir = objectCenter.cross(rotAxis);

  Mat objectCenterMat;
  point2col(objectCenter, objectCenterMat);

  Mat rvecUpsideDown_cam;
  point2col(rvecDir, rvecUpsideDown_cam);
  rvecUpsideDown_cam *= CV_PI / norm(rvecUpsideDown_cam);
  Mat tvecUpsideDown_cam = 2*objectCenterMat;

  composeRT(rvec_cam, tvec_cam, rvecUpsideDown_cam, tvecUpsideDown_cam, usRvec_cam, usTvec_cam);

//        namedWindow("upside down");
//        displayEdgels(images, edgeModel.points, rvecFullUpsideDown_cam, tvecFullUpsideDown_cam, allCameraMatrices, allDistCoeffs, allExtrinsicsRt);
//        waitKey();
}
#endif

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
  CV_Assert(fabs(norm(groundModel.rotationAxis) - 1.0) < eps);
  CV_Assert(fabs(norm(estimatedModel.rotationAxis) - 1.0) < eps);

  double hartleyDiff, tvecDiff;
  Point3d tvecPoint = groundModel.getObjectCenter() - estimatedModel.getObjectCenter();
  tvecDiff = norm(tvecPoint);
  hartleyDiff = acos(groundModel.rotationAxis.ddot(estimatedModel.rotationAxis));

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

void loadStereoData(double testCannyThresh1, double testCannyThresh2, const string &testFolder, int testImageIdx, const vector<bool> &camerasMask, vector<cv::Mat> &testImages, vector<cv::Mat> &testEdges)
{
  testImages.clear();
  testEdges.clear();
  for(size_t i=0; i<camerasMask.size(); i++)
  {
    if(camerasMask[i])
    {
      stringstream imageFilename;
      imageFilename << testFolder << "/image_" << std::setfill('0') << std::setw(5) << testImageIdx << "_" << i << ".jpg";

      Mat img = imread(imageFilename.str(), CV_LOAD_IMAGE_GRAYSCALE);
      CV_Assert(!img.empty());
      testImages.push_back(img);

      Mat edges;
      Canny(img, edges, testCannyThresh1, testCannyThresh2);
      string edgesFilename = imageFilename.str() + "_edges.png";
//        cout << "edges: " << edgesFilename << endl;
//        imwrite(edgesFilename, edges);
//        edges = imread(edgesFilename, CV_LOAD_IMAGE_GRAYSCALE);
//        CV_Assert(!edges.empty());


//        stringstream depthFilename;
//        depthFilename << testFolder << "/depth_image_" << std::setfill('0') << std::setw(5) << testImageIdx << ".xml.gz";
//        FileStorage depthFS(depthFilename.str(), FileStorage::READ);
//        CV_Assert(depthFS.isOpened());
//        Mat depthImage;
//        depthFS["depth_image"] >> depthImage;
//        depthFS.release();
//
//        Mat validDepth = (depthImage != 0);
//        Mat dt;
//        distanceTransform(validDepth, dt, CV_DIST_L2, CV_DIST_MASK_PRECISE);
//
//        const float maxDistToInvalidDepth = 5.0f;
//        edges = edges & (dt < maxDistToInvalidDepth);

      testEdges.push_back(edges);

#ifdef VISUALIZE_POSE_REFINEMENT
      stringstream title;
      title << "edges " << i;
      imshow(title.str(), edges);
#endif
    }
  }
}

void segmentGlass(const Mat &depth, const Mat &kinectBgrImage, Mat &glassMask, int &numberOfComponents)
{
  //findGlassMask(centralBgrImage, depthMat, numberOfComponents, glassMask);
  findGlassMask(kinectBgrImage, depth, numberOfComponents, glassMask, 8, 15, 8);
#ifdef VISUALIZE_POSE_REFINEMENT
  if(numberOfComponents > 0)
  {
    Mat centralImage = kinectBgrImage.clone();
    Mat shadowedCentralImage = centralImage * 0.3;
    shadowedCentralImage.copyTo(centralImage, ~glassMask);

    Mat maskClone = glassMask.clone();
    vector<vector<Point> > contours;
    findContours(maskClone, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
//        drawContours(centralImage, contours, -1, Scalar(0, 255, 0), 1);
    imshow("glass segmentation", centralImage);
    //waitKey();
  }
#endif

#ifdef WRITE_GLASS_SEGMENTATION
  {
    Mat segmentation = drawSegmentation(centralBgrImage, glassMask);
    std::stringstream segmentationFilename;
    segmentationFilename << "segmentation/" + objectName + "/" << testImageIdx << ".png";
    imwrite(segmentationFilename.str(), segmentation);
  }
#endif

#ifdef VISUALIZE_POSE_REFINEMENT
/*
  imshow("glass mask", glassMask);
  imshow("center mask", centerMask);

  Mat maskClone = glassMask.clone();
  vector<vector<Point> > contours;
  findContours(maskClone, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
  drawContours(centralBgrImage, contours, -1, Scalar(0, 255, 0), 3);
  imshow("glass segmentation", centralBgrImage);
*/
#endif
}

int main(int argc, char **argv)
{
  std::system("date");

  CV_Assert(argc == 3);
  string baseFolder = argv[1];
  string testObjectName = argv[2];

  //const string modelFilename = "finalModels/" + objectName + ".xml";
  //const string modelsPath = "centralizedModels/";
  const string modelsPath = "lastModels/";
  //const string modelsPath = "finalModels/";
  const string trainFolder ="/media/2Tb/base_with_ground_truth/base/wh_" + testObjectName + "/";
  //const string trainFolder ="/media/2Tb/fullKineos/trainBase/wh_" + objectName + "/";
  const string testFolder = baseFolder + "/" + testObjectName + "/";
  const string camerasListFilename = baseFolder + "/cameras.txt";
  const string kinectCameraFilename = baseFolder + "/center.yml";
  const string visualizationPath = "visualized_results/";
  //const string errorsVisualizationPath = "errors/" + objectName;
  //const vector<string> objectNames = {"bank", "bucket"};
  //const vector<string> objectNames = {"bank", "bottle", "bucket", "glass", "wineglass"};

  const vector<string> objectNames = {testObjectName};


#ifdef VISUALIZE_POSE_REFINEMENT
  ros::init(argc, argv, "transparent4");
  ros::NodeHandle nh("~");
  ros::Publisher pt_pub = nh.advertise<visualization_msgs::Marker> ("pose_points", 0);
  ros::Publisher *publisher = &pt_pub;
#else
  ros::Publisher *publisher = 0;
#endif

  TODBaseImporter dataImporter(trainFolder, testFolder, publisher);

  vector<PinholeCamera> allCameras;
  vector<bool> camerasMask;

#ifdef USE_STEREO
  {
    try
    {
      dataImporter.readMultiCameraParams(camerasListFilename, allCameras, camerasMask);
    }
    catch(Exception &e)
    {
      allCameras.resize(1);
      dataImporter.readCameraParams(testFolder, allCameras[0], true);
      camerasMask.push_back(true);
    }
  }
#endif

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
  }

//  EdgeModel edgeModel;
//  dataImporter.importEdgeModel(modelsPath, testObjectName, edgeModel);

#ifdef VISUALIZE_POSE_REFINEMENT
  publishPoints(edgeModels[0].points, pt_pub, 0, Scalar(255, 255, 0));
  publishPoints(edgeModels[0].stableEdgels, pt_pub, 1, Scalar(0, 0, 0));

  Scalar meanVal = mean(Mat(edgeModels[0].points));
  Point3d center(meanVal[0], meanVal[1], meanVal[2]);

  drawAxis(center, Point3d(0, 0, 1), pt_pub, 40, Scalar(255, 0, 255));

  publishPoints(vector<Point3f>(1, edgeModels[0].tableAnchor), pt_pub, 11203, Scalar(0, 255, 0));
  publishPoints(vector<Point3f>(1, edgeModels[0].tableAnchor + 0.1 * edgeModels[0].rotationAxis), pt_pub, 11205, Scalar(0, 0, 0));
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


  cout << "Rt: " << endl;
  cout << edgeModels[0].Rt_obj2cam << endl;

  //TODO: remove
  edgeModels[0].Rt_obj2cam = Mat::eye(4, 4, CV_64FC1);

  Ptr<const PinholeCamera> centralCameraPtr = new PinholeCamera(kinectCamera);
  vector<vector<Silhouette> > silhouettes(objectNames.size());
  //int silhouetteCount = 100;
  const int silhouetteCount = 10;
  float downFactor = 1.0f;
  int closingIterationsCount = 10;
  for (size_t i = 0; i < objectNames.size(); ++i)
  {
    edgeModels[i].generateSilhouettes(centralCameraPtr, silhouetteCount, silhouettes[i], downFactor, closingIterationsCount);
  }

  vector<PoseEstimator> poseEstimators;
  for (size_t i = 0; i < objectNames.size(); ++i)
  {
    PoseEstimator estimator(kinectCamera);
    estimator.addObject(edgeModels[i]);
    poseEstimators.push_back(estimator);
  }



  vector<int> testIndices;
  dataImporter.importTestIndices(testIndices);

  vector<size_t> initialPoseCount;
  vector<PoseError> bestPoses;
  int segmentationFailuresCount = 0;
  int badSegmentationCount = 0;

  vector<int> indicesOfRecognizedObjects;
  for(size_t testIdx = 0; testIdx < testIndices.size(); testIdx++)
  {
    int testImageIdx = testIndices[ testIdx ];
    cout << "Test: " << testIdx << " " << testImageIdx << endl;

    TickMeter recognitionTime;
    recognitionTime.start();

    Mat glassMask, kinectBgrImage;
    if(!kinectCameraFilename.empty())
    {
      dataImporter.importBGRImage(testImageIdx, kinectBgrImage);
      Mat depth;
      dataImporter.importDepth(testImageIdx, depth);

      int numberOfComponents;
      segmentGlass(depth, kinectBgrImage, glassMask, numberOfComponents);

      if (numberOfComponents == 0)
      {
        ++segmentationFailuresCount;
        continue;
      }

      if (numberOfComponents != 1)
      {
        ++badSegmentationCount;
        cout << "Wrong number of components: " << numberOfComponents << endl;
#ifdef VISUALIZE_POSE_REFINEMENT
        imshow("depth mat", depth);
        waitKey();
    //        exit(-1);
#endif
      }
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
      cout << "sz: " <<testPointCloud.size() << endl;
      publishPoints(testPointCloud, pt_pub, 222, Scalar(0, 255, 0));
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


    recognitionTime.stop();
    PoseRT model2test_ground;
    dataImporter.importGroundTruth(testImageIdx, model2test_ground);
//    cout << "Ground truth: " << model2test_ground << endl;

    pcl::PointCloud<pcl::PointXYZ> testPointCloud;
    dataImporter.importPointCloud(testImageIdx, testPointCloud);

    vector<Mat> images, allTestEdges;
#ifdef USE_STEREO
    const double testCannyThresh1 = 100.0;
    const double testCannyThresh2 = 50.0;
    loadStereoData(testCannyThresh1, testCannyThresh2, testFolder, testImageIdx, camerasMask, images, allTestEdges);
#endif

#ifdef WRITE_RESULTS
    if(testIdx == 0)
    {
      vector<Mat> groundTruthImages = displayEdgels(images, edgeModel.points, rvec_model2test_ground, tvec_model2test_ground, allCameraMatrices, allDistCoeffs, allExtrinsicsRt);;
      writeImages(groundTruthImages, errorsVisualizationPath, 0, 0, 0, "ground");
    }
#endif

#ifdef VISUALIZE_POSE_REFINEMENT
      displayEdgels(images, edgeModels[0].points, model2test_ground, allCameras);
      if(!kinectCameraFilename.empty())
      {
        displayEdgels(glassMask, edgeModels[0].points, model2test_ground, kinectCamera, "kinect");
        displayEdgels(kinectBgrImage, edgeModels[0].points, model2test_ground, kinectCamera, "ground truth");
        displayEdgels(kinectBgrImage, edgeModels[0].stableEdgels, model2test_ground, kinectCamera, "ground truth surface");
      }
      publishPoints(edgeModels[0].points, model2test_ground.getRvec(), model2test_ground.getTvec(), pt_pub, 1, Scalar(0, 0, 255), kinectCamera.extrinsics.getProjectiveMatrix());
      namedWindow("ground truth");
      waitKey();
      destroyWindow("ground truth");
#endif


      recognitionTime.start();

      vector<vector<PoseRT> > initPoses_cam(objectNames.size());
      vector<vector<float> > initPosesQualities(objectNames.size());
      float bestPoseQuality = std::numeric_limits<float>::max();
      int bestObjectIndex = -1;
      for (size_t objectIndex = 0; objectIndex < objectNames.size(); ++objectIndex)
      {
        poseEstimators[objectIndex].estimatePose(kinectBgrImage, glassMask, testPointCloud, initPoses_cam[objectIndex], initPosesQualities[objectIndex]);

        float currentBestQuality = *std::min_element(initPosesQualities[objectIndex].begin(), initPosesQualities[objectIndex].end());
        if (currentBestQuality < bestPoseQuality)
        {
          bestPoseQuality = currentBestQuality;
          bestObjectIndex = static_cast<int>(objectIndex);
        }
      }
      recognitionTime.stop();
      if (bestObjectIndex < 0)
      {
        bestObjectIndex = rand() % objectNames.size();
      }

      indicesOfRecognizedObjects.push_back(bestObjectIndex);
      cout << "Recognized object: " << objectNames[bestObjectIndex] << endl;
      cout << "Recognition time: " << recognitionTime.getTimeSec() << "s" << endl;





      if (objectNames.size() == 1)
      {
        int objectIndex = 0;
        initialPoseCount.push_back(initPoses_cam[objectIndex].size());

        vector<PoseError> currentPoseErrors(initPoses_cam[objectIndex].size());
        for (size_t i = 0 ; i < initPoses_cam[objectIndex].size(); ++i)
        {
          evaluatePoseWithRotation(edgeModels[objectIndex], initPoses_cam[objectIndex][i], model2test_ground, currentPoseErrors[i]);
          cout << currentPoseErrors[i] << endl;

#ifdef VISUALIZE_POSE_REFINEMENT
          namedWindow("pose is ready");
          waitKey();
          destroyWindow("pose is ready");
#endif

    #ifdef VISUALIZE_POSE_REFINEMENT
          displayEdgels(glassMask, edgeModels[objectIndex].points, initPoses_cam[objectIndex][i], kinectCamera, "initial");
          publishPoints(edgeModels[objectIndex].points, initPoses_cam[objectIndex][i].rvec, initPoses_cam[objectIndex][i].tvec, pt_pub, 1, Scalar(0, 0, 255), kinectCamera.extrinsics.getProjectiveMatrix());
          displayEdgels(kinectBgrImage, edgeModels[objectIndex].points, initPoses_cam[objectIndex][i], kinectCamera, "final");
          displayEdgels(kinectBgrImage, edgeModels[objectIndex].stableEdgels, initPoses_cam[objectIndex][i], kinectCamera, "final surface");
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


#ifdef USE_STEREO
    vector<PoseRT> refinedPoses_cam(initPoses_cam.size());
    vector<double> refinedCosts(initPoses_cam.size());

    cout << "refining " << initPoses_cam.size() << " initial poses..." << endl;
    TickMeter optimizationTime;
    optimizationTime.start();


//#pragma omp parallel for
    for (size_t initPoseIdx = 0; initPoseIdx < initPoses_cam.size(); ++initPoseIdx)
    {
      PoseRT finalPose_cam = initPoses_cam[initPoseIdx];

#ifdef VISUALIZE_POSE_REFINEMENT
        displayEdgels(images, edgeModel.points, finalPose_cam, allCameras);
        if(!kinectCameraFilename.empty())
        {
          displayEdgels(glassMask, edgeModel.points, finalPose_cam, kinectCamera, "kinect");
        }

        publishPoints(edgeModel.points, finalPose_cam.rvec, finalPose_cam.tvec, pt_pub, 1, Scalar(0, 0, 255));
        namedWindow("init");
        waitKey();
        destroyWindow("init");
//          Mat rvecZeros(3, 1, CV_64FC1, Scalar(0));
//          Mat tvecZeros(3, 1, CV_64FC1, Scalar(0));
//          displayEdgels(testImage, vector<Point2f>(), edgeModel.points, rvecZeros, tvecZeros, cameraMatrix, distCoeffs);
#endif

      EdgesPoseRefinerParams params;

      const double cm = 0.01;
      params.maxTranslations = {8*cm, 4*cm, 2*cm, 2*cm};
      params.maxRotationAngles = {CV_PI, CV_PI, CV_PI, CV_PI / 4.0};

      params.localParams.useViewDependentEdges = true;
      params.localParams.useOrientedChamferMatching = false;

      cout << "src. normals: " << edgeModel.normals.size() << endl;
      EdgesPoseRefiner poseRefiner(edgeModel, allCameras, params);
      poseRefiner.setCenterMask(kinectCamera, glassMask);
      double straightCost = poseRefiner.refine(allTestEdges, finalPose_cam.rvec, finalPose_cam.tvec, true);

//        namedWindow("refined");
//        displayEdgels(images, edgeModel.points, rvecFinal_cam, tvecFinal_cam, allCameraMatrices, allDistCoeffs, allExtrinsicsRt);
//        waitKey();

      //start refinement with upside-down pose
      PoseRT upsideDownPose_cam;
      getUpsideDownPose(edgeModel, finalPose_cam.rvec, finalPose_cam.tvec, upsideDownPose_cam.rvec, upsideDownPose_cam.tvec);

      params.maxTranslations = {4*cm, 2*cm,};
      params.maxRotationAngles = {CV_PI / 4.0, CV_PI / 4.0};
      poseRefiner.setParams(params);

      double upsideDownCost = poseRefiner.refine(allTestEdges, upsideDownPose_cam.rvec, upsideDownPose_cam.tvec, true);
      double cost = straightCost;
      if(upsideDownCost < straightCost)
      {
        finalPose_cam = upsideDownPose_cam;
        cost = upsideDownCost;
      }

      refinedCosts[initPoseIdx] = cost;
      refinedPoses_cam[initPoseIdx] = finalPose_cam;
    }

    std::vector<double>::iterator minCostIt = std::min_element(refinedCosts.begin(), refinedCosts.end());
    size_t minCostIdx = std::distance(refinedCosts.begin(), minCostIt);

    PoseRT final_cam = refinedPoses_cam[minCostIdx];

    optimizationTime.stop();

    cout << "Final pose: " << final_cam << endl;

    evaluatePoseWithRotation(edgeModel, initPoses_cam[minCostIdx], model2test_ground, "", 0, Mat());
    bool isRefinementSuccessful = evaluatePoseWithRotation(edgeModel, final_cam, model2test_ground, "global_", 0, Mat());

    isConverged[testIdx] = isRefinementSuccessful;

    stringstream diffFilename;
    diffFilename << "diffs/" << objectName << "/" << testImageIdx << ".xml" << endl;
    evaluatePoseWithRotation(edgeModel, final_cam, model2test_ground, "", 0, Mat(), diffFilename.str());

#ifdef WRITE_RESULTS
    Mat initImage = displayEdgels(glassMask, edgeModel.points, initRvecs_cam[minCostIdx], initTvecs_cam[minCostIdx], centerCameraMatrix, centerDistCoeffs, centerExtrinsicsRt, "kinect");
    writeImages(vector<Mat>(1, initImage), errorsVisualizationPath, testImageIdx, translationIdx, rotationIdx, "kinect_init_");
#endif


#ifdef WRITE_RESULTS
    if(!isRefinementSuccessful)
    {
      vector<Mat> initialImages = displayEdgels(images, edgeModel.points, initRvecs_cam[minCostIdx], initTvecs_cam[minCostIdx], allCameraMatrices, allDistCoeffs, allExtrinsicsRt);
      vector<Mat> stableImages = displayEdgels(images, edgeModel.stableEdgels, rvecGlobal_cam, tvecGlobal_cam, allCameraMatrices, allDistCoeffs, allExtrinsicsRt);
      vector<Mat> viewDependentEdges = displayEdgels(images, edgeModel.points, rvecGlobal_cam, tvecGlobal_cam, allCameraMatrices, allDistCoeffs, allExtrinsicsRt);

      writeImages(initialImages, errorsVisualizationPath, testImageIdx, translationIdx, rotationIdx, "initial");
      writeImages(stableImages, errorsVisualizationPath, testImageIdx, translationIdx, rotationIdx, "stable");
      writeImages(viewDependentEdges, errorsVisualizationPath, testImageIdx, translationIdx, rotationIdx, "viewDependent");
    }
#endif

#ifdef VISUALIZE_POSE_REFINEMENT
    namedWindow("Finished!");
    waitKey();

    displayEdgels(images, edgeModel.stableEdgels, final_cam, allCameras);
    namedWindow("stable");
    waitKey();
    destroyWindow("stable");

    displayEdgels(images, edgeModel.points, final_cam, allCameras);
    if(!kinectCameraFilename.empty())
    {
      displayEdgels(glassMask, edgeModel.points, final_cam, kinectCamera, "kinect");
    }

    publishPoints(edgeModel.points, final_cam.getRvec(), final_cam.getTvec(), pt_pub, 1, Scalar(0, 0, 255));
    waitKey();
    destroyWindow("final");
#endif

    cout << "optimiaztion Time = " << optimizationTime.getTimeSec() << "s" << endl;
#endif

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

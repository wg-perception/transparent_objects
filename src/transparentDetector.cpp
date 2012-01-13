/*
 * transparentDetector.cpp
 *
 *  Created on: Dec 13, 2011
 *      Author: ilysenkov
 */

#include "edges_pose_refiner/transparentDetector.hpp"
#include "edges_pose_refiner/utils.hpp"
#include "edges_pose_refiner/glassDetector.hpp"
#include "edges_pose_refiner/pclProcessing.hpp"

#ifdef USE_3D_VISUALIZATION
#include <boost/thread/thread.hpp>
#endif

#include <opencv2/opencv.hpp>

//#define VISUALIZE_DETECTION

using namespace cv;
using std::cout;
using std::endl;

TransparentDetector::TransparentDetector(const PinholeCamera &_camera, const TransparentDetectorParams &_params)
{
  initialize(_camera, _params);
}

void TransparentDetector::initialize(const PinholeCamera &_camera, const TransparentDetectorParams &_params)
{
  srcCamera = _camera;
  params = _params;
}

void TransparentDetector::addPoints(const std::string &name, const std::vector<cv::Point3f> &points, bool isModelUpsideDown, bool centralize)
{
  EdgeModel edgeModel(points, isModelUpsideDown, centralize);
  addModel(name, edgeModel);
}

void TransparentDetector::addModel(const std::string &name, const EdgeModel &edgeModel)
{
  PoseEstimator estimator(srcCamera);
  estimator.setModel(edgeModel);

  addObject(name, estimator);
}

void TransparentDetector::addObject(const std::string &name, const PoseEstimator &estimator)
{
  if (poseEstimators.empty())
  {
    validTestImageSize = estimator.getValitTestImageSize();
  }
  else
  {
    CV_Assert(validTestImageSize == estimator.getValitTestImageSize());
  }

  poseEstimators.push_back(estimator);
  objectNames.push_back(name);
}

/*
void TransparentDetector::detect(const cv::Mat &srcBgrImage, const cv::Mat &srcDepth, const cv::Mat &srcRegistrationMask, const pcl::PointCloud<pcl::PointXYZ> &sceneCloud, std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities, std::vector<std::string> &detectedObjectNames) const
{
  detect(srcBgrImage, srcDepth, srcRegistrationMask, sceneCloud, poses_cam, posesQualities, objectNames);
}
*/

cv::Mat TransparentDetector::detect(const cv::Mat &srcBgrImage, const cv::Mat &srcDepth, const cv::Mat &srcRegistrationMask, const pcl::PointCloud<pcl::PointXYZ> &sceneCloud, std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities, std::vector<std::string> &detectedObjectNames) const
{
  CV_Assert(srcBgrImage.size() == srcDepth.size());
  CV_Assert(srcRegistrationMask.size() == srcDepth.size());
  PinholeCamera validTestCamera = srcCamera;
  if (validTestCamera.imageSize != validTestImageSize)
  {
    validTestCamera.resize(validTestImageSize);
  }

  Mat bgrImage, depth, registrationMask;
  if (bgrImage.size() != validTestImageSize)
  {
    resize(srcBgrImage, bgrImage, validTestImageSize);
    resize(srcDepth, depth, validTestImageSize);
    resize(srcRegistrationMask, registrationMask, validTestImageSize);
  }
  else
  {
    bgrImage = srcBgrImage;
    depth = srcDepth;
    registrationMask = srcRegistrationMask;
  }

  CV_Assert(bgrImage.size() == validTestImageSize);
  CV_Assert(depth.size() == validTestImageSize);
  CV_Assert(registrationMask.size() == validTestImageSize);

#ifdef VISUALIZE_DETECTION
  cv::imshow("bgrImage", bgrImage);
  cv::imshow("depth", depth);
  cv::waitKey(100);
#endif

  cv::Vec4f tablePlane;
  pcl::PointCloud<pcl::PointXYZ> tableHull;
  bool isEstimated = computeTableOrientation(params.downLeafSize, params.kSearch, params.distanceThreshold, sceneCloud, tablePlane, &tableHull, params.clusterTolerance, params.verticalDirection);
//  bool isEstimated = tmpComputeTableOrientation(validTestCamera, bgrImage, tablePlane);
  if (!isEstimated)
  {
    std::cerr << "Cannot find a table plane" << std::endl;
    return cv::Mat();
  }
  else
  {
    std::cout << "table plane is estimated" << std::endl;
  }

  int numberOfComponents;
  cv::Mat glassMask;
  GlassSegmentator glassSegmentator(params.glassSegmentationParams);
  glassSegmentator.segment(bgrImage, depth, registrationMask, numberOfComponents, glassMask, &validTestCamera, &tablePlane, &tableHull);
//  glassSegmentator.segment(bgrImage, depth, registrationMask, numberOfComponents, glassMask);
  std::cout << "glass is segmented" << std::endl;
  if (numberOfComponents == 0)
  {
    CV_Error(CV_StsOk, "Cannot segment a transparent object");
  }

#ifdef VISUALIZE_DETECTION
  cv::Mat segmentation = drawSegmentation(bgrImage, glassMask);
  cv::imshow("glassMask", glassMask);
  cv::imshow("segmentation", segmentation);
  cv::waitKey(100);
#endif

#ifdef TRANSPARENT_DEBUG
  cv::imwrite("color.png", *color_);
  cv::imwrite("depth.png", *depth_);
  cv::imwrite("glass.png", glassMask);
  cv::FileStorage fs("input.xml", cv::FileStorage::WRITE);
  fs << "K" << *K_;
  fs << "image" << *color_;
  fs << "depth" << *depth_;
  fs << "points3d" << *cloud_;
  fs.release();
#endif


  poses_cam.clear();
  detectedObjectNames.clear();
  posesQualities.clear();
  for (size_t i = 0; i < poseEstimators.size(); ++i)
  {
    std::cout << "starting to estimate pose..." << std::endl;
    std::vector<PoseRT> currentPoses;
    std::vector<float> currentPosesQualities;

    poseEstimators[i].estimatePose(bgrImage, glassMask, currentPoses, currentPosesQualities, &tablePlane);
    std::cout << "done." << std::endl;
    std::cout << "detected poses: " << currentPoses.size() << std::endl;
    if (!currentPoses.empty())
    {
      poses_cam.push_back(currentPoses[0]);
      posesQualities.push_back(currentPosesQualities[0]);
      detectedObjectNames.push_back(objectNames[i]);
    }
  }
  return glassMask;
}

int TransparentDetector::getObjectIndex(const std::string &name) const
{
  std::vector<std::string>::const_iterator it = std::find(objectNames.begin(), objectNames.end(), name);
  CV_Assert(it != objectNames.end());

  return std::distance(objectNames.begin(), it);
}

void TransparentDetector::visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames, cv::Mat &image) const
{
  CV_Assert(poses.size() == objectNames.size());
  if (image.size() != validTestImageSize)
  {
    resize(image, image, validTestImageSize);
  }

  for (size_t i = 0; i < poses.size(); ++i)
  {
    cv::Scalar color(128 + rand() % 128, 128 + rand() % 128, 128 + rand() % 128);
    int objectIndex = getObjectIndex(objectNames[i]);
    poseEstimators[objectIndex].visualize(poses[i], image, color);
  }
}

void TransparentDetector::visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames, pcl::PointCloud<pcl::PointXYZ> &cloud) const
{
#ifdef USE_3D_VISUALIZATION
  CV_Assert(poses.size() == objectNames.size());

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("detected objects"));
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> sceneColor(cloud.makeShared(), 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud.makeShared(), sceneColor, "scene");

  for (size_t i = 0; i < poses.size(); ++i)
  {
    cv::Scalar color(128 + rand() % 128, 128 + rand() % 128, 128 + rand() % 128);
    int objectIndex = getObjectIndex(objectNames[i]);
    poseEstimators[objectIndex].visualize(poses[i], viewer, color, objectNames[i]);
  }

  while (!viewer->wasStopped ())
   {
     viewer->spinOnce (100);
     boost::this_thread::sleep (boost::posix_time::microseconds (100000));
   }
#endif
}

bool TransparentDetector::tmpComputeTableOrientation(const PinholeCamera &camera, const cv::Mat &centralBgrImage, Vec4f &tablePlane) const
{
  Mat blackBlobsObject, whiteBlobsObject, allBlobsObject;
  const string fiducialFilename = "/media/2Tb/transparentBases/fiducial.yml";
  readFiducial(fiducialFilename, blackBlobsObject, whiteBlobsObject, allBlobsObject);

  SimpleBlobDetector::Params params;
  params.filterByInertia = true;
  params.minArea = 10;
  params.minDistBetweenBlobs = 5;

  params.blobColor = 0;
  Ptr<FeatureDetector> blackBlobDetector = new SimpleBlobDetector(params);

  params.blobColor = 255;
  Ptr<FeatureDetector> whiteBlobDetector = new SimpleBlobDetector(params);

  const Size boardSize(4, 11);

  Mat blackBlobs, whiteBlobs;
  bool isBlackFound = findCirclesGrid(centralBgrImage, boardSize, blackBlobs, CALIB_CB_ASYMMETRIC_GRID | CALIB_CB_CLUSTERING, blackBlobDetector);
  bool isWhiteFound = findCirclesGrid(centralBgrImage, boardSize, whiteBlobs, CALIB_CB_ASYMMETRIC_GRID | CALIB_CB_CLUSTERING, whiteBlobDetector);
  if (!isBlackFound && !isWhiteFound)
  {
    cout << isBlackFound << " " << isWhiteFound << endl;
    imshow("can't estimate", centralBgrImage);
    waitKey();
    return false;
  }

  Mat rvec, tvec;
  Mat allBlobs = blackBlobs.clone();
  allBlobs.push_back(whiteBlobs);

  Mat blobs, blobsObject;
  if (isBlackFound && isWhiteFound)
  {
    blobs = allBlobs;
    blobsObject = allBlobsObject;
  }
  else
  {
    if (isBlackFound)
    {
      blobs = blackBlobs;
      blobsObject = blackBlobsObject;
    }
    if (isWhiteFound)
    {
      blobs = whiteBlobs;
      blobsObject = whiteBlobsObject;
    }
  }

  solvePnP(blobsObject, blobs, camera.cameraMatrix, camera.distCoeffs, rvec, tvec);

  PoseRT pose_cam(rvec, tvec);

  Point3d tableAnchor;
  transformPoint(pose_cam.getProjectiveMatrix(), Point3d(0.0, 0.0, 0.0), tableAnchor);

/*
  if (pt_pub != 0)
  {
    vector<Point3f> points;

    vector<Point3f> objectPoints = blackBlobsObject;

    for (size_t i = 0; i < objectPoints.size(); ++i)
    {
      Point3d pt;
      transformPoint(pose_cam.getProjectiveMatrix(), objectPoints[i], pt);
      points.push_back(pt);
      tableAnchor = pt;
    }
    objectPoints = whiteBlobsObject;
    for (size_t i = 0; i < objectPoints.size(); ++i)
    {
      Point3d pt;
      transformPoint(pose_cam.getProjectiveMatrix(), objectPoints[i], pt);
      points.push_back(pt);
    }


    publishPoints(points, *pt_pub, 234, Scalar(255, 0, 255));
  }
*/


  pose_cam.tvec = Mat::zeros(3, 1, CV_64FC1);
  Point3d tableNormal;
  transformPoint(pose_cam.getProjectiveMatrix(), Point3d(0.0, 0.0, -1.0), tableNormal);

  const int dim = 3;
  for (int i = 0; i < dim; ++i)
  {
    tablePlane[i] = Vec3d(tableNormal)[i];
  }
  tablePlane[dim] = -tableNormal.ddot(tableAnchor);

  return true;
}

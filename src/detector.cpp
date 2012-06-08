/*
 * transparentDetector.cpp
 *
 *  Created on: Dec 13, 2011
 *      Author: ilysenkov
 */

#include "edges_pose_refiner/detector.hpp"
#include "edges_pose_refiner/utils.hpp"
#include "edges_pose_refiner/pclProcessing.hpp"

#ifdef USE_3D_VISUALIZATION
#include <boost/thread/thread.hpp>
#endif

#include <opencv2/opencv.hpp>

//#define VISUALIZE_DETECTION

using namespace cv;
using std::cout;
using std::endl;

namespace transpod
{

Detector::Detector(const PinholeCamera &_camera, const DetectorParams &_params)
{
  initialize(_camera, _params);
}

void Detector::initialize(const PinholeCamera &_camera, const DetectorParams &_params)
{
  srcCamera = _camera;
  params = _params;
}

void Detector::addTrainObject(const std::string &objectName, const std::vector<cv::Point3f> &points, bool isModelUpsideDown, bool centralize)
{
  EdgeModel edgeModel(points, isModelUpsideDown, centralize);
  addTrainObject(objectName, edgeModel);
}

void Detector::addTrainObject(const std::string &objectName, const EdgeModel &edgeModel)
{
  PoseEstimator estimator(srcCamera);
  estimator.setModel(edgeModel);

  addTrainObject(objectName, estimator);
}

void Detector::addTrainObject(const std::string &objectName, const PoseEstimator &estimator)
{
  if (poseEstimators.empty())
  {
    validTestImageSize = estimator.getValidTestImageSize();
  }
  else
  {
    CV_Assert(validTestImageSize == estimator.getValidTestImageSize());
  }

  std::pair<std::map<string, PoseEstimator>::iterator, bool> result;
  result = poseEstimators.insert(std::make_pair(objectName, estimator));
  if (!result.second)
  {
    CV_Error(CV_StsBadArg, "Object name '" + objectName + "' is not unique");
  }
}

void Detector::detect(const cv::Mat &srcBgrImage, const cv::Mat &srcDepth, const cv::Mat &srcRegistrationMask, const cv::Mat &sceneCloud, std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities, std::vector<std::string> &detectedObjectNames, Detector::DebugInfo *debugInfo) const
{
  pcl::PointCloud<pcl::PointXYZ> pclCloud;
  cv2pcl(sceneCloud.reshape(3, 1), pclCloud);
  detect(srcBgrImage, srcDepth, srcRegistrationMask, pclCloud, poses_cam, posesQualities, detectedObjectNames, debugInfo);
}

void Detector::detect(const cv::Mat &srcBgrImage, const cv::Mat &srcDepth, const cv::Mat &srcRegistrationMask, const pcl::PointCloud<pcl::PointXYZ> &sceneCloud, std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities, std::vector<std::string> &detectedObjectNames, Detector::DebugInfo *debugInfo) const
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

  if (bgrImage.size() != validTestImageSize)
  {
    std::stringstream error;
    error << "RGB resolution is " << bgrImage.cols << "x" << bgrImage.rows;
    error << ", but valid resolution is " << validTestImageSize.width << "x" << validTestImageSize.height;
    CV_Error(CV_StsBadArg, error.str());
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
  bool isEstimated = computeTableOrientation(params.planeSegmentationParams.downLeafSize,
                       params.planeSegmentationParams.kSearch, params.planeSegmentationParams.distanceThreshold,
                       sceneCloud, tablePlane, &tableHull, params.planeSegmentationParams.clusterTolerance, params.planeSegmentationParams.verticalDirection);
//  bool isEstimated = tmpComputeTableOrientation(validTestCamera, bgrImage, tablePlane);
  if (!isEstimated)
  {
    CV_Error(CV_StsOk, "Cannot find a table plane");
  }

#ifdef VERBOSE
  std::cout << "table plane is estimated" << std::endl;
#endif

  int numberOfComponents;
  cv::Mat glassMask;
  GlassSegmentator glassSegmentator(params.glassSegmentationParams);
  glassSegmentator.segment(bgrImage, depth, registrationMask, numberOfComponents, glassMask, &validTestCamera, &tablePlane, &tableHull);
//  glassSegmentator.segment(bgrImage, depth, registrationMask, numberOfComponents, glassMask);
  if (debugInfo != 0)
  {
    debugInfo->glassMask = glassMask;
  }
#ifdef VERBOSE
  std::cout << "glass is segmented" << std::endl;
#endif
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
  for (std::map<std::string, PoseEstimator>::const_iterator it = poseEstimators.begin(); it != poseEstimators.end(); ++it)
  {
#ifdef VERBOSE
    std::cout << "starting to estimate pose..." << std::endl;
#endif
    std::vector<PoseRT> currentPoses;
    std::vector<float> currentPosesQualities;

    vector<Mat> initialSilhouettes;
    vector<Mat> *initialSilhouettesPtr = debugInfo == 0 ? 0 : &initialSilhouettes;
    it->second.estimatePose(bgrImage, glassMask, currentPoses, currentPosesQualities, &tablePlane, initialSilhouettesPtr);
#ifdef VERBOSE
    std::cout << "done." << std::endl;
    std::cout << "detected poses: " << currentPoses.size() << std::endl;
#endif
    if (debugInfo != 0)
    {
      std::copy(initialSilhouettes.begin(), initialSilhouettes.end(), std::back_inserter(debugInfo->initialSilhouettes));
    }
    if (!currentPoses.empty())
    {
      poses_cam.push_back(currentPoses[0]);
      posesQualities.push_back(currentPosesQualities[0]);
      detectedObjectNames.push_back(it->first);
    }
  }
}

void Detector::visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames, cv::Mat &image) const
{
  CV_Assert(poses.size() == objectNames.size());
  if (image.size() != validTestImageSize)
  {
    resize(image, image, validTestImageSize);
  }

  for (size_t i = 0; i < poses.size(); ++i)
  {
    //TODO: use randomization
    cv::Scalar color(255, 0, 255);
    switch (i)
    {
      case 0:
        color = cv::Scalar(255, 0, 0);
        break;
      case 1:
        color = cv::Scalar(0, 0, 255);
        break;
      case 2:
        color = cv::Scalar(0, 255, 0);
        break;
    }

    poseEstimators.find(objectNames[i])->second.visualize(poses[i], image, color);
  }
}

void Detector::showResults(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames,
                           const cv::Mat &image, const std::string title) const
{
  Mat visualization = image.clone();
  visualize(poses, objectNames, visualization);
  imshow(title, visualization);
}

void Detector::visualize(const std::vector<PoseRT> &poses, const std::vector<std::string> &objectNames, pcl::PointCloud<pcl::PointXYZ> &cloud) const
{
#ifdef USE_3D_VISUALIZATION
  CV_Assert(poses.size() == objectNames.size());

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("detected objects"));
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> sceneColor(cloud.makeShared(), 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(cloud.makeShared(), sceneColor, "scene");

  for (size_t i = 0; i < poses.size(); ++i)
  {
    cv::Scalar color(128 + rand() % 128, 128 + rand() % 128, 128 + rand() % 128);
    poseEstimators.find(objectNames[i])->second.visualize(poses[i], viewer, color, objectNames[i]);
  }

  while (!viewer->wasStopped ())
   {
     viewer->spinOnce (100);
     boost::this_thread::sleep (boost::posix_time::microseconds (100000));
   }
#endif
}

bool Detector::tmpComputeTableOrientation(const PinholeCamera &camera, const cv::Mat &centralBgrImage, Vec4f &tablePlane) const
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

} //end of namespace transpod

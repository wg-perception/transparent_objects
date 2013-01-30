/*
 * tableSegmentation.cpp
 *
 *  Created on: 12/21/2012
 *      Author: ilysenkov
 */

#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <boost/make_shared.hpp>

#include <opencv2/opencv.hpp>

#include "edges_pose_refiner/tableSegmentation.hpp"
#include "edges_pose_refiner/utils.hpp"
#include "edges_pose_refiner/pclProcessing.hpp"
#include "edges_pose_refiner/pcl.hpp"

#define USE_RGBD_MODULE

#ifdef USE_RGBD_MODULE
#include <opencv2/rgbd/rgbd.hpp>
#endif

using namespace cv;
using std::cout;
using std::endl;

bool computeTableOrientationByPCL(float downLeafSize, int kSearch, float distanceThreshold, const std::vector<cv::Point3f> &cvFullSceneCloud,
                                  cv::Vec4f &tablePlane, const PinholeCamera *camera, std::vector<cv::Point2f> *tableHull, float clusterTolerance, cv::Point3f verticalDirection)
{
  CV_Assert(!cvFullSceneCloud.empty());
  pcl::PointCloud<pcl::PointXYZ> fullSceneCloud;
  cv2pcl(cvFullSceneCloud, fullSceneCloud);
#ifdef VERBOSE
  cout << "Estimating table plane...  " << std::flush;
#endif
  pcl::PointCloud<pcl::PointXYZ> withoutNaNsCloud;
  filterNaNs(fullSceneCloud, withoutNaNsCloud);

  pcl::PointCloud<pcl::PointXYZ> sceneCloud;
  downsample(downLeafSize, withoutNaNsCloud, sceneCloud);

  pcl::PointCloud<pcl::Normal> sceneNormals;
  estimateNormals(kSearch, sceneCloud, sceneNormals);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  bool isTableSegmented = segmentTable(distanceThreshold, sceneCloud, sceneNormals, inliers, coefficients);
  if (!isTableSegmented)
  {
    return false;
  }

  const int coeffsCount = 4;
  Point3f tableNormal(coefficients->values[0],
                      coefficients->values[1],
                      coefficients->values[2]);
  if (tableNormal.dot(verticalDirection) < 0)
  {
    for (int i = 0; i < coeffsCount; ++i)
    {
      coefficients->values[i] *= -1;
    }
  }

  for (int i = 0; i < coeffsCount; ++i)
  {
    tablePlane[i] = coefficients->values[i];
  }

  if (camera != 0 && tableHull != 0)
  {
    pcl::PointCloud<pcl::PointXYZ> projectedInliers;
    projectInliersOnTable(sceneCloud, inliers, coefficients, projectedInliers);

//    reconstructConvexHull(projectedInliers, *tableHull);


    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(projectedInliers.makeShared());

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setSearchMethod(tree);
    ec.setInputCloud(projectedInliers.makeShared());
    ec.extract(clusterIndices);

    int maxClusterIndex = 0;
    for (size_t i = 1; i < clusterIndices.size(); ++i)
    {
      if (clusterIndices[maxClusterIndex].indices.size() < clusterIndices[i].indices.size())
      {
        maxClusterIndex = i;
      }
    }

    pcl::PointCloud<pcl::PointXYZ> table;
    extractPointCloud(projectedInliers, boost::make_shared<pcl::PointIndices>(clusterIndices[maxClusterIndex]), table);

    pcl::PointCloud<pcl::PointXYZ> tableHull3D;
    reconstructConvexHull(table, tableHull3D);

    vector<Point3f> cvTableHull3D;
    pcl2cv(tableHull3D, cvTableHull3D);
    camera->projectPoints(cvTableHull3D, PoseRT(), *tableHull);
  }
#ifdef VERBOSE
  cout << "Done." << endl;
#endif


#ifdef VISUALIZE_TABLE_ESTIMATION
  pcl::PointCloud<pcl::PointXYZ> table;
  extractPointCloud(sceneCloud, inliers, table);

  pcl::visualization::CloudViewer viewer ("test cloud");
  viewer.showCloud(sceneCloud.makeShared(), "points");

  while (!viewer.wasStopped ())
  {
  }

  pcl::visualization::CloudViewer viewer2 ("table");
  viewer2.showCloud(table.makeShared(), "table");
  while (!viewer2.wasStopped ())
  {
  }
#endif

  return true;
}

bool computeTableOrientationByRGBD(const Mat &depth, const PinholeCamera &camera,
                                   cv::Vec4f &tablePlane, std::vector<cv::Point> *tableHull,
                                   Point3f verticalDirection)
{
  //TODO: fix compilation with Jenkins
#ifndef USE_RGBD_MODULE
  CV_Assert(false);
#else
  //TODO: move up
  const uchar nonPlanarMark = 255;

  Mat points3d;
  depthTo3d(depth, camera.cameraMatrix, points3d);
  RgbdNormals normalsEstimator(depth.rows, depth.cols, depth.depth(), camera.cameraMatrix);
  Mat normals;
  normalsEstimator(points3d, normals);

  RgbdPlane planeEstimator;
  Mat planesMask;
  vector<Vec4f> planeCoefficients;
  planeEstimator(points3d, normals, planesMask, planeCoefficients);
  CV_Assert(planesMask.type() == CV_8UC1);

  vector<int> pixelCounts(planeCoefficients.size(), 0);
  for (int i = 0; i < planesMask.rows; ++i)
  {
    for (int j = 0; j < planesMask.cols; ++j)
    {
      if (planesMask.at<uchar>(i, j) != nonPlanarMark)
      {
        pixelCounts[planesMask.at<uchar>(i, j)] += 1;
      }
    }
  }
  std::vector<int>::iterator largestPlaneIt = std::max_element(pixelCounts.begin(), pixelCounts.end());
  int largestPlaneIndex = std::distance(pixelCounts.begin(), largestPlaneIt);

  tablePlane = planeCoefficients[largestPlaneIndex];

  Point3f tableNormal(tablePlane[0],
                      tablePlane[1],
                      tablePlane[2]);
  if (tableNormal.dot(verticalDirection) < 0)
  {
    tablePlane *= -1;
  }


  if (tableHull != 0)
  {
    vector<Point> tablePoints;
    for (int i = 0; i < planesMask.rows; ++i)
    {
      for (int j = 0; j < planesMask.cols; ++j)
      {
        if (planesMask.at<uchar>(i, j) == largestPlaneIndex)
        {
          tablePoints.push_back(Point(j, i));
        }
      }
    }
    convexHull(tablePoints, *tableHull);
  }

  return true;
#endif
}


int computeTableOrientationByFiducials(const PinholeCamera &camera, const cv::Mat &centralBgrImage, Vec4f &tablePlane)
{
  Mat blackBlobsObject, whiteBlobsObject, allBlobsObject;
  //TODO: move up parameters
  const string fiducialFilename = "fiducial.yml";
//  const string fiducialFilename = "/media/2Tb/transparentBases/fiducial.yml";
//  const string fiducialFilename = "/u/ilysenkov/transparentBases/base/fiducial.yml";
  readFiducial(fiducialFilename, blackBlobsObject, whiteBlobsObject, allBlobsObject);

  Mat blackBlobs, whiteBlobs;
  detectFiducial(centralBgrImage, blackBlobs, whiteBlobs);
  bool isBlackFound = !blackBlobs.empty();
  bool isWhiteFound = !whiteBlobs.empty();

  if (!isBlackFound && !isWhiteFound)
  {
    cout << isBlackFound << " " << isWhiteFound << endl;
    imshow("can't estimate", centralBgrImage);
    waitKey();
    return 0;
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

  int numberOfPatternsFound = static_cast<int>(isBlackFound) + static_cast<int>(isWhiteFound);
  return numberOfPatternsFound;
}

void drawTable(const std::vector<cv::Point2f> &tableHull, cv::Mat &image,
               cv::Scalar color, int thickness)
{
    if (image.channels() == 1)
    {
        Mat drawImage;
        cvtColor(image, drawImage, CV_GRAY2BGR);
        image = drawImage;
    }
    CV_Assert(image.channels() == 3);

    if (!tableHull.empty())
    {
        Mat tableHull_int;
        Mat(tableHull).convertTo(tableHull_int, CV_32SC2);
        bool isClosed = true;
        polylines(image, tableHull_int, isClosed, color, thickness);
    }
}

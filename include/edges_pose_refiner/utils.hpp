/*
 * utils.hpp
 *
 *  Created on: Apr 23, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "edges_pose_refiner/poseRT.hpp"
#include "edges_pose_refiner/pinholeCamera.hpp"

void createProjectiveMatrix(const cv::Mat &R, const cv::Mat &t, cv::Mat &Rt);
void getRvecTvec(const cv::Mat &projectiveMatrix, cv::Mat &rvec, cv::Mat &tvec);
void getTransformationMatrix(const cv::Mat &R_obj2cam, const cv::Mat &t_obj2cam, const cv::Mat &rvec_Object, const cv::Mat &tvec_Object, cv::Mat &transformationMatrix);
void getTransformationMatrix(const cv::Mat &Rt_obj2cam, const cv::Mat &rvec_Object, const cv::Mat &tvec_Object, cv::Mat &transformationMatrix);

void getRotationTranslation(const cv::Mat &projectiveMatrix, cv::Mat &R, cv::Mat &t);

void publishPoints(const std::vector<cv::Point3f>& points, cv::Scalar color = cv::Scalar(0, 255, 0), const std::string &id = "", const PoseRT &pose = PoseRT());
void publishPoints(const std::vector<std::vector<cv::Point3f> >& points);
//void publishTable(const cv::Vec4f &tablePlane, int id, cv::Scalar color, ros::Publisher *pt_pub = 0);

void writePointCloud(const std::string &filename, const std::vector<cv::Point3f> &pointCloud);
void readPointCloud(const std::string &filename, std::vector<cv::Point3f> &pointCloud, std::vector<cv::Point3f> *normals = 0);
void readPointCloud(const std::string &filename, std::vector<cv::Point3f> &pointCloud, std::vector<cv::Point3i> &colors, std::vector<cv::Point3f> &normals);

void transformPoint(const cv::Mat &Rt, const cv::Point3d &point, cv::Point3d &transformedPoint);

void readLinesInFile(const std::string &filename, std::vector<std::string> &lines);

template<class T>
void point2col(cv::Point3_<T> pt, cv::Mat &mat)
{
  std::vector<cv::Point3_<T> > ptVec(1, pt);
  mat = cv::Mat(ptVec).clone();
  const int dim = 3;
  mat = mat.reshape(1, dim);
}

template<class T>
void point2row(cv::Point3_<T> pt, cv::Mat &mat)
{
  std::vector<cv::Point3_<T> > ptVec(1, pt);
  mat = cv::Mat(ptVec).clone();
  mat = mat.reshape(1, 1);
}

bool isPointInside(const cv::Mat &image, cv::Point pt);

void mask2contour(const cv::Mat &mask, std::vector<cv::Point2f> &contour);

void hcat(const cv::Mat &A, const cv::Mat &B, cv::Mat &result);

void readFiducial(const std::string &filename, cv::Mat &blackBlobsObject, cv::Mat &whiteBlobsObject, cv::Mat &allBlobsObject);
void detectFiducial(const cv::Mat &bgrImage, cv::Mat &blackBlobs, cv::Mat &whiteBlobs);

cv::Mat drawSegmentation(const cv::Mat &image, const cv::Mat &mask, const cv::Scalar &color = cv::Scalar(0, 255, 0), int thickness = 1);

cv::Mat drawEdgels(const cv::Mat &image, const std::vector<cv::Point3f> &edgels3d, const PoseRT &pose_cam, const PinholeCamera &camera,
                   cv::Scalar color = cv::Scalar(0, 0, 255), float blendingFactor = 1.0f);
std::vector<cv::Mat> drawEdgels(const std::vector<cv::Mat> &images, const std::vector<cv::Point3f> &edgels3d,
                                   const PoseRT &pose_cam,
                                   const std::vector<PinholeCamera> &cameras,
                                   cv::Scalar color = cv::Scalar(0, 0, 255), float blendingFactor = 1.0f);

cv::Mat showEdgels(const cv::Mat &image, const std::vector<cv::Point3f> &edgels3d, const PoseRT &pose_cam, const PinholeCamera &camera, const std::string &title = "projected model", cv::Scalar color = cv::Scalar(0, 0, 255));
std::vector<cv::Mat> showEdgels(const std::vector<cv::Mat> &images, const std::vector<cv::Point3f> &edgels3d,
                                const PoseRT &pose_cam,
                                const std::vector<PinholeCamera> &cameras,
                                const std::string &title = "projected model",
                                cv::Scalar color = cv::Scalar(0, 0, 255));

void project3dPoints(const std::vector<cv::Point3f>& points, const cv::Mat& rvec, const cv::Mat& tvec, std::vector<cv::Point3f>& modif_points);
void project3dPoints(const std::vector<cv::Point3f>& points, const PoseRT &pose, std::vector<cv::Point3f>& modif_points);

template<class T>
void drawPoints(const std::vector<cv::Point_<T> > &points, cv::Mat &image, cv::Scalar color = cv::Scalar::all(255), int radius = 1)
{
  CV_Assert(!image.empty());
  if (image.channels() == 1)
  {
    cv::Mat colorImage;
    cv::cvtColor(image, colorImage, CV_GRAY2BGR);
    image = colorImage;
  }

  for (size_t i = 0; i < points.size(); ++i)
  {
    cv::Point pt = points[i];
    if (isPointInside(image, pt))
    {
      cv::circle(image, pt, radius, color, -1);
    }
  }
}

void saveToCache(const std::string &name, const cv::Mat &mat);
cv::Mat getFromCache(const std::string &name);

cv::Mat getInvalidDepthMask(const cv::Mat &depthMat, const cv::Mat &registrationMask);

void computeOrientations(const cv::Mat &edges, cv::Mat &orientationsImage);

template <typename T> int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

void imshow3d(const std::string &windowName, const cv::Mat &image3d);
void cvtColor3d(const cv::Mat &src, cv::Mat &dst, int code);

template <typename T>
T getInterpolatedValue(const cv::Mat &mat, cv::Point2f pt)
{
  int xFloor = cvFloor(pt.x);
  int yFloor = cvFloor(pt.y);
  float x = pt.x - xFloor;
  float y = pt.y - yFloor;
  //bilinear interpolation
  T result = mat.at<T>(yFloor    , xFloor    ) * (1.0 - x) * (1.0 - y) +
             mat.at<T>(yFloor    , xFloor + 1) * x * (1.0 - y) +
             mat.at<T>(yFloor + 1, xFloor    ) * (1.0 - x) * y +
             mat.at<T>(yFloor + 1, xFloor + 1) * x * y;
  return result;
}


void markContourByUser(const cv::Mat &image, std::vector<cv::Point> &contour,
                       const std::string &windowName = "manual contour marking");


#endif /* UTILS_HPP_ */

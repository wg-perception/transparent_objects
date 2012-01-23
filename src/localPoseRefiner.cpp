/*
 * localPoseRefiner.cpp
 *
 *  Created on: Apr 21, 2011
 *      Author: Ilya Lysenkov
 */

#include "edges_pose_refiner/localPoseRefiner.hpp"
#include "edges_pose_refiner/pinholeCamera.hpp"
#include "edges_pose_refiner/poseRT.hpp"
#include <iostream>
#include "edges_pose_refiner/utils.hpp"

#include <opencv2/opencv.hpp>

//#define VISUALIZE_NLOPT

//#define OCM_VISUALIZE
//#define VISUALIZE_CHAMFER

//#define VERBOSE
//#define VISUALIZE

#ifdef USE_ORIENTED_CHAMFER_MATCHING
#include "chamfer_matching/chamfer_matching.h"
#endif

using namespace cv;
using std::cout;
using std::endl;

LocalPoseRefiner::LocalPoseRefiner(const EdgeModel &_edgeModel, const cv::Mat &_edgesImage, const cv::Mat &_cameraMatrix, const cv::Mat &_distCoeffs, const cv::Mat &_extrinsicsRt, const LocalPoseRefinerParams &_params)
{
  dim = 3;
  params = _params;
  edgesImage = _edgesImage.clone();
  CV_Assert(!edgesImage.empty());
  _cameraMatrix.copyTo(cameraMatrix);
  _distCoeffs.copyTo(distCoeffs);
  _extrinsicsRt.copyTo(extrinsicsRt);


  if(params.useOrientedChamferMatching)
  {
    CV_Assert(false);
#ifdef USE_ORIENTED_CHAMFER_MATCHING
    //otherwise chamfer matching doesn't work
    edgesImage.row(0).setTo(Scalar(0));
    edgesImage.row(edgesImage.rows - 1).setTo(Scalar(0));
    edgesImage.col(0).setTo(Scalar(0));
    edgesImage.col(edgesImage.cols - 1).setTo(Scalar(0));

  //  Mat edgesWideImage(edgesImage.rows + 2 * margin, edgesImage.cols + 2*margin, edgesImage.type(), Scalar(0));
  //  Rect roiRect(margin, margin, edgesImage.cols, edgesImage.rows);
  //  Mat roi = edgesWideImage(roiRect);
  //  edgesImage.copyTo(roi);
  //  imshow("narrow", edgesImage);
  //  imshow("wide", edgesWideImage);
  //  waitKey();

    IplImage edge_img = edgesImage;
    IplImage *dist_img = cvCreateImage(cvSize(edge_img.width, edge_img.height), IPL_DEPTH_32F, 1);
    cvSetZero(dist_img);
    IplImage *annotated_img = cvCreateImage(cvSize(edge_img.width, edge_img.height), IPL_DEPTH_32S, 2);
    cvSetZero(annotated_img);

    //this param is not used because we don't use computed dist_img
    const float dtTruncation = -1;
    ::computeDistanceTransform(&edge_img, dist_img, annotated_img, dtTruncation);

    IplImage *orientation_img = cvCreateImage(cvSize(edge_img.width, edge_img.height), IPL_DEPTH_32F, 1);
    cvSetZero(orientation_img);
    IplImage* edge_clone = cvCloneImage(&edge_img);
    computeEdgeOrientations(edge_clone, orientation_img, params.testM);
    cvReleaseImage(&edge_clone);
    fillNonContourOrientations(annotated_img, orientation_img);

    orientationImage = Mat(orientation_img).clone();

  #ifdef OCM_VISUALIZE
    Mat orView = min(orientationImage, CV_PI - orientationImage);

    Mat view;
    orView.convertTo(view, CV_8UC1, 235. / (CV_PI / 2.0), 20);
    view.setTo(Scalar(0), ~edgesImage);
    imshow("orientation", view);
    waitKey();
    cout << "Orientation: " << orientationImage(Rect(0, 0, 10, 10)) << endl;
    cout << "Edges: " << edgesImage(Rect(0, 0, 10, 10)) << endl;

    Mat nonNan = Mat(orientationImage.size(), CV_8UC1, Scalar(255));
    for(int i=0; i<orientationImage.rows; i++)
    {
      for(int j=0; j<orientationImage.cols; j++)
      {
        if(isnan(orientationImage.at<float>(i, j)))
        {
          nonNan.at<uchar>(i, j) = 0;
        }
      }
    }

    imshow("orientation non-nan", nonNan);
    waitKey();
  #endif

    double minOrientation, maxOrientation;
    minMaxLoc(orientationImage, &minOrientation, &maxOrientation);
    cout << "Orientations: " << minOrientation << " " << maxOrientation << endl;

    cvReleaseImage(&annotated_img);
    cvReleaseImage(&dist_img);
    cvReleaseImage(&orientation_img);
#endif
  }

  cameraMatrix.convertTo(cameraMatrix64F, CV_64FC1);


  computeDistanceTransform(edgesImage, dtImage, dtDx, dtDy);

  originalEdgeModel = _edgeModel;
  //TODO: remove copy operation
  rotatedEdgeModel = _edgeModel;

  setObjectCoordinateSystem(originalEdgeModel.Rt_obj2cam);

  centerMask = Mat();
}

void LocalPoseRefiner::setSilhouetteEdges(const cv::Mat &_silhouetteEdges)
{
  silhouetteEdges = _silhouetteEdges;
  computeDistanceTransform(silhouetteEdges, silhouetteDtImage, silhouetteDtDx, silhouetteDtDy);
}

void LocalPoseRefiner::setObjectCoordinateSystem(const cv::Mat &Rt_obj2cam)
{
  Rt_obj2cam_cached = Rt_obj2cam.clone();
  Rt_cam2obj_cached = Rt_obj2cam.inv(DECOMP_SVD);
}

void LocalPoseRefiner::getObjectCoordinateSystem(cv::Mat &Rt_obj2cam) const
{
  Rt_obj2cam_cached.copyTo(Rt_obj2cam);
}

void LocalPoseRefiner::setInitialPose(const PoseRT &pose_cam)
{
  originalEdgeModel.rotate_cam(pose_cam, rotatedEdgeModel);
  setObjectCoordinateSystem(rotatedEdgeModel.Rt_obj2cam);
}

void LocalPoseRefiner::computeDistanceTransform(const cv::Mat &edges, cv::Mat &distanceImage, cv::Mat &dx, cv::Mat &dy)
{
  if(edges.empty())
  {
    CV_Error(CV_HeaderIsNull, "edges are empty");
  }

  distanceTransform(~edges, distanceImage, CV_DIST_L2, CV_DIST_MASK_PRECISE);

  Mat kx_dx, ky_dx;
  int ksize=3;
  getDerivKernels(kx_dx, ky_dx, 1, 0, ksize, true);
  Mat kx_dy, ky_dy;
  getDerivKernels(kx_dy, ky_dy, 0, 1, ksize, true);

  sepFilter2D(distanceImage, dx, CV_32F, kx_dx, ky_dx);
  sepFilter2D(distanceImage, dy, CV_32F, kx_dy, ky_dy);
  //TODO: remove after OpenCV fix
//  dy = -dy;


  assert(dx.size() == distanceImage.size());
  assert(dy.size() == distanceImage.size());

#ifdef VERBOSE
/*
  Mat dxView, dyView;
  Mat absDx = abs(dx);
  Mat absDy = abs(dy);
  absDx.convertTo(dxView, CV_8UC1, 50);
  absDy.convertTo(dyView, CV_8UC1, 50);

  Mat dt;
  double maxVal;
  minMaxLoc(distanceImage, 0, &maxVal);
  distanceImage.convertTo(dt, CV_8UC1, 255. / maxVal);


  Mat scaledDxView, scaledDyView, scaledDT;
  float s = 0.3;
  resize(dxView, scaledDxView, Size(), s, s);
  resize(dyView, scaledDyView, Size(), s, s);
  resize(dt, scaledDT, Size(), s, s);
  imshow("dx", scaledDxView);
  imshow("dy", scaledDyView);
  imshow("dt", scaledDT);
*/
#endif


#ifdef VERBOSE
  //imshow("frame", frame);
  //Mat dt;
  //double maxVal;
  //minMaxLoc(distanceImage, 0, &maxVal);
  //distanceImage.convertTo(dt, CV_8UC1, 255. / maxVal);
  //imshow("distanceImage", distanceImage);
  //imshow("distanceImage", dt);
#endif
}

double getInterpolatedDT(const Mat &dt, Point2f pt)
{
  //if( pt.x < 0 || pt.y < 0 || pt.x+1 >= dt.cols || pt.y+1 >= dt.rows)
    //throw Exception();
    //CV_Error(CV_StsOk, "pixel is out of range");

/*
  //TODO: process boundary pixels
  CV_Assert(dt.type() == CV_32FC1);
  Mat map1 = Mat(pt).reshape(2);
  //std::cout << map1 << std::endl;
  //std::cout << pt << std::endl;
  CV_Assert( map1.type() == CV_32FC2);
  Mat result;
  remap(dt, result, map1, Mat(), CV_INTER_LINEAR);
  CV_Assert(result.rows == 1 && result.cols == 1);
  CV_Assert(result.type() == CV_32FC1);

  //std::cout << result << std::endl;

  //Range rowRange = Range(cvFloor(pt.y), cvFloor(pt.y)+2);
  //Range colRange = Range(cvFloor(pt.x), cvFloor(pt.x)+2);
  //Mat roi = dt(rowRange, colRange);

  //std::cout << "roi: " << roi << std::endl;
  //return result.at<float>(0, 0);
*/

  int xFloor = cvFloor(pt.x);
  int yFloor = cvFloor(pt.y);
  double x = pt.x - xFloor;
  double y = pt.y - yFloor;
  //bilinear interpolation
  double result = dt.at<float>(yFloor, xFloor)*(1. - x)*(1.-y) + dt.at<float>(yFloor, xFloor+1)*x*(1.-y) + dt.at<float>(yFloor+1, xFloor)*(1. - x)*y + dt.at<float>(yFloor+1, xFloor+1)*x*y;
  return result;
}


bool LocalPoseRefiner::isOutlier(cv::Point2f pt) const
{
  if( pt.x < 0 || pt.y < 0 || pt.x+1 >= dtImage.cols || pt.y+1 >= dtImage.rows)
    return true;

  CV_Assert(dtImage.type() == CV_32FC1);
  if(dtImage.at<float>(pt) >params.outlierDistance)
    return true;

  return false;
}

double LocalPoseRefiner::getFilteredDistance(cv::Point2f pt, bool useInterpolation, double inlierMaxDistance, double outlierError, const cv::Mat &distanceTransform) const
{
  Mat dt = distanceTransform.empty() ? dtImage : distanceTransform;

  if( pt.x < 0 || pt.y < 0 || pt.x+1 >= dt.cols || pt.y+1 >= dt.rows)
    return outlierError;

  CV_Assert(dt.type() == CV_32FC1);

  double dist = useInterpolation ? getInterpolatedDT(dt, pt) : dt.at<float>(pt);

  //cout << dist << " vs. " << inlierMaxDistance << endl;
  if(dist > inlierMaxDistance)
    return outlierError;

  return dist;
}

void LocalPoseRefiner::computeResidualsForTrimmedError(cv::Mat &projectedPoints, std::vector<float> &residuals) const
{
  CV_Assert(projectedPoints.cols == 1);
  CV_Assert(projectedPoints.type() == CV_32FC2);
  CV_Assert(dtImage.type() == CV_32FC1);

  //projectedPoints -= origin;
  residuals.reserve(projectedPoints.rows);
  const float outlierError = std::numeric_limits<float>::max();
  for(int i=0; i<projectedPoints.rows; i++)
  {
    //Point2f pt2f = projectedPoints.at<Vec2f>(i, 0);
    //pt2f -= origin;
    Point pt2f = Point2f(projectedPoints.at<Vec2f>(i, 0));

    if(pt2f.x < 0 || pt2f.y < 0 || pt2f.x+1 >= dtImage.cols || pt2f.y+1 >= dtImage.rows)
    {
      residuals.push_back(outlierError);
      continue;
    }
    residuals.push_back(dtImage.at<float>(pt2f));
  }
}

void LocalPoseRefiner::computeResiduals(const cv::Mat &projectedPoints, cv::Mat &residuals, double inlierMaxDistance, double outlierError, const cv::Mat &distanceTransform, const bool useInterpolation) const
{
  CV_Assert(projectedPoints.cols == 1);
  CV_Assert(projectedPoints.type() == CV_32FC2);

  residuals.create(projectedPoints.rows, 1, CV_64FC1);
  for(int i=0; i<projectedPoints.rows; i++)
  {
    Point2f pt2f = projectedPoints.at<Vec2f>(i, 0);
    //cout << inlierMaxDistance << endl;
    residuals.at<double>(i, 0) = getFilteredDistance(pt2f, useInterpolation, inlierMaxDistance, outlierError, distanceTransform);
  }
}

void LocalPoseRefiner::computeResidualsWithInliersMask(const cv::Mat &projectedPoints, cv::Mat &residuals, double inlierMaxDistance, double outlierError, const cv::Mat &distanceTransform, const bool useInterpolation, float inliersRatio, cv::Mat &inliersMask) const
{
  computeResiduals(projectedPoints, residuals, params.outlierDistance, params.outlierError, distanceTransform, useInterpolation);

  CV_Assert(residuals.cols == 1);
  Mat sortedIndices;
  sortIdx(residuals, sortedIndices, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
  int maxRow = cvRound(inliersRatio * residuals.rows);
  CV_Assert(0 < maxRow && maxRow <= residuals.rows);
  CV_Assert(sortedIndices.type() == CV_32SC1);

  inliersMask = Mat(residuals.size(), CV_8UC1, Scalar(255));
  for (int i = maxRow; i < residuals.rows; ++i)
  {
    inliersMask.at<uchar>(sortedIndices.at<int>(i)) = 0;
  }
}

//Attention! projectedPoints is not const for efficiency
double LocalPoseRefiner::calcTrimmedError(cv::Mat &projectedPoints, bool useInterpolation, float h) const
{
  vector<float> residuals;
  computeResidualsForTrimmedError(projectedPoints, residuals);
  std::sort(residuals.begin(), residuals.end());

  int maxRow = cvRound(h * residuals.size());
  CV_Assert(0 < maxRow && static_cast<size_t>(maxRow) <= residuals.size());

  double error = sum(Mat(residuals).rowRange(0, maxRow))[0];
  error /= maxRow;
  //return sqrt(error);
  return error;
}

void LocalPoseRefiner::computeJacobian( const cv::Mat &projectedPoints, const cv::Mat &JaW, const cv::Mat &distanceImage, const cv::Mat &dx, const cv::Mat &dy, cv::Mat &J)
{
  CV_Assert(JaW.rows == 2*projectedPoints.rows);
  CV_Assert(JaW.type() == CV_64FC1);


  J.create(projectedPoints.rows, JaW.cols, CV_64FC1);

  //Rect imageRect(0, 0, distanceImage.cols, distanceImage.rows);
  const double outlierJ = 0;
  for(int i=0; i<projectedPoints.rows; i++)
  {
    Point2f pt2f = projectedPoints.at<Vec2f>(i);

    for(int j=0; j<J.cols; j++)
    {
      //if(imageRect.contains(pt))
      try
      {
        float x = getInterpolatedDT(dx, pt2f);
        float y = getInterpolatedDT(dy, pt2f);
        J.at<double>(i, j) = x * JaW.at<double>(2*i, j) + y * JaW.at<double>(2*i + 1, j);
        //J.at<double>(i, j) = dx.at<float>(pt) * JaW.at<double>(2*i, j) + dy.at<float>(pt) * JaW.at<double>(2*i + 1, j);
      }
      //else
      catch(Exception &e)
      {
        J.at<double>(i, j) = outlierJ;
      }
    }
  }
}

void LocalPoseRefiner::computeObjectJacobian(const cv::Mat &projectedPoints, const cv::Mat &JaW, const cv::Mat &distanceImage, const cv::Mat &dx, const cv::Mat &dy, const cv::Mat &R_obj2cam, const cv::Mat &t_obj2cam, const cv::Mat &rvec_obj, const cv::Mat &tvec_obj, cv::Mat &J)
{
  CV_Assert(JaW.rows == 2*projectedPoints.rows);
  CV_Assert(JaW.type() == CV_64FC1);
  J.create(projectedPoints.rows, JaW.cols, CV_64FC1);

  Mat J_camobj(JaW.cols, JaW.cols, CV_64FC1);
  Mat row0 = R_obj2cam.row(0);
  Mat row1 = R_obj2cam.row(1);
  Mat row2 = R_obj2cam.row(2);

  vector<Mat> drs;
  Mat drx = row1.cross(row2);
  CV_Assert(drx.rows == 1 && drx.cols == dim);
  CV_Assert(drx.type() == CV_64FC1);
  Mat dry = row2.cross(row0);
  Mat drz = row0.cross(row1);

  drs.push_back(drx);
  drs.push_back(dry);
  drs.push_back(drz);
  for(int ridx_cam=0; ridx_cam<dim; ridx_cam++)
  {
    for(int ridx_obj=0; ridx_obj<dim; ridx_obj++)
    {
      J_camobj.at<double>(ridx_cam, ridx_obj) = drs[ridx_cam].at<double>(0, ridx_obj);
    }
    for(int tidx_obj=0; tidx_obj<dim; tidx_obj++)
    {
      J_camobj.at<double>(ridx_cam, dim+tidx_obj) = 0;
    }
  }

  Mat R_obj, J_rodrigues;
  Rodrigues(rvec_obj, R_obj, J_rodrigues);
  CV_Assert(J_rodrigues.rows == 3 && J_rodrigues.cols == 9);
  Mat t_cam2obj = -R_obj2cam.t()*t_obj2cam;

  for(int tidx_cam=0; tidx_cam<dim; tidx_cam++)
  {
    for(int ridx_obj=0; ridx_obj<dim; ridx_obj++)
    {
      Mat dr = R_obj2cam * J_rodrigues.row(ridx_obj).reshape(1, dim) * t_cam2obj;
      CV_Assert(dr.type() == CV_64FC1);
      J_camobj.at<double>(dim+tidx_cam, ridx_obj) = dr.at<double>(tidx_cam, 0);
    }


    for(int tidx_obj=0; tidx_obj<dim; tidx_obj++)
    {
      J_camobj.at<double>(dim+tidx_cam, dim+tidx_obj) = R_obj2cam.at<double>(tidx_cam, tidx_obj);
    }
  }


  //Rect imageRect(0, 0, distanceImage.cols, distanceImage.rows);
  for(int i=0; i<projectedPoints.rows; i++)
  {
    Point2f pt2f = projectedPoints.at<Vec2f>(i);
    if(isOutlier(pt2f))
    {
      for(int j=0; j<J.cols; j++)
      {
        J.at<double>(i, j) = params.outlierJacobian;
      }

      continue;
    }

    double x = getInterpolatedDT(dx, pt2f);
    double y = getInterpolatedDT(dy, pt2f);

    for(int j=0; j<J.cols; j++)
    {
        double sumX = 0., sumY = 0.;

        for(int k=0; k<J.cols; k++)
        {
          sumX += JaW.at<double>(2*i, k) * J_camobj.at<double>(k, j);
          sumY += JaW.at<double>(2*i+1, k) * J_camobj.at<double>(k, j);
        }
        J.at<double>(i, j) = x * sumX + y * sumY;
        //J.at<double>(i, j) = x * JaW.at<double>(2*i, j) + y * JaW.at<double>(2*i + 1, j);
        //J.at<double>(i, j) = dx.at<float>(pt) * JaW.at<double>(2*i, j) + dy.at<float>(pt) * JaW.at<double>(2*i + 1, j);
    }
  }
}

void LocalPoseRefiner::displayProjection( const cv::Mat &projectedPoints, const string &title ) const
{
/*
  vector<Point2f> hull;
  convexHull(projectedPoints, hull);
  double area = contourArea(Mat(hull));
  cout << "Area: " << area << endl;
  cout << "hull: " << Mat(hull) << endl;
  cout << "Points: " << projectedPoints << endl;
*/
  static int iter = 0;
  //if(iter >= 2)
  //  return;
  //if( (iter % 10 != 0)  && iter >= 20)
    //return;
  iter++;


  CV_Assert(projectedPoints.type() == CV_32FC2);

#ifdef VERBOSE
  //std::cout << "Common rect: " << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << std::endl;
#endif

//  static Rect rect;
//  static int k=0;
//  if( k == 0)
//    getCommonBoundingRect(imagePoints, projectedPoints, rect);
//  k++;


  //Mat image(rect.height, rect.width, CV_8UC3, Scalar(255,255,255));
  static Mat image(edgesImage.size(), CV_8UC3, Scalar(255,255,255));

  image.setTo(Scalar(255,0,0), edgesImage);
  image.setTo(Scalar(255,255,255), ~edgesImage);

/*
 * CV_Assert(imagePoints.type() == CV_32FC2);
  Rect rect;
  //getCommonBoundingRect(imagePoints, projectedPoints, rect);
  getCommonBoundingRect(imagePoints, imagePoints, rect);

  for(int i=0; i<imagePoints.rows; i++)
  {
    Point pt = Point2f(imagePoints.at<Vec2f>(i, 0));
    circle(image, pt - rect.tl(), 2, Scalar(255,0,0),-1);
  }
*/
  for(int i=0; i<projectedPoints.rows; i++)
  {
    Point pt = Point2f(projectedPoints.at<Vec2f>(i, 0));
    //circle(image, pt - rect.tl(), 20, Scalar(0,0,255),-1);
    //circle(image, pt, 10, Scalar(0,0,255),-1);
    circle(image, pt, 1, Scalar(0,0,255),-1);
  }


#ifdef VERBOSE
  Mat scaled;
  //resize(image, scaled, Size(), 0.8, 0.8);
  //resize(image, scaled, Size(), 0.5, 0.5);
//  resize(image, scaled, Size(), 0.3, 0.3);
//  imshow("smallProjection", scaled);
  imshow(title, image);
  waitKey(20);
  if(iter == 1)
    waitKey();
#endif
}

double LocalPoseRefiner::getError(const cv::Mat &residuals) const
{
  return cv::norm(residuals) / sqrt((double) residuals.rows);
}

void LocalPoseRefiner::computeWeights(const vector<Point2f> &projectedPointsVector, const cv::Mat &silhouetteEdges, cv::Mat &weights) const
{
  for (size_t i = 0; i < projectedPointsVector.size(); ++i)
  {
    Point2f pt = projectedPointsVector[i];
    CV_Assert(!isnan(pt.x));
    CV_Assert(!isnan(pt.y));
  }
  CV_Assert(!weights.empty());

  Mat pointsMask;
  Point tl;
  EdgeModel::computePointsMask(projectedPointsVector, silhouetteEdges.size(), params.lmDownFactor, params.lmClosingIterationsCount, pointsMask, tl);
  if (pointsMask.empty())
  {
    weights = Mat();
    return;
  }
  vector<vector<Point> > contours;
  findContours(pointsMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
  Mat footprintImage(pointsMask.size(), CV_8UC1, Scalar(255));
  drawContours(footprintImage, contours, -1, Scalar(0));

  Mat footprintDT;
  distanceTransform(footprintImage, footprintDT, CV_DIST_L2, CV_DIST_MASK_PRECISE);

  for (int i = 0; i < weights.rows; ++i)
  {
    Point projectedPt = projectedPointsVector[i];
    Point pt = params.lmDownFactor * projectedPt - tl;

    CV_Assert(footprintDT.type() == CV_32FC1);
    float dist = isPointInside(footprintDT, pt) ? footprintDT.at<float>(pt) : params.outlierError;
    //cout << dist << endl;

    weights.at<double>(i) = 2 * exp(-dist);
    CV_Assert(!std::isnan(weights.at<double>(i)));
    //weights.at<double>(i) = 1 / (1 + footprintMatches[i].distance);
    //weights.at<double>(i) = 1;
  }


/*
  Mat projectedPointsFootprint;
  EdgeModel::computeFootprint(projectedPointsVector, silhouetteEdges.size(), projectedPointsFootprint, params.lmDownFactor, params.lmClosingIterationsCount);
  if (projectedPointsFootprint.empty())
  {
    weights = Mat();
    return;
  }
  CV_Assert(!projectedPointsFootprint.empty());

  Mat projectedPoints = Mat(projectedPointsVector);
  CV_Assert(projectedPoints.type() == projectedPointsFootprint.type());
  CV_Assert(projectedPoints.type() == CV_32FC2);

  FlannBasedMatcher flannBasedMatcher(new flann::KDTreeIndexParams());
  std::vector<DMatch> footprintMatches;
  flannBasedMatcher.match(projectedPoints.reshape(1), projectedPointsFootprint.reshape(1), footprintMatches);
  CV_Assert(static_cast<size_t>(projectedPoints.rows) == footprintMatches.size());
  CV_Assert(static_cast<size_t>(weights.rows) == footprintMatches.size());

  for (int i = 0; i < weights.rows; ++i)
  {
    weights.at<double>(i) = 2 * exp(-footprintMatches[i].distance);
    CV_Assert(!isnan(weights.at<double>(i)));
    //weights.at<double>(i) = 1 / (1 + footprintMatches[i].distance);
    //weights.at<double>(i) = 1;
  }
*/
}

void LocalPoseRefiner::computeWeightsObjectJacobian(const vector<Point3f> &points, const cv::Mat &silhouetteEdges, const PoseRT &pose_obj, Mat &weightsJacobian) const
{
  vector<Point2f> projectedPointsVector;
  {
    Mat rvec_cam, tvec_cam, Rt_cam;
    projectPoints_obj(Mat(points), pose_obj.getRvec(), pose_obj.getTvec(), rvec_cam, tvec_cam, Rt_cam, projectedPointsVector);
  }

  Mat weightsSrc(points.size(), 1, CV_64FC1);
  computeWeights(projectedPointsVector, silhouetteEdges, weightsSrc);

  const int colsCount = 6;
  const int dim = 3;
//  const double rvecDiff = CV_PI / 180;
//  const double tvecDiff = 0.001;
  const double rvecDiff = CV_PI / 10;
  const double tvecDiff = 0.01;

  weightsJacobian.create(points.size(), colsCount, CV_64FC1);
  for(int i = 0; i < colsCount; ++i)
  {
    PoseRT newPose_obj = pose_obj;
    double diff;
    if (i < dim)
    {
      newPose_obj.rvec.at<double>(i) += rvecDiff;
      diff = rvecDiff;
    }
    else
    {
      newPose_obj.tvec.at<double>(i - dim) += tvecDiff;
      diff = tvecDiff;
    }

    Mat weightsProjected(points.size(), 1, CV_64FC1);
    {
      Mat rvec_cam, tvec_cam, Rt_cam;
      projectPoints_obj(Mat(points), newPose_obj.getRvec(), newPose_obj.getTvec(), rvec_cam, tvec_cam, Rt_cam, projectedPointsVector);
    }
    computeWeights(projectedPointsVector, silhouetteEdges, weightsProjected);

    Mat col = weightsJacobian.col(i);
    Mat dw = (weightsProjected - weightsSrc) / diff;
    dw.copyTo(col);
  }
}

float LocalPoseRefiner::refineUsingSilhouette(PoseRT &pose_cam, bool usePoseGuess, const cv::Vec4f &tablePlane, cv::Mat *finalJacobian)
{
#ifdef VERBOSE
  std::cout << "Local refinement started!" << std::endl;
#endif
  PoseRT poseInit_cam;
  if(usePoseGuess)
  {
    poseInit_cam = pose_cam;
    setInitialPose(pose_cam);
  }

  bool hasRotationSymmetry = rotatedEdgeModel.hasRotationSymmetry;
  const int verticalDirectionIndex = 2;

  bool useObjectCoordinateSystem = !Rt_obj2cam_cached.empty();
  if(!useObjectCoordinateSystem)
    CV_Error(CV_StsBadArg, "camera coordinate system is not supported");

  Mat R_obj2cam, t_obj2cam;
  getRotationTranslation(Rt_obj2cam_cached, R_obj2cam, t_obj2cam);

  Mat newTranslationBasis2old;
  const float eps = 1e-4;
  if (cv::norm(tablePlane) > eps)
  {
    Mat Rot_obj2cam = rotatedEdgeModel.Rt_obj2cam.clone();
    Rot_obj2cam(Range(0, 3), Range(3, 4)).setTo(0);
    Point3d tableNormal(tablePlane[0], tablePlane[1], tablePlane[2]);
    Point3d tableNormal_obj;
    transformPoint(Rot_obj2cam.inv(DECOMP_SVD), tableNormal, tableNormal_obj);
    Point3d t1(tableNormal_obj.z, 0, -tableNormal_obj.x);
    //TODO: fix
    CV_Assert(norm(t1) > eps);
    Point3d t2 = tableNormal_obj.cross(t1);

#if 0
    //
    // A dependency on C++11... for this?
    //
    // This code is fine, but to compile it, the -std=c++0x flag
    // pulls in the entirety of boost C++11 support, and the boost.thread library
    // in 1.42.0 has some problem involving rvalue refs that has probably been fixed
    // in more recent versions.
    //
    // PLEASE: *always* test on all common platforms.  It is much,
    // much easier to have the original author catch these things as
    // they're being written than for, 1.  some unsuspecting user to
    // lose hours trying to figure out if it is "just them" or not
    // and, 2. somebody like me to drop what they're doing, just to
    // tell said user that it isn't their fault.  In this case the
    // user was one of the Good Ones with a carefully prepared
    // machine, no funny customizations.
    //
    vector<Point3d> basis = {tableNormal_obj, t1, t2};
#else
    vector<Point3d> basis;
    basis.push_back(tableNormal_obj);
    basis.push_back(t1);
    basis.push_back(t2);
#endif
    Mat oldBasis2new = Mat(basis).reshape(1);
    Mat newBasis2old = oldBasis2new.inv(DECOMP_SVD);

    newTranslationBasis2old = newBasis2old.colRange(1, 3).clone();
  }

  int paramsCount = 2*this->dim;
  if (!newTranslationBasis2old.empty())
  {
    paramsCount = newTranslationBasis2old.cols;
    if (!hasRotationSymmetry)
    {
      ++paramsCount;
    }
  }
  const int residualsCount = originalEdgeModel.stableEdgels.size() + originalEdgeModel.points.size();

  Mat params(paramsCount, 1, CV_64FC1, Scalar(0));
  Mat rvecParams, tvecParams;
  if (newTranslationBasis2old.empty())
  {
    rvecParams = params.rowRange(0, this->dim);
    tvecParams = params.rowRange(this->dim, 2*this->dim);
  }
  else
  {
    rvecParams = Mat::zeros(dim, 1, CV_64FC1);
    tvecParams = Mat::zeros(dim, 1, CV_64FC1);
  }

  //TODO: check TermCriteria from solvePnP
  CvLevMarq solver(paramsCount, residualsCount);
  CvMat paramsCvMat = params;
  cvCopy( &paramsCvMat, solver.param );
  //params and solver.params must use the same memory

  vector<Point2f> projectedStableEdgelsVector, projectedPointsVector;
  Mat projectedStableEdgels, projectedPoints;
  Mat err;
  int iter = 0;
  float startError = -1.f;
  float finishError = -1.f;

  Mat silhouetteWeights(rotatedEdgeModel.points.size(), 1, CV_64FC1);
  //TODO: number of iterations is greater in 2 times than needed (empty iterations)
  for(;;)
  {
    //cout << "Params: " << params << endl;
    //cout << "Iteration N: " << iter++ << endl;

    CvMat *matJ = 0, *_err = 0;
    const CvMat *__param = 0;
    bool proceed = solver.update( __param, matJ, _err );
    //if( iter != 1 )
      //proceed = solver.update( __param, matJ, _err );
    cvCopy( __param, &paramsCvMat );
    if( !proceed || !_err )
        break;

    Mat silhouetteErr, surfaceErr;
//      if( matJ )
    {
      if (!newTranslationBasis2old.empty())
      {
        Mat translationPart = params.rowRange(paramsCount - newTranslationBasis2old.cols, paramsCount);
        tvecParams = newTranslationBasis2old * translationPart;
        CV_Assert(tvecParams.rows == dim);
        CV_Assert(tvecParams.cols == 1);
      }

      //TODO: get rid of code duplication
      Mat surfaceJaW(2 * originalEdgeModel.stableEdgels.size(), 2 * dim, CV_64F);
      Mat surfaceDpdrot = surfaceJaW.colRange(0, this->dim);
      Mat surfaceDpdt = surfaceJaW.colRange(this->dim, 2 * this->dim);
      Mat rvecParams_cam, tvecParams_cam, RtParams_cam;
      projectPoints_obj(Mat(rotatedEdgeModel.stableEdgels), rvecParams, tvecParams, rvecParams_cam, tvecParams_cam, RtParams_cam, projectedStableEdgelsVector, &surfaceDpdrot, &surfaceDpdt);
      projectedStableEdgels = Mat(projectedStableEdgelsVector);
      Mat surfaceInliersMask;
      computeResidualsWithInliersMask(projectedStableEdgels, surfaceErr, this->params.outlierDistance, this->params.outlierError, dtImage, true, this->params.lmInliersRatio, surfaceInliersMask);
      Mat surfaceJ;
      computeObjectJacobian(projectedStableEdgels, surfaceJaW, dtImage, dtDx, dtDy, R_obj2cam, t_obj2cam, rvecParams, tvecParams, surfaceJ);
      surfaceErr.setTo(0, ~surfaceInliersMask);
      for (int i = 0; i < surfaceJ.cols; ++i)
      {
        Mat col = surfaceJ.col(i);
        col.setTo(0, ~surfaceInliersMask);
      }

      if (!newTranslationBasis2old.empty())
      {
        Mat newSurfaceJ(surfaceJ.rows, paramsCount, surfaceJ.type());
        if (!hasRotationSymmetry)
        {
          Mat rotationJ = surfaceJ.colRange(verticalDirectionIndex, verticalDirectionIndex + 1);
          Mat theFirstCol = newSurfaceJ.col(0);
          rotationJ.copyTo(theFirstCol);
        }
        Mat translationJ = surfaceJ.colRange(dim, 2*dim) * newTranslationBasis2old;
        Mat lastCols = newSurfaceJ.colRange(paramsCount - newTranslationBasis2old.cols, paramsCount);
        translationJ.copyTo(lastCols);
        surfaceJ = newSurfaceJ;
      }

      Mat silhouetteJaW(2 * originalEdgeModel.points.size(), 2 * dim, CV_64F);
      Mat silhouetteDpdrot = silhouetteJaW.colRange(0, this->dim);
      Mat silhouetteDpdt = silhouetteJaW.colRange(this->dim, 2 * this->dim);
      projectPoints_obj(Mat(rotatedEdgeModel.points), rvecParams, tvecParams, rvecParams_cam, tvecParams_cam, RtParams_cam, projectedPointsVector, &silhouetteDpdrot, &silhouetteDpdt);
      projectedPoints = Mat(projectedPointsVector);
      Mat silhouetteInliersMask;
      computeResidualsWithInliersMask(projectedPoints, silhouetteErr, this->params.outlierDistance, this->params.outlierError, silhouetteDtImage, true, this->params.lmInliersRatio, silhouetteInliersMask);
      Mat silhouetteJ;
      computeObjectJacobian(projectedPoints, silhouetteJaW, silhouetteDtImage, silhouetteDtDx, silhouetteDtDy, R_obj2cam, t_obj2cam, rvecParams, tvecParams, silhouetteJ);
      silhouetteErr.setTo(0, ~silhouetteInliersMask);
      for (int i = 0; i < silhouetteJ.cols; ++i)
      {
        Mat col = silhouetteJ.col(i);
        col.setTo(0, ~silhouetteInliersMask);
      }

      if (!newTranslationBasis2old.empty())
      {
        Mat newSilhouetteJ(silhouetteJ.rows, paramsCount, silhouetteJ.type());
        if (!hasRotationSymmetry)
        {
          Mat rotationJ = silhouetteJ.colRange(verticalDirectionIndex, verticalDirectionIndex + 1);
          Mat theFirstCol = newSilhouetteJ.col(0);
          rotationJ.copyTo(theFirstCol);
        }
        Mat translationJ = silhouetteJ.colRange(dim, 2*dim) * newTranslationBasis2old;
        Mat lastCols = newSilhouetteJ.colRange(paramsCount - newTranslationBasis2old.cols, paramsCount);
        translationJ.copyTo(lastCols);
        silhouetteJ = newSilhouetteJ;
      }

      computeWeights(projectedPointsVector, silhouetteEdges, silhouetteWeights);
      if (silhouetteWeights.empty())
      {
        return std::numeric_limits<float>::max();
      }

      for (int i = 0; i < silhouetteJ.cols; ++i)
      {
        Mat col = silhouetteJ.col(i);
        Mat mulCol = silhouetteWeights.mul(col);
        mulCol.copyTo(col);
      }

      const bool isJacoianNumerical = false;

      if (isJacoianNumerical)
      {
        Mat weightsJacobian;
        computeWeightsObjectJacobian(rotatedEdgeModel.points, silhouetteEdges, PoseRT(rvecParams, tvecParams), weightsJacobian);
        for (int i = 0; i < weightsJacobian.cols; ++i)
        {
          Mat col = weightsJacobian.col(i);
          Mat mulCol = silhouetteErr.mul(col);
          mulCol.copyTo(col);
        }

        silhouetteJ += weightsJacobian;
      }

      Mat J = surfaceJ.clone();
      J.push_back(silhouetteJ);

      CvMat JcvMat = J;
      //cvCopy(&JcvMat, matJ);
      cvCopy(&JcvMat, solver.J);
    }

    err = surfaceErr.clone();
    CV_Assert(silhouetteWeights.type() == silhouetteErr.type());
    Mat weightedSilhouetteErr = silhouetteWeights.mul(silhouetteErr);
    err.push_back(weightedSilhouetteErr);

#ifdef VERBOSE
    cout << "Errors:" << endl;
    cout << "  surface: " << getError(surfaceErr) << endl;
    cout << "  silhouette: " << getError(weightedSilhouetteErr) << endl;
#endif

    if(iter == 0)
    {
      startError = getError(err);
    }

#ifdef VERBOSE
    if(iter % 2 == 0)
    {
      //cout << "Error[" << iter / 2 << "]: " << getError(err) << endl;
    }
    if(iter == 0)
    {
      startError = getError(err);
      cout << "Start error: " << startError << endl;
    }
    //if(index != 1 )

#ifdef VISUALIZE
    displayProjection(projectedPoints, "points");
    displayProjection(projectedStableEdgels, "surface points");
#endif
#endif
    CvMat errCvMat = err;
    cvCopy( &errCvMat, _err);
    ++iter;
  }

  if (finalJacobian != 0)
  {
    CvMat cvJ = *(solver.J);
    Mat J = &cvJ;
    J.copyTo(*finalJacobian);
  }

  finishError = getError(err);
#ifdef VERBOSE
  cout << "Start error: " << startError << endl;
  cout << "Final error: " << finishError << endl;
  cout << "Optimization errors' ratio: " << finishError / startError << endl;
#endif
  cvCopy( solver.param, &paramsCvMat );

  if (!newTranslationBasis2old.empty())
  {
    Mat translationPart = params.rowRange(paramsCount - newTranslationBasis2old.cols, paramsCount);
    tvecParams = newTranslationBasis2old * translationPart;
  }

  Mat transMat;
  getTransformationMatrix(R_obj2cam, t_obj2cam, rvecParams, tvecParams, transMat);
  transMat = transMat * poseInit_cam.getProjectiveMatrix();
  pose_cam.setProjectiveMatrix(transMat);

  return finishError;
}

void LocalPoseRefiner::refine(cv::Mat &rvec_cam, cv::Mat &tvec_cam, bool usePoseGuess)
{
  CV_Assert(false);
/*
  std::cout << "Local refinement started!" << std::endl;
  Mat rvecInit_cam(dim, 1, CV_64FC1, Scalar(0));
  Mat tvecInit_cam(dim, 1, CV_64FC1, Scalar(0));
  if(usePoseGuess)
  {
    rvec_cam.copyTo(rvecInit_cam);
    tvec_cam.copyTo(tvecInit_cam);
    setInitialPose(rvec_cam, tvec_cam);
  }

  //bool useObjectCoordinateSystem = (!R_obj2cam.empty()) && (!t_obj2cam.empty());
  //bool useObjectCoordinateSystem = true;

  bool useObjectCoordinateSystem = !Rt_obj2cam_cached.empty();

  Mat R_obj2cam, t_obj2cam;
  getRotationTranslation(Rt_obj2cam_cached, R_obj2cam, t_obj2cam);

  if(!useObjectCoordinateSystem)
    CV_Error(CV_StsBadArg, "camera coordinate system is not tested");

  const int paramsCount = 2*this->dim;
  const int pointsCount = originalEdgeModel.points.size();

  Mat params(paramsCount, 1, CV_64FC1, Scalar(0));
  Mat rvecParams = params.rowRange(0, this->dim);
  Mat tvecParams = params.rowRange(this->dim, 2*this->dim);

  //rvec.copyTo(rvecParams);
  //tvec.copyTo(tvecParams);


  //TODO: check TermCriteria from solvePnP
  int residualsCount = pointsCount;
  CvLevMarq solver(paramsCount, residualsCount);
  CvMat paramsCvMat = params;
  cvCopy( &paramsCvMat, solver.param );
  //params and solver.params must use the same memory

  Mat dpdrot, dpdt;
  Mat dpdf, dpdc, dpddist;
  vector<Point2f> projectedPointsVector;
  Mat projectedPoints;
  Mat err;
  int iter = 0;
  float startError = -1.f;
  float finishError = -1.f;

  //TODO: number of iterations is greater in 2 times than needed (empty iterations)
  for(;;)
  {
      //cout << "Params: " << params << endl;
      //cout << "Iteration N: " << iter++ << endl;

      CvMat *matJ = 0, *_err = 0;
      const CvMat *__param = 0;
      bool proceed = solver.update( __param, matJ, _err );
      //if( iter != 1 )
        //proceed = solver.update( __param, matJ, _err );
      cvCopy( __param, &paramsCvMat );
      if( !proceed || !_err )
          break;

//      if( matJ )
      {
          Mat JaW(2*pointsCount, paramsCount, CV_64F);
          dpdrot = JaW.colRange(0, this->dim);
          dpdt = JaW.colRange(this->dim, 2 * this->dim);

          if(useObjectCoordinateSystem)
          {
            //projectObjectPoints(objectPoints, R_obj2cam, t_obj2cam, cameraMatrix, distCoeffs, rvecParams, tvecParams, projectedPointsVector, dpdrot, dpdt);
            Mat rvec_cam, tvec_cam, Rt_cam;
            projectPoints_obj(Mat(rotatedEdgeModel.points), rvecParams, tvecParams, rvec_cam, tvec_cam, Rt_cam, projectedPointsVector, &dpdrot, &dpdt);
          }
          else
          {
            projectPoints(Mat(rotatedEdgeModel.points), rvecParams, tvecParams, cameraMatrix, distCoeffs, projectedPointsVector, dpdrot, dpdt, dpdf, dpdc, dpddist, 0);
          }
          projectedPoints = Mat(projectedPointsVector);

          Mat J;
          if(useObjectCoordinateSystem)
          {
            //computeObjectJacobian(R_obj2cam, t_obj2cam, rvec, tvec, rvecParams, tvecParams, J);
            //computeObjectJacobian(projectedPoints, JaW, dtImage, dtDx, dtDy, dtOrigin, R_obj2cam, t_obj2cam, rvecParams, tvecParams, J);
            computeObjectJacobian(projectedPoints, JaW, dtImage, dtDx, dtDy, R_obj2cam, t_obj2cam, rvecParams, tvecParams, J);
          }
          else
          {
            //TODO: normalize tvec and rvec (currently small diff in tvec can bring too big diff. in the cost function)
            computeJacobian(projectedPoints, JaW, dtImage, dtDx, dtDy, J);
          }
#ifdef VERBOSE
          //cout << "Jacobian: " << J << endl;
#endif
          CvMat JcvMat = J;
          //cvCopy(&JcvMat, matJ);
          cvCopy(&JcvMat, solver.J);
      }
//      else
//      {
//          projectPoints(objectPoints, rvecParams, tvecParams, cameraMatrix, Mat(), projectedPointsVector);
//          projectedPoints = Mat(projectedPointsVector);
//#ifdef USE_DT
//          computeDistanceTransform(projectedPoints, distanceImage, origin);
//#endif
//      }


      computeResiduals(projectedPoints, err, this->params.outlierDistance, this->params.outlierError);

      if(iter == 0)
      {
        startError = getError(err);
      }

#ifdef VERBOSE
      if(iter % 2 == 0)
      {
        //cout << "Error[" << iter / 2 << "]: " << getError(err) << endl;
      }
      if(iter == 0)
      {
        startError = getError(err);
        cout << "Start error: " << startError << endl;
      }
      //if(index != 1 )

#ifdef VISUALIZE
        displayProjection(projectedPoints);
#endif
#endif
      CvMat errCvMat = err;
      cvCopy( &errCvMat, _err);
      iter++;
  }
  finishError = getError(err);
  cout << "Final error: " << finishError << endl;
  cout << "Optimization errors' ratio: " << finishError / startError << endl;
  cvCopy( solver.param, &paramsCvMat );

  Mat Rt_init;
  createProjectiveMatrix(rvecInit_cam, tvecInit_cam, Rt_init);
  if(useObjectCoordinateSystem)
  {
    Mat transMat;
    //getTransformationMatrix(R_obj2cam, t_obj2cam, rvecParams, tvecParams, transMat);
    getTransformationMatrix(R_obj2cam, t_obj2cam, rvecParams, tvecParams, transMat);

    transMat = transMat * Rt_init;

    getRvecTvec(transMat, rvec_cam, tvec_cam);
  }
  else
  {
    Mat Rt_found;
    createProjectiveMatrix(rvecParams, tvecParams, Rt_found);
    Mat Rt_result = Rt_found * Rt_init;

    getRvecTvec(Rt_result, rvec_cam, tvec_cam);
  }
*/
}

void LocalPoseRefiner::object2cameraTransformation(const cv::Mat &rvec_obj, const cv::Mat &tvec_obj, cv::Mat &Rt_cam) const
{
  CV_Assert(!Rt_obj2cam_cached.empty() && !Rt_cam2obj_cached.empty());

  Mat Rt_object;
  createProjectiveMatrix(rvec_obj, tvec_obj, Rt_object);
  Mat Rt = Rt_obj2cam_cached * Rt_object * Rt_cam2obj_cached;
  Rt_cam = extrinsicsRt * Rt;
}

void LocalPoseRefiner::projectPoints_obj(const Mat &points, const Mat &rvec_Object, const Mat &tvec_Object, Mat &rvec_cam, Mat &tvec_cam, Mat &Rt_cam, vector<Point2f> &imagePoints, Mat *dpdrot, Mat *dpdt) const
{
  CV_Assert(points.type() == CV_32FC3);
  if(rvec_cam.empty() || tvec_cam.empty())
  {
    object2cameraTransformation(rvec_Object, tvec_Object, Rt_cam);
    getRvecTvec(Rt_cam, rvec_cam, tvec_cam);
  }

  if(dpdrot != 0 && dpdt != 0)
  {
    //Mat dpdf, dpdc, dpddist;
    //projectPoints(points, rvec_cam, tvec_cam, cameraMatrix, distCoeffs, imagePoints, *dpdrot, *dpdt, dpdf, dpdc, dpddist);
    Mat jacobian;
    projectPoints(points, rvec_cam, tvec_cam, cameraMatrix, distCoeffs, imagePoints, jacobian);
    const int dim = 3;

    jacobian.colRange(0, dim).copyTo(*dpdrot);
    jacobian.colRange(dim, 2 * dim).copyTo(*dpdt);
  }
  else
  {
    projectPoints(points, rvec_cam, tvec_cam, cameraMatrix, distCoeffs, imagePoints);
  }

  CV_Assert(static_cast<size_t>(points.rows) == imagePoints.size());
}

void LocalPoseRefiner::setCenterMask(const cv::Mat &_mask)
{
  centerMask = _mask.clone();
  distanceTransform(~centerMask, dtCenter, CV_DIST_L2, CV_DIST_MASK_PRECISE);
}

void filterOutliers(const cv::Mat &samples, cv::Mat &filteredSamples, double outlierThreshold = 1e10)
{
  vector<bool> isCorrect(samples.rows);
  int filteredSamplesCount = 0;
  for(int i=0; i<samples.rows; i++)
  {
    double maxVal, minVal;
    minMaxIdx(samples.row(i), &minVal, &maxVal);
    isCorrect[i] = fabs(maxVal) < outlierThreshold && fabs(minVal) < outlierThreshold;
    filteredSamplesCount += isCorrect[i];
  }

  if(filteredSamplesCount == samples.rows)
  {
    samples.copyTo(filteredSamples);
  }
  else
  {
    filteredSamples.create(filteredSamplesCount, samples.cols, samples.type());
    int curRow = 0;
    for(int i=0; i<samples.rows; i++)
    {
      if(isCorrect[i])
      {
        Mat row = filteredSamples.row(curRow);
        samples.row(i).copyTo(row);
        curRow++;
      }
    }
    CV_Assert(curRow == filteredSamplesCount);
  }
}

void computeDots(const Mat &mat1, const Mat &mat2, Mat &dst)
{
  Mat m1 = mat1.reshape(1);
  Mat m2 = mat2.reshape(1);
  CV_Assert(m1.size() == m2.size());
  CV_Assert(m1.type() == m2.type());

  Mat products = m1.mul(m2);
  reduce(products, dst, 1, CV_REDUCE_SUM);
}

void computeNormalDots(const Mat &Rt, const EdgeModel &rotatedEdgeModel, Mat &dots)
{
  Mat R = Rt(Range(0, 3), Range(0, 3));
  Mat t = Rt(Range(0, 3), Range(3, 4));

  Mat vec;
  Mat(t.t() * R).reshape(3).convertTo(vec, CV_32FC1);
  Scalar scalar = Scalar(vec.at<Vec3f>(0));
  Mat edgelsMat = Mat(rotatedEdgeModel.points) + scalar;

  Mat norms;
  computeDots(edgelsMat, edgelsMat, norms);
  sqrt(norms, norms);

  computeDots(edgelsMat, Mat(rotatedEdgeModel.normals), dots);
  float epsf = 1e-4;
  Scalar eps = Scalar::all(epsf);
  dots /= (norms + epsf);
}

//#define USE_NORMALS
//#define USE_CONVEXHULL
//#define DEBUG_CENTER_MASK
double LocalPoseRefiner::estimatePoseQuality(const cv::Mat &rvec_Object, const cv::Mat &tvec_Object, float hTrimmedError, double *detOfCovarianceMatrix) const
{
  if(!centerMask.empty())
  {
    const double invalidValue = std::numeric_limits<double>::max();
    Point3f objectCenter = rotatedEdgeModel.getObjectCenter();
    vector<Point3f> points(1, objectCenter);
    vector<Point2f> projectedPoints;
    Mat rvec_cam, tvec_cam, Rt;
    projectPoints_obj(Mat(points), rvec_Object, tvec_Object, rvec_cam, tvec_cam, Rt, projectedPoints);

    Point pt = projectedPoints[0];

#ifdef DEBUG_CENTER_MASK
    Mat centerImage(edgesImage.size(), CV_8UC1, Scalar(0));
    imshow("center image", centerImage);
#endif

    if(pt.x < 0 || pt.x >= centerMask.cols || pt.y < 0 || pt.y >= centerMask.rows)
    {
#ifdef DEBUG_CENTER_MASK
      imshow("center image", centerImage);
      waitKey();
#endif
      return invalidValue;
    }

#ifdef DEBUG_CENTER_MASK
    circle(centerImage, pt, 3, Scalar(255), -1);
    imshow("center image", centerImage);
    waitKey();
#endif

    CV_Assert(centerMask.type() == CV_8UC1);
    double result = (centerMask.at<uchar>(pt) == 255) ? 0.0 : invalidValue;
    return result;
  }


  vector<Point2f> projectedPointsVector;

  Mat projectedViewDependentEdgels;

//Mat projectedPointsImg;
//Mat imageROI;
//vector<Point2f> boundary;

  vector<Point2f> projectedStableEdgelsVector;
  Mat rvec_cam, tvec_cam, Rt;
  projectPoints_obj(Mat(rotatedEdgeModel.stableEdgels), rvec_Object, tvec_Object, rvec_cam, tvec_cam, Rt, projectedStableEdgelsVector);
  Mat projectedStableEdgels = Mat(projectedStableEdgelsVector);

  if(params.useViewDependentEdges)
  {
//TODO: edgesImage.size() may be not equal to calibration resolution -- proccess this case
    PoseRT extrinsics;
    extrinsics.setProjectiveMatrix(extrinsicsRt);

    cv::Ptr<const PinholeCamera> pinholeCamera = new PinholeCamera(cameraMatrix, distCoeffs, extrinsics, edgesImage.size());
    PoseRT pose_cam(rvec_cam, tvec_cam);
    pose_cam = extrinsics.inv() * pose_cam;

    Silhouette silhouette;
    rotatedEdgeModel.getSilhouette(pinholeCamera, pose_cam, silhouette, params.downFactor, params.closingIterations);
    silhouette.getEdgels(projectedViewDependentEdgels);

    if(projectedViewDependentEdgels.empty())
    {
      return std::numeric_limits<double>::max();
    }
  }

  Mat allProjectedPoints = projectedStableEdgels.clone();
  if(params.useViewDependentEdges)
  {
    allProjectedPoints.push_back(projectedViewDependentEdgels);
  }

  if(detOfCovarianceMatrix != 0)
  {
    Mat covar, mean;
    calcCovarMatrix(allProjectedPoints.reshape(1), covar, mean, CV_COVAR_NORMAL | CV_COVAR_SCALE | CV_COVAR_ROWS);

    //TODO: why det < 0 ?
//    if(det < 0)
//    {
//      hullSize = 0;
//      return std::numeric_limits<double>::max();
//    }
//    *hullSize = sqrt(det);

    *detOfCovarianceMatrix = sqrt(determinant(covar));
  }



#ifdef VISUALIZE_NLOPT
  displayProjection(projectedPoints);
#endif

  double orientationCost = 0.0;
  int orientationsNumber = 0;
  if(params.useOrientedChamferMatching)
  {
    CV_Assert(false);

/*
    if(params.useViewDependentEdges)
    {
//      projectedPointsImg = Scalar(0);
//      for(size_t i=0; i<boundary.size(); i++)
//      {
//        Point pt = boundary[i];
//        projectedPointsImg.at<uchar>(pt) = 255;
//      }

      IplImage *projectedOrientationsImage = cvCreateImage(cvSize(edgesImage.size().width, edgesImage.size().height), IPL_DEPTH_32F, 1);
      cvSetZero(projectedOrientationsImage);
      IplImage edge_img = imageROI;
      computeEdgeOrientations(&edge_img, projectedOrientationsImage, params.objectM);
      //computeContoursOrientations(contours, projectedOrientationsImage);
      Mat projectedOrientations = projectedOrientationsImage;

  #ifdef OCM_VISUALIZE
      Mat ocmViz(orientationImage.size(), CV_32FC1, Scalar(0));
  #endif
      for(size_t i=0; i<boundary.size(); i++)
      {
        Point projectedPt = boundary[i];
        CV_Assert(projectedViewDependentEdgels.channels() == 2);
        CV_Assert(projectedViewDependentEdgels.rows == 1 || projectedViewDependentEdgels.cols == 1);
        Point2f srcProjectedPt = projectedViewDependentEdgels.at<Vec2f>(i);

        CV_Assert(projectedOrientations.type() == CV_32FC1);
        float projectedAngle = projectedOrientations.at<float>(projectedPt);
  #ifdef OCM_VISUALIZE
      ocmViz.at<float>(projectedPt) = 50.0 + 205.0 * std::min(projectedAngle, CV_PI-projectedAngle) / (CV_PI / 2.0);
      //ocmViz.at<float>(projectedPt) = 150.0 + 105.0 * std::min(projectedAngle, CV_PI-projectedAngle) / (CV_PI / 2.0);
      //ocmViz.at<float>(projectedPt) = 255;
  #endif

        CV_Assert(orientationImage.type() == CV_32FC1);
        float testAngle = orientationImage.at<float>(srcProjectedPt);
        if(isnan(testAngle) || isnan(projectedAngle))
          continue;

        const float eps = 1e-4;
        CV_Assert(-eps <= projectedAngle && projectedAngle <= CV_PI+eps);
        CV_Assert(-eps <= testAngle && testAngle <= CV_PI+eps);

        //orientations.push_back(orientation);
        float or2 = CV_PI - fabs(projectedAngle - testAngle);
        orientationCost += std::min(fabs(projectedAngle - testAngle), or2 );
        orientationsNumber++;
      }
      cvReleaseImage(&projectedOrientationsImage);


  #ifdef OCM_VISUALIZE
    static int iter = 0;
  //  if(iter == 0)
  //    imwrite("orientations.png", ocmViz);
    imshow("orientations.png", ocmViz / 255.0);
    waitKey();
    iter++;
  #endif

    }
    else
    {
      //TODO: use distortion
      Mat P = cameraMatrix64F * Rt(Rect(0, 0, 4, 3));

      Mat projectedObjectPoints, projectedObjectOrientations;
      transform(Mat(originalEdgeModel.points), projectedObjectPoints, P);
      transform(Mat(originalEdgeModel.orientations), projectedObjectOrientations, P);
      CV_Assert(projectedObjectPoints.type() == CV_32FC3);
      CV_Assert(projectedObjectOrientations.type() == CV_32FC3);

      //vector<float> orientations;
      CV_Assert(originalEdgeModel.orientations.size() == projectedPointsVector.size());
    #ifdef OCM_VISUALIZE
      Mat ocmViz(orientationImage.size(), CV_32FC1, Scalar(0));
    #endif
      for(size_t i=0; i<projectedPointsVector.size(); i++)
      {
        Point projectedPt = projectedPointsVector[i];
        if(projectedPt.x < 0 || projectedPt.y < 0 || projectedPt.x >= orientationImage.cols || projectedPt.y >= orientationImage.rows)
          continue;

        CV_Assert(orientationImage.type() == CV_32FC1);
        double testAngle = orientationImage.at<float>(projectedPt);
        if(isnan(testAngle))
          continue;


        Vec3f pt = projectedObjectPoints.at<Vec3f>(i, 0);
        Vec3f ort = projectedObjectOrientations.at<Vec3f>(i, 0);

        //double dx = (ort[0] * pt[2] - pt[0] * ort[2]) / (pt[2] * pt[2]);
        //double dy = (ort[1] * pt[2] - pt[1] * ort[2]) / (pt[2] * pt[2]);

        //you need only orientation so you can ignore denominator
        double dx = (ort[0] * pt[2] - pt[0] * ort[2]);
        double dy = (ort[1] * pt[2] - pt[1] * ort[2]);


        //TODO: use -dy?
        double orientation = atan2(-dy, dx);

        if(orientation<0)
        {
          orientation += CV_PI;
        }

        //orientations.push_back(orientation);
        double or2 = CV_PI - fabs(orientation - testAngle);
        orientationCost += std::min(fabs(orientation - testAngle), or2 );
        orientationsNumber++;

    #ifdef OCM_VISUALIZE
        ocmViz.at<float>(projectedPt) = 50.0 + 205.0 * std::min(orientation, CV_PI-orientation) / (CV_PI / 2.0);
    #endif
      }

    #ifdef OCM_VISUALIZE
      static int iter = 0;
      if(iter == 0)
        imwrite("orientations.png", ocmViz);
      iter++;
    #endif

      //CV_Assert(orientations.size() == projectedPointsVector.size());


//      for(size_t i=0; i<projectedPointsVector.size(); i++)
//      {
//        Point pt = projectedPointsVector[i];
//        if(pt.x < 0 || pt.y < 0 || pt.x >= orientationImage.cols || pt.y >= orientationImage.rows)
//          continue;
//
//
//        if(!isnan(testAngle))
//        {
//          //orientationCost += fabs(tplAngle - testAngle);
//          //cout << orientations[i] << endl;
//          float or2 = CV_PI - fabs(orientations[i] - testAngle);
//          orientationCost += std::min(fabs(orientations[i] - testAngle), or2 );
//          orientationsNumber++;
//        }
//      }

    }
*/
  }

  CV_Assert(params.viewDependentEdgesWeight >= 0.0f && params.viewDependentEdgesWeight <= 1.0f);
  double viewDependentError = 0.0;
  if(params.useViewDependentEdges)
  {
    viewDependentError = calcTrimmedError(projectedViewDependentEdgels, false, hTrimmedError);
  }

  double stableEdgelsError = calcTrimmedError(projectedStableEdgels, false, hTrimmedError);
  double error = params.viewDependentEdgesWeight * viewDependentError + (1 - params.viewDependentEdgesWeight) * stableEdgelsError;
  if(params.useOrientedChamferMatching)
  {
    error *= params.edgesWeight;
    error = (orientationsNumber == 0) ? error : (error + orientationCost / orientationsNumber);
  }

  return error;
}

PoseQualityEstimator::PoseQualityEstimator(const Ptr<LocalPoseRefiner> &_poseRefiner, float _hTrimmedError)
{
  poseRefiner = _poseRefiner;
  hTrimmedError = _hTrimmedError;

  Mat zeros = Mat::zeros(3, 1, CV_64FC1);
  createProjectiveMatrix(zeros, zeros, Rt_init_cam);
}

void PoseQualityEstimator::setInitialPose(const PoseRT &pose_cam)
{
  poseRefiner->setInitialPose(pose_cam);
  Rt_init_cam = pose_cam.getProjectiveMatrix();
}

double PoseQualityEstimator::evaluate(const std::vector<double> &point)
{
  Mat rvec, tvec;
  vec2mats(point, rvec, tvec);

  double detOfCovarianceMatrix = 0;

  double chamferDistance = poseRefiner->estimatePoseQuality(rvec, tvec, hTrimmedError, &detOfCovarianceMatrix);
  double eps = 1e-4;
  double result = chamferDistance / (eps + sqrt(detOfCovarianceMatrix));


#ifdef VERBOSE_NLOPT
  static int iteration = 0;
  iteration++;
  cout << "Value (" << iteration << ") = " << result << " = " << chamferDistance << " / " << sqrt(detOfCovarianceMatrix) << endl;
//  cout << "regions: " << regionsQuality << endl;
#endif

  return result;
}

void PoseQualityEstimator::obj2cam(const std::vector<double> &point_obj, cv::Mat &rvec_cam, cv::Mat &tvec_cam) const
{
  Mat rvec_obj, tvec_obj;
  vec2mats(point_obj, rvec_obj, tvec_obj);

  Mat Rt_obj2cam;
  poseRefiner->getObjectCoordinateSystem(Rt_obj2cam);

  Mat transMat;
  getTransformationMatrix(Rt_obj2cam, rvec_obj, tvec_obj, transMat);

  transMat = transMat * Rt_init_cam;
  getRvecTvec(transMat, rvec_cam, tvec_cam);
}

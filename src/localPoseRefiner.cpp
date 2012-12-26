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
#include "fdcm/fdcm.hpp"

#include <opencv2/opencv.hpp>

//#define VERBOSE
//#define VISUALIZE
//#define DEBUG_ORIENTATIONS


using namespace cv;
using std::cout;
using std::endl;

LocalPoseRefiner::LocalPoseRefiner(const EdgeModel &_edgeModel, const cv::Mat &_bgrImage, const cv::Mat &_edgesImage, const PinholeCamera &camera, const LocalPoseRefinerParams &_params)
{
  verticalDirectionIndex = 2;
  dim = 3;

  params = _params;
  edgesImage = _edgesImage.clone();
  CV_Assert(!edgesImage.empty());
  camera.cameraMatrix.copyTo(cameraMatrix);
  camera.distCoeffs.copyTo(distCoeffs);
  camera.extrinsics.getProjectiveMatrix().copyTo(extrinsicsRt);

  cameraMatrix.convertTo(cameraMatrix64F, CV_64FC1);
  computeDistanceTransform(edgesImage, params.distanceType, params.distanceMask, dtImage, dtDx, dtDy);

  if (params.useEdgeOrientations)
  {
    computeDistanceTransform3D(edgesImage, surfaceDtImages);
    surfaceDtImagesDx.resize(surfaceDtImages.size());
    surfaceDtImagesDy.resize(surfaceDtImages.size());
    surfaceDtImagesDor.resize(surfaceDtImages.size());

    double orDelta = CV_PI / surfaceDtImages.size();
    for (int i = 0; i < surfaceDtImages.size(); ++i)
    {
      computeDerivatives(surfaceDtImages[i], surfaceDtImagesDx[i], surfaceDtImagesDy[i]);

      int nextIndex = (i + 1) % surfaceDtImages.size();
      int prevIndex = (static_cast<int>(surfaceDtImages.size()) + i - 1) % surfaceDtImages.size();
      surfaceDtImagesDor[i] = (surfaceDtImages[nextIndex] - surfaceDtImages[prevIndex]) / (2.0 * orDelta);
    }

  }

  originalEdgeModel = _edgeModel;
  //TODO: remove copy operation
  rotatedEdgeModel = _edgeModel;
  hasRotationSymmetry = rotatedEdgeModel.hasRotationSymmetry;

  setObjectCoordinateSystem(originalEdgeModel.Rt_obj2cam);

  bgrImage = _bgrImage.clone();
  //TODO: experiment with color gradients
  Mat grayImage;
  cvtColor(bgrImage, grayImage, CV_BGR2GRAY);
  computeDerivatives(grayImage, bgrImageDx, bgrImageDy);
}

void LocalPoseRefiner::setParams(const LocalPoseRefinerParams &_params)
{
  params = _params;
}

void LocalPoseRefiner::setSilhouetteEdges(const cv::Mat &_silhouetteEdges)
{
  silhouetteEdges = _silhouetteEdges;
  computeDistanceTransform(silhouetteEdges, params.distanceType, params.distanceMask, silhouetteDt, silhouetteDtDx, silhouetteDtDy);

  if (params.useEdgeOrientations)
  {
//    computeDistanceTransform3D(edgesImage, silhouetteDtImages);
    silhouetteDtImages = surfaceDtImages;
    silhouetteDtImagesDx = surfaceDtImagesDx;
    silhouetteDtImagesDy = surfaceDtImagesDy;

/*
    computeDistanceTransform3D(silhouetteEdges, silhouetteDtImages);
    silhouetteDtImagesDx.resize(silhouetteDtImages.size());
    silhouetteDtImagesDy.resize(silhouetteDtImages.size());

    for (int i = 0; i < silhouetteDtImages.size(); ++i)
    {
      computeDerivatives(silhouetteDtImages[i], silhouetteDtImagesDx[i], silhouetteDtImagesDy[i]);
    }
*/
  }
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

void LocalPoseRefiner::computeDerivatives(const cv::Mat &image, cv::Mat &dx, cv::Mat &dy)
{
  Mat kx_dx, ky_dx;
  int ksize=3;
  getDerivKernels(kx_dx, ky_dx, 1, 0, ksize, true);
  Mat kx_dy, ky_dy;
  getDerivKernels(kx_dy, ky_dy, 0, 1, ksize, true);

  sepFilter2D(image, dx, CV_32F, kx_dx, ky_dx);
  sepFilter2D(image, dy, CV_32F, kx_dy, ky_dy);

  CV_Assert(dx.size() == image.size());
  CV_Assert(dy.size() == image.size());
}

void LocalPoseRefiner::computeDistanceTransform(const cv::Mat &edges, int distanceType, int distanceMask, cv::Mat &distanceImage, cv::Mat &dx, cv::Mat &dy)
{
  if(edges.empty())
  {
    CV_Error(CV_HeaderIsNull, "edges are empty");
  }

  distanceTransform(~edges, distanceImage, distanceType, distanceMask);
  //TODO: experiment with different kernels
  computeDerivatives(distanceImage, dx, dy);

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
  return (pt.x < 0 || pt.y < 0 || pt.x+1 >= dtImage.cols || pt.y+1 >= dtImage.rows);
}

double LocalPoseRefiner::getFilteredDistance(cv::Point2f pt, bool useInterpolation, double outlierError, const cv::Mat &distanceTransform) const
{
  Mat dt = distanceTransform.empty() ? dtImage : distanceTransform;

  if (pt.x < 0 || pt.y < 0 || pt.x+1 >= dt.cols || pt.y+1 >= dt.rows)
    return outlierError;

  CV_Assert(dt.type() == CV_32FC1);

  double dist = useInterpolation ? getInterpolatedDT(dt, pt) : dt.at<float>(pt);
  return dist;
}

void LocalPoseRefiner::computeResiduals(const cv::Mat &projectedPoints, cv::Mat &residuals, double outlierError, const cv::Mat &distanceTransform, bool useInterpolation) const
{
  vector<int> orientationIndices(projectedPoints.rows, 0);
  vector<Mat> dtImages(1, distanceTransform);
  computeResiduals(projectedPoints, orientationIndices, dtImages, residuals, outlierError, useInterpolation);
}

void LocalPoseRefiner::computeResiduals(const cv::Mat &projectedPoints, const std::vector<int> &orientationIndices, const std::vector<cv::Mat> &dtImages,
                                        cv::Mat &residuals, double outlierError, bool useInterpolation) const
{
  CV_Assert(projectedPoints.cols == 1);
  CV_Assert(projectedPoints.type() == CV_32FC2);
  CV_Assert(orientationIndices.size() == projectedPoints.rows);

  residuals.create(projectedPoints.rows, 1, CV_64FC1);
  for(int i=0; i<projectedPoints.rows; i++)
  {
    Point2f pt2f = projectedPoints.at<Vec2f>(i);
    int orIndex = orientationIndices[i];
    residuals.at<double>(i) = getFilteredDistance(pt2f, useInterpolation, outlierError, dtImages[orIndex]);
  }
}

void LocalPoseRefiner::computeResidualsWithInliersMask(const cv::Mat &projectedPoints, cv::Mat &residuals, double outlierError, const cv::Mat &distanceTransform, bool useInterpolation, float inliersRatio, cv::Mat &inliersMask) const
{
  vector<int> orientationIndices(projectedPoints.rows, 0);
  vector<Mat> dtImages(1, distanceTransform);
  computeResidualsWithInliersMask(projectedPoints, orientationIndices, dtImages, residuals, outlierError, useInterpolation, inliersRatio, inliersMask);
}

void LocalPoseRefiner::computeResidualsWithInliersMask(const cv::Mat &projectedPoints, const std::vector<int> &orientationIndices, const std::vector<cv::Mat> &dtImages,
                                                       cv::Mat &residuals, double outlierError, bool useInterpolation, float inliersRatio, cv::Mat &inliersMask) const
{
  computeResiduals(projectedPoints, orientationIndices, dtImages, residuals, outlierError, useInterpolation);

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

void computePoseJacobian_dCam_dObj(const cv::Mat &R_obj2cam, const cv::Mat &rvec_obj, const cv::Mat &t_obj2cam, cv::Mat &J_camobj, cv::Mat &J_rodrigues)
{
  const int dim = 3;
  J_camobj.create(2 * dim, 2 * dim, CV_64FC1);
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

  Mat R_obj;
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
}

void LocalPoseRefiner::computeObjectJacobian(const cv::Mat &projectedPoints, const cv::Mat &inliersMask, const std::vector<int> &orientationIndices, const cv::Mat &error,
                                             const cv::Mat &silhouetteWeights, const cv::Mat &silhouetteWeightsJacobian,  const cv::Mat &surfaceOrientationsJacobian, const cv::Mat &JaW,
                                             const std::vector<cv::Mat> &distanceImages, const std::vector<cv::Mat> &distanceImagesDx, const std::vector<cv::Mat> &distanceImagesDy,
                                             const cv::Mat &R_obj2cam, const cv::Mat &t_obj2cam, const cv::Mat &rvec_obj, const cv::Mat &tvec_obj,
                                             cv::Mat &J) const
{
  CV_Assert(JaW.rows == 2*projectedPoints.rows);
  CV_Assert(JaW.type() == CV_64FC1);
  J.create(projectedPoints.rows, JaW.cols, CV_64FC1);

  Mat J_camobj, J_rodrigues;
  computePoseJacobian_dCam_dObj(R_obj2cam, rvec_obj, t_obj2cam, J_camobj, J_rodrigues);

  bool useOrientationsInJacobian = !surfaceOrientationsJacobian.empty();
  bool useWeightsJacobian = !silhouetteWeightsJacobian.empty();
  bool isSilhouette = !silhouetteWeights.empty();
  if (useOrientationsInJacobian)
  {
    CV_Assert(!isSilhouette && !useWeightsJacobian);
  }
  if (useWeightsJacobian)
  {
    CV_Assert(isSilhouette);
  }

  CV_Assert(inliersMask.type() == CV_8UC1);
  const float outlierJacobian = 0.0f;
  for(int i=0; i<projectedPoints.rows; i++)
  {
    if (inliersMask.at<uchar>(i) == 0)
    {
      Mat row = J.row(i);
      row.setTo(outlierJacobian);
      continue;
    }

    Point2f pt2f = projectedPoints.at<Vec2f>(i);
    if(isOutlier(pt2f))
    {
      for(int j=0; j<J.cols; j++)
      {
        J.at<double>(i, j) = outlierJacobian;
      }

      continue;
    }

    int orIndex = orientationIndices[i];
    double x = getInterpolatedDT(distanceImagesDx[orIndex], pt2f);
    double y = getInterpolatedDT(distanceImagesDy[orIndex], pt2f);

    for(int j=0; j<J.cols; j++)
    {
        double sumX = 0., sumY = 0., sumOr = 0., sumWeights = 0.;

        for(int k=0; k<J.cols; k++)
        {
          sumX += JaW.at<double>(2*i, k) * J_camobj.at<double>(k, j);
          sumY += JaW.at<double>(2*i+1, k) * J_camobj.at<double>(k, j);
          if (useOrientationsInJacobian)
          {
            sumOr += surfaceOrientationsJacobian.at<double>(i, k) * J_camobj.at<double>(k, j);
          }
          if (useWeightsJacobian)
          {
            sumWeights += silhouetteWeightsJacobian.at<double>(i, k) * J_camobj.at<double>(k, j);
          }
        }

        J.at<double>(i, j) = x * sumX + y * sumY;
        if (useOrientationsInJacobian)
        {
          J.at<double>(i, j) += surfaceDtImagesDor[orIndex].at<float>(pt2f) * sumOr;
        }
        if (isSilhouette)
        {
          J.at<double>(i, j) *= silhouetteWeights.at<double>(i);
          J.at<double>(i, j) += sumWeights * error.at<double>(i);
        }
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

double LocalPoseRefiner::normalizeError(const PoseRT &pose_cam, double error) const
{
    PinholeCamera camera(cameraMatrix, distCoeffs, extrinsicsRt);
    vector<Point2f> projectedPoints;
    camera.projectPoints(originalEdgeModel.points, pose_cam, projectedPoints);

    Mat covar, mean;
    calcCovarMatrix(Mat(projectedPoints).reshape(1), covar, mean, CV_COVAR_NORMAL | CV_COVAR_SCALE | CV_COVAR_ROWS);
    //scale errors to ~1.0 scale
    double normalizationFactor = sqrt(determinant(covar));
    const double eps = 1e-4;
    //TODO: use a smarter approach
    const double scaleFactor = 100.0;
    double normalizedError = scaleFactor * (normalizationFactor > eps ? error / normalizationFactor : error / eps);

    return normalizedError;
}

float LocalPoseRefiner::estimateOutlierError(const cv::Mat &distanceImage, int distanceType)
{
  CV_Assert(!distanceImage.empty());

  switch (distanceType)
  {
    case CV_DIST_L2:
        return sqrt(static_cast<float>(distanceImage.rows * distanceImage.rows +
                                       distanceImage.cols * distanceImage.cols));
    default:
      CV_Assert(false);
  }
}

void LocalPoseRefiner::computeWeights(const vector<Point2f> &projectedPointsVector, const cv::Mat &silhouetteEdges, cv::Mat &weights) const
{
  for (size_t i = 0; i < projectedPointsVector.size(); ++i)
  {
    Point2f pt = projectedPointsVector[i];
    CV_Assert(!cvIsNaN(pt.x));
    CV_Assert(!cvIsNaN(pt.y));
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
  distanceTransform(footprintImage, footprintDT, params.distanceType, params.distanceMask);

  float outlierError = estimateOutlierError(footprintDT, params.distanceType);
  for (int i = 0; i < weights.rows; ++i)
  {
    Point projectedPt = projectedPointsVector[i];
    Point pt = params.lmDownFactor * projectedPt - tl;

    CV_Assert(footprintDT.type() == CV_32FC1);
    float dist = isPointInside(footprintDT, pt) ? footprintDT.at<float>(pt) : outlierError;
    //cout << dist << endl;

    weights.at<double>(i) = 2 * exp(-dist / params.lmDownFactor);
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

void reduceJacobianToNewBasis(const cv::Mat &newTranslationBasis2old, int paramsCount, bool hasRotationSymmetry, int verticalDirectionIndex,
                              cv::Mat &J)
{
  CV_Assert(!J.empty());
  const int dim = 3;
  Mat newJ(J.rows, paramsCount, J.type());
  if (!hasRotationSymmetry)
  {
    CV_Assert(verticalDirectionIndex < J.cols);
    Mat rotationJ = J.colRange(verticalDirectionIndex, verticalDirectionIndex + 1);
    Mat theFirstCol = newJ.col(0);
    rotationJ.copyTo(theFirstCol);
  }
  Mat translationJ = J.colRange(dim, 2*dim) * newTranslationBasis2old;
  Mat lastCols = newJ.colRange(paramsCount - newTranslationBasis2old.cols, paramsCount);
  translationJ.copyTo(lastCols);
  J = newJ;
}

//optional jacobian is derivative of orientation with regard to 6 pose parameters
void computeProjectedOrientations(const std::vector<Point3f> &edgels, const std::vector<Point3f> &tangentVectors, const PoseRT &pose_cam, const cv::Mat &cameraMatrix,
                                  std::vector<float> &orientations, cv::Mat *J_rodrigues = 0, cv::Mat *jacobian = 0)
{
  CV_Assert(edgels.size() == tangentVectors.size());
  Mat RtParams_cam = pose_cam.getProjectiveMatrix();
  Mat P = RtParams_cam(Rect(0, 0, 4, 3));
  Mat P_rot = RtParams_cam(Rect(0, 0, 4, 3)).clone();
  P_rot.col(3).setTo(0);

  Mat rotatedPoints, rotatedOrientations;
  transform(Mat(edgels), rotatedPoints, P);
  transform(Mat(tangentVectors), rotatedOrientations, P_rot);

  CV_Assert(rotatedPoints.type() == CV_32FC3);
  CV_Assert(rotatedOrientations.type() == CV_32FC3);
  CV_Assert(rotatedPoints.size() == rotatedOrientations.size());
  CV_Assert(cameraMatrix.type() == CV_64FC1);

  vector<Mat> transformedPoints, transformedOrientations;
  const int dim = 3;
  if (jacobian != 0)
  {
    jacobian->create(edgels.size(), 2*dim, CV_64FC1);
    for (int axisIndex = 0; axisIndex < dim; ++axisIndex)
    {
      Mat R = J_rodrigues->row(axisIndex).reshape(1, 3);
      Mat currentTransformedPoints, currentTransformedOrientations;
      transform(Mat(edgels), currentTransformedPoints, R);
      transform(Mat(tangentVectors), currentTransformedOrientations, R);
      transformedPoints.push_back(currentTransformedPoints);
      transformedOrientations.push_back(currentTransformedOrientations);
    }
  }

  double fx = cameraMatrix.at<double>(0, 0);
  double fy = cameraMatrix.at<double>(1, 1);
  orientations.clear();
  //TODO: take distortion into account
  for (int i = 0; i < rotatedOrientations.rows; ++i)
  {
    Vec3f pt = rotatedPoints.at<Vec3f>(i);
    Vec3f ort = rotatedOrientations.at<Vec3f>(i);

    //double dx = fx * (ort[0] * pt[2] - pt[0] * ort[2]) / (pt[2] * pt[2]);
    //double dy = fy * (ort[1] * pt[2] - pt[1] * ort[2]) / (pt[2] * pt[2]);
    //you need only orientation so you can ignore denominator
    float dx = fx * (ort[0] * pt[2] - pt[0] * ort[2]);
    float dy = fy * (ort[1] * pt[2] - pt[1] * ort[2]);

    //TODO: use -dy?
    //float theta = atan2(-dy, dx);
    float theta = atan2(dy, dx);

    while (theta < 0)
    {
      theta += CV_PI;
    }
    orientations.push_back(theta);

    if (jacobian != 0)
    {
      double dorFactor = 1.0 / (dx*dx + dy*dy);

      for (int axisIndex = 0; axisIndex < dim; ++axisIndex)
      {
        Vec3f dpt = transformedPoints[axisIndex].at<Vec3f>(i);
        Vec3f dort = transformedOrientations[axisIndex].at<Vec3f>(i);
        double dor = dorFactor * (dx * (dpt[2] * ort[1] + pt[2] * dort[1] - dpt[1] * ort[2] - pt[1] * dort[2])
                                 -dy * (dpt[2] * ort[0] + pt[2] * dort[0] - dpt[0] * ort[2] - pt[0] * dort[2]));
        jacobian->at<double>(i, axisIndex) = dor;
      }

      jacobian->at<double>(i, 3) = dorFactor * dy * ort[2];
      jacobian->at<double>(i, 4) = -dorFactor * dx * ort[2];
      jacobian->at<double>(i, 5) = dorFactor * (dx * ort[1] - dy * ort[0]);
    }
  }
}

void LocalPoseRefiner::computeLMIterationData(int paramsCount, bool isSilhouette,
       const cv::Mat R_obj2cam, const cv::Mat &t_obj2cam, bool computeJacobian,
       const cv::Mat &newTranslationBasis2old, const cv::Mat &rvecParams, const cv::Mat &tvecParams,
       cv::Mat &rvecParams_cam, cv::Mat &tvecParams_cam, cv::Mat &RtParams_cam,
       cv::Mat &J, cv::Mat &error)
{
  //TODO: move up
  const int defaultOrIndex = 0;
  const int directionsCount = 60;

  const vector<Point3f> &edgels = isSilhouette ? rotatedEdgeModel.points : rotatedEdgeModel.stableEdgels;
  const Mat dt = isSilhouette ? silhouetteDt : dtImage;
  const Mat dx = isSilhouette ? silhouetteDtDx : dtDx;
  const Mat dy = isSilhouette ? silhouetteDtDy : dtDy;

  vector<Mat> distanceImages, distanceImagesDx, distanceImagesDy;
  if (params.useEdgeOrientations)
  {
    distanceImages = isSilhouette ? silhouetteDtImages : surfaceDtImages;
    distanceImagesDx = isSilhouette ? silhouetteDtImagesDx : surfaceDtImagesDx;
    distanceImagesDy = isSilhouette ? silhouetteDtImagesDy : surfaceDtImagesDy;
  }
  else
  {
    distanceImages.resize(1, dt);
    distanceImagesDx.resize(1, dx);
    distanceImagesDy.resize(1, dy);
  }
  vector<int> orientationIndices(edgels.size(), 0);

  vector<Point2f> projectedPointsVector;
  Mat JaW;
  if (computeJacobian)
  {
    JaW.create(2 * edgels.size(), 2 * dim, CV_64F);
    Mat Dpdrot = JaW.colRange(0, this->dim);
    Mat Dpdt = JaW.colRange(this->dim, 2 * this->dim);
    projectPoints_obj(Mat(edgels), rvecParams, tvecParams, rvecParams_cam, tvecParams_cam, RtParams_cam,
                      projectedPointsVector, &Dpdrot, &Dpdt);
  }
  else
  {
    projectPoints_obj(Mat(edgels), rvecParams, tvecParams, rvecParams_cam, tvecParams_cam, RtParams_cam,
                      projectedPointsVector);
  }
  PoseRT pose_cam(RtParams_cam);

  Mat projectedPoints = Mat(projectedPointsVector);
  Mat inliersMask;
  float outlierError = estimateOutlierError(dt, params.distanceType);

  Mat silhouetteWeights, silhouetteWeightsJacobian;
  Mat surfaceOrientationsJacobian;
  if (isSilhouette)
  {
    rotatedEdgeModel.computeWeights(PoseRT(rvecParams_cam, tvecParams_cam), params.decayConstant, params.maxWeight, silhouetteWeights, &silhouetteWeightsJacobian);
    CV_Assert(silhouetteWeights.rows == edgels.size());
    CV_Assert(silhouetteWeights.cols == 1);
    CV_Assert(silhouetteWeights.type() == CV_64FC1);

#ifdef VISUALIZE
      Mat weightsImage(edgesImage.size(), CV_8UC1, Scalar(0));
      for (size_t i = 0; i < projectedPointsVector.size(); ++i)
      {
        if (silhouetteWeights.at<double>(i) > 0.1)
        {
//          double intensity = std::min(255.0, 1000 * silhouetteWeights.at<double>(i));
 //         circle(weightsImage, projectedPointsVector[i], 0, Scalar(intensity));
          circle(weightsImage, projectedPointsVector[i], 0, Scalar(255.0));
        }
      }
      imshow("weights", weightsImage);
      waitKey();
#endif

    Mat silhouettePointsMask = silhouetteWeights > params.minSilhouetteWeight;
    vector<Point2f> silhouettePointsVec;
    vector<int> silhouettePointsIndices;

    for (size_t i = 0; i < edgels.size(); ++i)
    {
      if (silhouettePointsMask.at<uchar>(i) == 255)
      {
        silhouettePointsVec.push_back(projectedPointsVector[i]);
        silhouettePointsIndices.push_back(i);
      }
    }

    Mat silhouettePointsError, silhouettePointsInliersMask;
    vector<int> silhouetteOrientationIndices;
    if (!params.useEdgeOrientations)
    {
      computeResidualsWithInliersMask(Mat(silhouettePointsVec), silhouettePointsError, outlierError, dt, true, this->params.lmInliersRatio, silhouettePointsInliersMask);
      silhouetteOrientationIndices.resize(silhouettePointsVec.size(), 0);
    }
    else
    {
      vector<float> projectedOrientations;
      //TODO: pass silhouettePointsMask to this function
      computeProjectedOrientations(edgels, rotatedEdgeModel.normals, pose_cam, cameraMatrix, projectedOrientations);
      for (size_t i = 0; i < projectedOrientations.size(); ++i)
      {
        if (silhouettePointsMask.at<uchar>(i) != 255)
        {
          continue;
        }

        float theta = projectedOrientations[i] - (CV_PI / 2.0);
        while (theta < 0)
        {
          theta += CV_PI;
        }

        int orIndex = theta2Index(theta, directionsCount);
        silhouetteOrientationIndices.push_back(orIndex);
      }

      computeResidualsWithInliersMask(Mat(silhouettePointsVec), silhouetteOrientationIndices, silhouetteDtImages, silhouettePointsError, outlierError, true, this->params.lmInliersRatio, silhouettePointsInliersMask);
    }

    error.create(edgels.size(), 1, CV_64FC1);
    error = Scalar(0.0);
    inliersMask.create(edgels.size(), 1, CV_8UC1);
    inliersMask = Scalar(0);

    for (int i = 0; i < silhouettePointsError.rows; ++i)
    {
      if (silhouettePointsInliersMask.at<uchar>(i))
      {
        int ptIndex = silhouettePointsIndices[i];
        error.at<double>(ptIndex) = silhouettePointsError.at<double>(i);
        inliersMask.at<uchar>(ptIndex) = 255;

        orientationIndices[ptIndex] = silhouetteOrientationIndices[i];
      }
    }
  }
  else
  {
    if (params.useEdgeOrientations)
    {
      vector<float> projectedOrientations;

      Mat R_obj, J_rodrigues;
      Rodrigues(rvecParams, R_obj, J_rodrigues);
      CV_Assert(J_rodrigues.rows == 3 && J_rodrigues.cols == 9);
      computeProjectedOrientations(rotatedEdgeModel.stableEdgels, rotatedEdgeModel.orientations, pose_cam, cameraMatrix, projectedOrientations, &J_rodrigues, &surfaceOrientationsJacobian);
      for (size_t i = 0; i < projectedOrientations.size(); ++i)
      {
        orientationIndices[i] = theta2Index(projectedOrientations[i], directionsCount);
      }
    }

    computeResidualsWithInliersMask(projectedPoints, orientationIndices, distanceImages, error, outlierError, true, this->params.lmInliersRatio, inliersMask);
  }
  error.setTo(0, ~inliersMask);

  if (computeJacobian)
  {
    computeObjectJacobian(projectedPoints, inliersMask, orientationIndices, error, silhouetteWeights, silhouetteWeightsJacobian, surfaceOrientationsJacobian, JaW, distanceImages, distanceImagesDx, distanceImagesDy, R_obj2cam, t_obj2cam, rvecParams, tvecParams, J);

    if (!newTranslationBasis2old.empty())
    {
      reduceJacobianToNewBasis(newTranslationBasis2old, paramsCount, hasRotationSymmetry, verticalDirectionIndex, J);
      if (isSilhouette)
      {
        reduceJacobianToNewBasis(newTranslationBasis2old, paramsCount, hasRotationSymmetry, verticalDirectionIndex, silhouetteWeightsJacobian);
      }
    }
  }

  if (isSilhouette)
  {
    CV_Assert(silhouetteWeights.type() == error.type());
    Mat mulError = silhouetteWeights.mul(error);
    error = mulError;

/*
    CV_Assert(error.type() == CV_64FC1);

    Mat errorViz;
    cvtColor(silhouetteEdges, errorViz, CV_GRAY2RGB);
    for (size_t i = 0; i < projectedPointsVector.size(); ++i)
    {
      if (inliersMask.at<uchar>(i) != 0)
      {
        Point pt = projectedPointsVector[i];
        errorViz.at<Vec3b>(pt) = Vec3b(0, 255, 0);
      }
//      Point pt = projectedPointsVector[i];
//      errorViz.at<Vec3b>(pt) = Vec3b(0, error.at<double>(i) * 25, 0);
//      circle(errorViz, pt, 2, Scalar(0, error.at<double>(i) * 15, 0), -1);
    }
    imshow("errorViz", errorViz);
    waitKey();
    exit(-1);
*/
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

  CvLevMarq solver(paramsCount, residualsCount, this->params.termCriteria);
  CvMat paramsCvMat = params;
  cvCopy( &paramsCvMat, solver.param );
  //params and solver.params must use the same memory

  Mat err;
  int iter = 0;
  float finishError = -1.f;
#ifdef VERBOSE
  float startError = -1.f;
#endif

  //TODO: number of iterations is greater in 2 times than needed (empty iterations)
  while (true)
  {
    //cout << "Params: " << params << endl;
    //cout << "Iteration: " << iter << endl;

    CvMat *matJ = 0, *_err = 0;
    const CvMat *__param = 0;
    bool proceed = solver.update( __param, matJ, _err );
    //if( iter != 1 )
      //proceed = solver.update( __param, matJ, _err );
    cvCopy( __param, &paramsCvMat );
    bool isDone = !proceed || !_err;
    if( isDone && !finalJacobian )
        break;

    Mat surfaceJ, surfaceErr;
    Mat silhouetteJ, silhouetteErr;

    bool computeJacobian = (solver.state == CvLevMarq::CALC_J) || (isDone && (finalJacobian != 0));
    CV_Assert((solver.state == CvLevMarq::CALC_J) == (matJ != 0));

    if (!newTranslationBasis2old.empty())
    {
      Mat translationPart = params.rowRange(paramsCount - newTranslationBasis2old.cols, paramsCount);
      tvecParams = newTranslationBasis2old * translationPart;
      CV_Assert(tvecParams.rows == dim);
      CV_Assert(tvecParams.cols == 1);
    }

    Mat rvecParams_cam, tvecParams_cam, RtParams_cam;
    computeLMIterationData(paramsCount, false, R_obj2cam, t_obj2cam, computeJacobian,
                           newTranslationBasis2old, rvecParams, tvecParams,
                           rvecParams_cam, tvecParams_cam, RtParams_cam, surfaceJ, surfaceErr);

    computeLMIterationData(paramsCount, true, R_obj2cam, t_obj2cam, computeJacobian,
                           newTranslationBasis2old, rvecParams, tvecParams,
                           rvecParams_cam, tvecParams_cam, RtParams_cam, silhouetteJ, silhouetteErr);

    if (silhouetteErr.empty())
    {
      return std::numeric_limits<float>::max();
    }

    //TODO: move up
//    surfaceJ *= 5;
//    surfaceErr *= 5;

    if (computeJacobian)
    {
      Mat J = surfaceJ.clone();
      J.push_back(silhouetteJ);

      CvMat JcvMat = J;
      //cvCopy(&JcvMat, matJ);
      cvCopy(&JcvMat, solver.J);
    }

    if (isDone)
    {
      break;
    }

    err = surfaceErr.clone();
    err.push_back(silhouetteErr);

#ifdef VERBOSE
    cout << "Errors:" << endl;
    cout << "  surface: " << getError(surfaceErr) << endl;
    cout << "  silhouette: " << getError(silhouetteErr) << endl;

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

#endif

#ifdef VISUALIZE
    displayProjection(projectedPoints, "points");
    displayProjection(projectedStableEdgels, "surface points");
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

  finishError = normalizeError(pose_cam, finishError);

  return finishError;
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

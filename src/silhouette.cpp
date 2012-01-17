/*
 * silhouette.cpp
 *
 *  Created on: Oct 17, 2011
 *      Author: Ilya Lysenkov
 */

#include "edges_pose_refiner/silhouette.hpp"
#include "edges_pose_refiner/utils.hpp"
#include "edges_pose_refiner/edgeModel.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using std::cout;
using std::endl;

//#define VISUALIZE_TRANSFORMS

Mat affine2homography(const cv::Mat &transformationMatrix)
{
  const Size affineTransformationSize(3, 2);
  const Size homographyTransformationSize(3, 3);

  CV_Assert(transformationMatrix.size() == affineTransformationSize);

  Mat homography = Mat::eye(homographyTransformationSize, transformationMatrix.type());
  Mat affinePart = homography.rowRange(0, 2);
  transformationMatrix.copyTo(affinePart);
  return homography;
}

void composeAffineTransformations(const Mat &transformation1, const Mat &transformation2, Mat &composedTransformation)
{
  CV_Assert(transformation1.type() == transformation2.type());

  Mat homography1 = affine2homography(transformation1);
  Mat homography2 = affine2homography(transformation2);
  Mat composedHomography = homography2 * homography1;

  composedTransformation = composedHomography.rowRange(0, 2).clone();
}

Silhouette::Silhouette()
{
}

void Silhouette::init(const cv::Mat &_edgels, const PoseRT &_initialPose_cam)
{
  edgels = _edgels;
  initialPose_cam = _initialPose_cam;

  getNormalizationTransform(edgels, silhouette2normalized);
}

void Silhouette::getEdgels(cv::Mat &_edgels) const
{
  _edgels = edgels;
}

void Silhouette::getInitialPose(PoseRT &pose_cam) const
{
  pose_cam = initialPose_cam;
}

void Silhouette::clear()
{
  edgels = Mat();
  silhouette2normalized = Mat();
}

void Silhouette::affine2poseRT(const EdgeModel &edgeModel, const PinholeCamera &camera, const cv::Mat &affineTransformation, bool useClosedFormPnP, PoseRT &pose_cam) const
{
  PoseRT poseWithExtrinsics_cam;
  if (useClosedFormPnP)
  {
    Mat homography = affine2homography(affineTransformation);
    //TODO: which inversion method is better?
    Mat fullTransform = camera.cameraMatrix.inv() * homography * camera.cameraMatrix;
    const int dim = 3;
    CV_Assert(fullTransform.rows == dim && fullTransform.cols == dim);
    CV_Assert(fullTransform.type() == CV_64FC1);
    Mat rotationalComponentWithScale = fullTransform(Range(0, 2), Range(0, 2));
    double det = determinant(rotationalComponentWithScale);
    const float eps = 1e-6;
    CV_Assert(det > eps);
    double scale = 1.0 / sqrt(det);

    Point3d objectCenter = edgeModel.getObjectCenter();
    Point3d initialObjectCenter;
    transformPoint(initialPose_cam.getProjectiveMatrix(), objectCenter, initialObjectCenter);
    double meanZ = initialObjectCenter.z;
    CV_Assert(meanZ > eps);

    Point3d tvec;
    tvec.z = (scale - 1.0) * meanZ;
    tvec.x = fullTransform.at<double>(0, 2) * (scale * meanZ);
    tvec.y = fullTransform.at<double>(1, 2) * (scale * meanZ);

    Mat R = Mat::eye(dim, dim, fullTransform.type());
    Mat rotation2d = R(Range(0, 2), Range(0, 2));
    Mat pureRotationalComponent = rotationalComponentWithScale * scale;
    pureRotationalComponent.copyTo(rotation2d);

    Mat tvecMat;
    point2col(tvec, tvecMat);
    PoseRT pose2d_cam(R, tvecMat);

    poseWithExtrinsics_cam = pose2d_cam * initialPose_cam;
  //  pose_cam = camera.extrinsics.inv() * pose2d_cam * initialPose_cam;
  }

  vector<Point2f> projectedObjectPoints;
  camera.projectPoints(edgeModel.points, initialPose_cam, projectedObjectPoints);

  Mat transformedObjectPoints;
  transform(Mat(projectedObjectPoints), transformedObjectPoints, affineTransformation);

  solvePnP(Mat(edgeModel.points), transformedObjectPoints, camera.cameraMatrix, camera.distCoeffs, poseWithExtrinsics_cam.rvec, poseWithExtrinsics_cam.tvec, useClosedFormPnP);
  pose_cam = camera.extrinsics.inv() * poseWithExtrinsics_cam;
}

void Silhouette::draw(cv::Mat &image, int thickness) const
{
  CV_Assert(image.type() == CV_8UC1);
  vector<Point2f> edgelsVec = edgels;
  for (size_t i = 0; i < edgelsVec.size(); ++i)
  {
    Point pt = edgelsVec[i];
    CV_Assert(isPointInside(image, pt));

    circle(image, pt, 1, Scalar(255), thickness);
  }
}

void Silhouette::getNormalizationTransform(const cv::Mat &points, cv::Mat &normalizationTransform)
{
  if (points.empty())
  {
    normalizationTransform = Mat();
    return;
  }

  CV_Assert(points.type() == CV_32FC2);

  Scalar mean, stddev;
  meanStdDev(points, mean, stddev);
  double tx = -mean[0];
  double ty = -mean[1];
  double scale = 1.0 / sqrt(stddev[0] * stddev[0] + stddev[1] * stddev[1]);

  normalizationTransform = scale * (Mat_<double>(2, 3) <<
                  1.0, 0.0, tx,
                  0.0, 1.0, ty);
}

float estimateScale(const Mat &src, const Mat &transformationMatrix)
{
  Mat transformedPoints;
  transform(src, transformedPoints, transformationMatrix);
  Mat covar, mean;
  calcCovarMatrix(transformedPoints.reshape(1), covar, mean, CV_COVAR_NORMAL + CV_COVAR_SCALE + CV_COVAR_ROWS);
  return determinant(covar);
}

void Silhouette::findSimilarityTransformation(const cv::Mat &src, const cv::Mat &dst, Mat &transformationMatrix, int iterationsCount, float min2dScaleChange)
{
  CV_Assert(src.type() == CV_32FC2);
  CV_Assert(dst.type() == CV_32FC2);
  CV_Assert(!transformationMatrix.empty());

  //Ptr<cv::flann::IndexParams> flannIndexParams = new cv::flann::KDTreeIndexParams();
  Ptr<cv::flann::IndexParams> flannIndexParams = new cv::flann::LinearIndexParams();
  cv::flann::Index flannIndex(dst.reshape(1), *flannIndexParams);

  Mat srcTransformationMatrix = transformationMatrix.clone();

  for (int iter = 0; iter < iterationsCount; ++iter)
  {
    Mat transformedPoints;
    transform(src, transformedPoints, transformationMatrix);
    Mat srcTwoChannels = transformedPoints;
    Mat srcOneChannel = srcTwoChannels.reshape(1);

    Mat correspondingPoints = Mat(srcTwoChannels.size(), srcTwoChannels.type());
    for (int i = 0; i < src.rows; ++i)
    {
      vector<float> query = srcOneChannel.row(i);
      int knn = 1;
      vector<int> indices(knn);
      vector<float> dists(knn);
      flannIndex.knnSearch(query, indices, dists, knn, cv::flann::SearchParams());
      Mat row = correspondingPoints.row(i);
      dst.row(indices[0]).copyTo(row);
    }

    const bool isFullAffine = false;
    CV_Assert(srcTwoChannels.channels() == 2);
    CV_Assert(correspondingPoints.channels() == 2);
    CV_Assert(srcTwoChannels.type() == correspondingPoints.type());
    Mat currentTransformationMatrix = Mat::zeros(2, 3, CV_64FC1);
    //currentTransformationMatrix = estimateRigidTransform(srcTwoChannels, correspondingPoints, isFullAffine);

    CvMat matA = srcTwoChannels, matB = correspondingPoints, matM = currentTransformationMatrix;
    bool isTransformEstimated = cvEstimateRigidTransform(&matA, &matB, &matM, isFullAffine );
    if (!isTransformEstimated)
    {
      break;
    }

    Mat composedTransformation;
    composeAffineTransformations(transformationMatrix, currentTransformationMatrix, composedTransformation);
    transformationMatrix = composedTransformation;
  }

  float srcScale = estimateScale(src, srcTransformationMatrix);
  float finalScale = estimateScale(src, transformationMatrix);
  const float eps = 1e-06;
  if (srcScale > eps)
  {
    float scaleChange = finalScale / srcScale;
    if (scaleChange < min2dScaleChange)
    {
      transformationMatrix = srcTransformationMatrix;
    }
  }
}


void Silhouette::showNormalizedPoints(const cv::Mat &points, const std::string &title)
{
  Mat image(480, 640, CV_8UC1, Scalar(0));
  //Mat pointsMat = 200 + 100 * points;
  Mat pointsMat = points;
  vector<Point2f> modelEdgels = pointsMat;

  for (size_t i = 0; i < modelEdgels.size(); ++i)
  {
    Point pt = modelEdgels[i];
    //CV_Assert(isPointInside(image, pt));
    if(isPointInside(image, pt))
    {
      image.at<uchar>(pt) = 255;
    }
  }
  imshow(title, image);
}

void Silhouette::match(const cv::Mat &inputEdgels, cv::Mat &silhouette2test, int icpIterationsCount, float min2dScaleChange) const
{
  Mat testEdgels;
  if (inputEdgels.type() != CV_32FC2)
  {
    inputEdgels.convertTo(testEdgels, CV_32FC2);
  }
  else
  {
    testEdgels = inputEdgels;
  }
  Mat test2normalized;
  getNormalizationTransform(testEdgels, test2normalized);

  Mat normalized2test;
  invertAffineTransform(test2normalized, normalized2test);

  Mat transformationMatrix;
  composeAffineTransformations(silhouette2normalized, normalized2test, transformationMatrix);

#ifdef VISUALIZE_TRANSFORMS
  cout << "initial transformation: " << endl << transformationMatrix << endl;
  Mat initialTransformedModel;
  transform(edgels, initialTransformedModel, transformationMatrix);
  showNormalizedPoints(initialTransformedModel, "initial transformed model");
#endif

  findSimilarityTransformation(edgels, testEdgels, transformationMatrix, icpIterationsCount, min2dScaleChange);

#ifdef VISUALIZE_TRANSFORMS
  cout << "final: " << endl << transformationMatrix << endl;
//  cout << testEdgelsMat << endl;
  showNormalizedPoints(testEdgels, "test");
  showNormalizedPoints(edgels, "model");

  Mat transformedModel;
  transform(edgels, transformedModel, transformationMatrix);
  showNormalizedPoints(transformedModel, "transformed model");

  //TODO: remove fixed resolution
  Mat initImage(480, 640, CV_8UC3, Scalar(0));
  vector<Point2f> testEdgelsVec = testEdgels;
  drawPoints(testEdgelsVec, initImage, Scalar(0, 255, 0), 3);
  vector<Point2f> modelEdgelsVec = initialTransformedModel;
  drawPoints(modelEdgelsVec, initImage, Scalar(0, 0, 255), 3);

  Mat finalImage(480, 640, CV_8UC3, Scalar(0));
  drawPoints(testEdgelsVec, finalImage, Scalar(0, 255, 0), 3);
  vector<Point2f> finalModelEdgelsVec = transformedModel;
  drawPoints(finalModelEdgelsVec, finalImage, Scalar(0, 0, 255), 3);

  Mat testImage(480, 640, CV_8UC3, Scalar(0));
  drawPoints(testEdgelsVec, testImage, Scalar(0, 255, 0), 3);
  vector<Point2f> templateEdgelsVec = edgels;
  drawPoints(templateEdgelsVec, testImage, Scalar(0, 0, 255), 3);

  imshow("initTransformation", initImage);
  imshow("finalTransformation", finalImage);
  imshow("testEdgels", testImage);

  waitKey();
#endif

  silhouette2test = transformationMatrix;
}

void Silhouette::read(const cv::FileNode &fn)
{
  fn["edgels"] >> edgels;
  fn["silhouette2normalized"] >> silhouette2normalized;

  initialPose_cam.read(fn);
}

void Silhouette::write(cv::FileStorage &fs) const
{
  fs << "edgels" << edgels;
  fs << "silhouette2normalized" << silhouette2normalized;

  initialPose_cam.write(fs);
}

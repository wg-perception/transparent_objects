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

cv::Mat affine2homography(const cv::Mat &transformationMatrix)
{
  const Size affineTransformationSize(3, 2);
  const Size homographyTransformationSize(3, 3);

  CV_Assert(transformationMatrix.size() == affineTransformationSize);

  Mat homography = Mat::eye(homographyTransformationSize, transformationMatrix.type());
  Mat affinePart = homography.rowRange(0, 2);
  transformationMatrix.copyTo(affinePart);
  return homography;
}

cv::Mat homography2affine(const cv::Mat &homography)
{
  return homography.rowRange(0, 2).clone();
}

void composeAffineTransformations(const Mat &firstTransformation, const Mat &secondTransformation, Mat &composedTransformation)
{
  CV_Assert(firstTransformation.type() == secondTransformation.type());

  Mat firstHomography = affine2homography(firstTransformation);
  Mat secondHomography = affine2homography(secondTransformation);
  Mat composedHomography = secondHomography * firstHomography;

  composedTransformation = homography2affine(composedHomography);
}

Silhouette::Silhouette()
{
}

void findSimilarityTransformation(const cv::Point2f &pt1, const cv::Point2f &pt2, cv::Mat &similarityTransformation)
{
//  cout << pt1 << " " << pt2 << endl;
  Point2f diff = pt2 - pt1;
  float distance = norm(diff);
  const float eps = 1e-4;
  CV_Assert(distance > eps);
  float cosAngle = diff.x / distance;
  float sinAngle = diff.y / distance;

  float cosRotationAngle = cosAngle;
  float sinRotationAngle = -sinAngle;
  Mat rotationMatrix = (Mat_<float>(2, 3) << cosRotationAngle, -sinRotationAngle, 0.0,
                                             sinRotationAngle, cosRotationAngle, 0.0);

  Point2f translation = -0.5 * (pt1 + pt2);
  Mat translationMatrix = (Mat_<float>(2, 3) << 1.0, 0.0, translation.x,
                                                0.0, 1.0, translation.y);
  Mat euclideanMatrix;
  composeAffineTransformations(translationMatrix, rotationMatrix, euclideanMatrix);

  float scale = 1.0 / distance;
  similarityTransformation = scale * euclideanMatrix;
}

void Silhouette::generateHashForBasis(int firstIndex, int secondIndex, cv::Mat &transformedEdgels)
{
  CV_Assert(firstIndex != secondIndex);
  CV_Assert(downsampledEdgels.type() == CV_32FC2);

  vector<Point2f> edgelsVec = downsampledEdgels;
  CV_Assert(0 <= firstIndex && firstIndex < edgelsVec.size());
  CV_Assert(0 <= secondIndex && secondIndex < edgelsVec.size());

  Mat similarityTransformation;
  ::findSimilarityTransformation(edgelsVec[firstIndex], edgelsVec[secondIndex], similarityTransformation);

  transform(downsampledEdgels, transformedEdgels, similarityTransformation);

  const Vec2f firstPoint(-0.5f, 0.0f);
  const Vec2f secondPoint(0.5f, 0.0f);
  const float eps = 1e-3;
  CV_Assert(norm(transformedEdgels.at<Vec2f>(firstIndex) - firstPoint) < eps);
  CV_Assert(norm(transformedEdgels.at<Vec2f>(secondIndex) - secondPoint) < eps);

//  Mat xChannel = transformedEdgels.reshape(1).col(0);
//  Mat yChannel = transformedEdgels.reshape(1).col(1);
//  double min_x, min_y, max_x, max_y;
//  minMaxLoc(xChannel, &min_x, &max_x);
//  minMaxLoc(yChannel, &min_y, &max_y);
//  cout << min_x << " " << min_y << " " << max_x << " " << max_y << endl;

/*
  float granularity = 10.0;
  for (int i = 0; i < transformedEdgels.size(); ++i)
  {
    Point pt = transformedEdgels[i] * granularity;
  }
*/
}

void Silhouette::generateGeometricHash(int silhouetteIndex, GHTable &hashTable, cv::Mat &canonicScale, float granularity, int hashBasisStep, float minDistanceBetweenPoints)
{
  vector<Point2f> edgelsVec = edgels;
  vector<Point2f> downsampledEdgelsVec;
  for (int i = 0; i < edgels.rows; i += hashBasisStep)
  {
    downsampledEdgelsVec.push_back(edgelsVec[i]);
  }
  downsampledEdgels = Mat(downsampledEdgelsVec).clone();

  canonicScale.create(downsampledEdgels.rows, downsampledEdgels.rows, CV_32FC1);
  for (int i = 0; i < downsampledEdgels.rows; ++i)
  {
    for (int j = i; j < downsampledEdgels.rows; ++j)
    {
      float dist = norm(downsampledEdgelsVec[i] - downsampledEdgelsVec[j]);
      float invDist = 1.0f;
      if (dist > minDistanceBetweenPoints)
      {
        invDist = 1.0f / dist;
      }

      canonicScale.at<float>(i, j) = invDist;
      canonicScale.at<float>(j, i) = invDist;
    }
  }

  for (int firstIndex = 0; firstIndex < downsampledEdgels.rows; ++firstIndex)
  {
    //TODO: use symmetry (i, j) and (j, i)
    for (int secondIndex = firstIndex + 1; secondIndex < downsampledEdgels.rows; ++secondIndex)
    {
      float dist = norm(downsampledEdgelsVec[firstIndex] - downsampledEdgelsVec[secondIndex]);
      if (dist < minDistanceBetweenPoints)
      {
        continue;
      }

      GHValue basisIndices(silhouetteIndex, firstIndex, secondIndex);
      GHValue invertedBasisIndices(silhouetteIndex, secondIndex, firstIndex);
      Mat transformedEdgels;
      generateHashForBasis(firstIndex, secondIndex, transformedEdgels);
      vector<Point2f> transformedEdgelsVec = transformedEdgels;
      for (int i = 0; i < transformedEdgelsVec.size(); ++i)
      {
        if (i == firstIndex || i == secondIndex)
        {
          continue;
        }
        float invertedGranularity = 1.0 / granularity;
        Point pt = transformedEdgelsVec[i] * invertedGranularity;
//        cout << pt << endl;

        GHKey ptPair(pt.x, pt.y);
        std::pair<GHKey, GHValue> value(ptPair, basisIndices);
        hashTable.insert(value);

        GHKey invertedPtPair(-pt.x, -pt.y);
        std::pair<GHKey, GHValue> invertedValue(invertedPtPair, invertedBasisIndices);
        hashTable.insert(invertedValue);
      }
    }
  }
}

void Silhouette::init(const cv::Mat &_edgels, const PoseRT &_initialPose_cam)
{
  edgels = _edgels;
  initialPose_cam = _initialPose_cam;

  CV_Assert(edgels.channels() == 2);
  Scalar center = mean(edgels);
  silhouetteCenter = Point2f(center[0], center[1]);

  getNormalizationTransform(edgels, silhouette2normalized);
}

void Silhouette::getEdgels(cv::Mat &_edgels) const
{
  _edgels = edgels;
}

void Silhouette::getDownsampledEdgels(cv::Mat &_edgels) const
{
  _edgels = downsampledEdgels;
}

void Silhouette::getInitialPose(PoseRT &pose_cam) const
{
  pose_cam = initialPose_cam;
}

int Silhouette::size() const
{
  CV_Assert(!edgels.empty());
  return edgels.rows;
}

int Silhouette::getDownsampledSize() const
{
  CV_Assert(!downsampledEdgels.empty());
  return downsampledEdgels.rows;
}

void Silhouette::clear()
{
  edgels = Mat();
  silhouette2normalized = Mat();
}

//TODO: undistort and scale the image
void Silhouette::affine2poseRT(const EdgeModel &edgeModel, const PinholeCamera &camera, const cv::Mat &affineTransformation, bool useClosedFormPnP, PoseRT &pose_cam) const
{
  PoseRT poseWithExtrinsics_cam;
  if (useClosedFormPnP)
  {
    CV_Assert(camera.cameraMatrix.type() == CV_64FC1);
    const float eps = 1e-6;
//    CV_Assert(fabs(camera.cameraMatrix.at<double>(0, 0) - camera.cameraMatrix.at<double>(1, 1)) < eps);
    CV_Assert(norm(camera.distCoeffs) < eps);

    Mat homography = affine2homography(affineTransformation);
    if (homography.type() != CV_64FC1)
    {
      Mat homographyDouble;
      homography.convertTo(homographyDouble, CV_64FC1);
      homography = homographyDouble;
    }

    //TODO: which inversion method is better?
    Mat fullTransform = camera.cameraMatrix.inv() * homography * camera.cameraMatrix;
    const int dim = 3;
    CV_Assert(fullTransform.rows == dim && fullTransform.cols == dim);
    CV_Assert(fullTransform.type() == CV_64FC1);
    Mat rotationalComponentWithScale = fullTransform(Range(0, 2), Range(0, 2));
    double det = determinant(rotationalComponentWithScale);
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
    pose_cam = camera.extrinsics.inv() * pose2d_cam * initialPose_cam;
  }
  else
  {
    vector<Point2f> projectedObjectPoints;
    camera.projectPoints(edgeModel.points, initialPose_cam, projectedObjectPoints);

    Mat transformedObjectPoints;
    transform(Mat(projectedObjectPoints), transformedObjectPoints, affineTransformation);

    solvePnP(Mat(edgeModel.points), transformedObjectPoints, camera.cameraMatrix, camera.distCoeffs, poseWithExtrinsics_cam.rvec, poseWithExtrinsics_cam.tvec, useClosedFormPnP);
    pose_cam = camera.extrinsics.inv() * poseWithExtrinsics_cam;
  }
}

void Silhouette::visualizeSimilarityTransformation(const cv::Mat &similarityTransformation, cv::Mat &image, cv::Scalar color) const
{
  Mat transformedEdgels;
  transform(edgels, transformedEdgels, similarityTransformation);
  vector<Point2f> transformedEdgelsVec = transformedEdgels;
  drawPoints(transformedEdgelsVec, image, color);
}

void Silhouette::draw(cv::Mat &image, cv::Scalar color, int thickness) const
{
  Mat edgelsInt;
  edgels.convertTo(edgelsInt, CV_32SC2);
  vector<vector<Point> > contours(1);
  contours[0] = edgelsInt;
  drawContours(image, contours, -1, color, thickness);
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
  fn["downsampledEdgels"] >> downsampledEdgels;

  Mat silhouetteCenterMat;
  fn["silhouetteCenter"] >> silhouetteCenterMat;
  CV_Assert(!silhouetteCenterMat.empty());
  silhouetteCenter = Point2f(silhouetteCenterMat);

  initialPose_cam.read(fn);
}

void Silhouette::write(cv::FileStorage &fs) const
{
  fs << "edgels" << edgels;
  fs << "silhouette2normalized" << silhouette2normalized;
  fs << "downsampledEdgels" << downsampledEdgels;
  fs << "silhouetteCenter" << Mat(silhouetteCenter);

  initialPose_cam.write(fs);
}


void Silhouette::camera2object(const cv::Mat &similarityTransformation_cam, cv::Mat &similarityTransformation_obj) const
{
  Mat similarity_cam = affine2homography(similarityTransformation_cam);
  Mat cam2obj = Mat::eye(3, 3, similarityTransformation_cam.type());
  CV_Assert(similarityTransformation_cam.type() == CV_32FC1);
  cam2obj.at<float>(0, 2) = -silhouetteCenter.x;
  cam2obj.at<float>(1, 2) = -silhouetteCenter.y;

  Mat similarity_obj = cam2obj * similarity_cam * cam2obj.inv();
  similarityTransformation_obj = homography2affine(similarity_obj);
}

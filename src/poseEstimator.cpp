/*
 * poseEstimator.cpp
 *
 *  Created on: Dec 2, 2011
 *      Author: Ilya Lysenkov
 */

#include "edges_pose_refiner/poseEstimator.hpp"
#include "edges_pose_refiner/localPoseRefiner.hpp"
#include "edges_pose_refiner/pclProcessing.hpp"
#include "edges_pose_refiner/nonMaximumSuppression.hpp"

#ifdef USE_3D_VISUALIZATION
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#endif

#include <opencv2/opencv.hpp>
#include <boost/thread/thread.hpp>

//#define VISUALIZE_TABLE_ESTIMATION

//#define VISUALIZE_INITIAL_POSE_REFINEMENT
//#define VISUALIZE_GEOMETRIC_HASHING

using namespace cv;
using std::cout;
using std::endl;

PoseEstimator::PoseEstimator(const PinholeCamera &_kinectCamera, const PoseEstimatorParams &_params)
{
  kinectCamera = _kinectCamera;
  params = _params;
}

void PoseEstimator::setModel(const EdgeModel &_edgeModel)
{
  edgeModel = _edgeModel;

  Ptr<const PinholeCamera> centralCameraPtr = new PinholeCamera(kinectCamera);
  edgeModel.generateSilhouettes(centralCameraPtr, params.silhouetteCount, silhouettes, params.downFactor, params.closingIterationsCount);
  generateGeometricHashes();

  votes.resize(silhouettes.size());
  for (size_t i = 0; i < silhouettes.size(); ++i)
  {
    votes[i] = Mat(silhouettes[i].size(), silhouettes[i].size(), CV_32SC1);
  }
}

void PoseEstimator::generateGeometricHashes()
{
  ghTable.clear();
  canonicScales.resize(silhouettes.size());
  cout << "number of train silhouettes: " << silhouettes.size() << endl;
  for (size_t i = 0; i < silhouettes.size(); ++i)
  {
    cout << "generating hashes for silhouette #" << i << endl;
    silhouettes[i].generateGeometricHash(i, ghTable, canonicScales[i], params.ghGranularity, params.ghBasisStep, params.ghMinDistanceBetweenBasisPoints);
  }
}

void PoseEstimator::estimatePose(const cv::Mat &kinectBgrImage, const cv::Mat &glassMask, std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities, const cv::Vec4f *tablePlane, std::vector<cv::Mat> *initialSilhouettes) const
{
  CV_Assert(kinectBgrImage.size() == glassMask.size());
  CV_Assert(kinectBgrImage.size() == getValidTestImageSize());

  if (silhouettes.empty())
  {
    std::cerr << "PoseEstimator is not initialized" << std::endl;
    return;
  }

  Mat testEdges, silhouetteEdges;
  computeCentralEdges(kinectBgrImage, glassMask, testEdges, silhouetteEdges);

//  getInitialPoses(glassMask, poses_cam, posesQualities);
  getInitialPosesByGeometricHashing(glassMask, poses_cam, posesQualities, initialSilhouettes);

//  refineInitialPoses(kinectBgrImage, glassMask, poses_cam, posesQualities);
  if (tablePlane != 0)
  {
    //TODO: move up
    params.lmParams.lmDownFactor = 0.5f;
    params.lmParams.lmClosingIterationsCount = 5;
    refinePosesByTableOrientation(*tablePlane, testEdges, silhouetteEdges, poses_cam, posesQualities);

    params.lmParams.lmDownFactor = 1.0f;
    params.lmParams.lmClosingIterationsCount = 10;
    refineFinalTablePoses(*tablePlane, testEdges, silhouetteEdges, poses_cam, posesQualities);
  }
}

void PoseEstimator::refineFinalTablePoses(const cv::Vec4f &tablePlane,
                    const cv::Mat &testEdges, const cv::Mat &silhouetteEdges,
                    std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities) const
{
  cout << "final refinement of poses by table orientation" << endl;
  if (poses_cam.empty())
  {
    return;
  }

  posesQualities.resize(poses_cam.size());
  LocalPoseRefiner localPoseRefiner(edgeModel, testEdges, kinectCamera.cameraMatrix, kinectCamera.distCoeffs, kinectCamera.extrinsics.getProjectiveMatrix(), params.lmParams);
  localPoseRefiner.setSilhouetteEdges(silhouetteEdges);
  for (size_t initPoseIdx = 0; initPoseIdx < poses_cam.size(); ++initPoseIdx)
  {
    PoseRT &initialPose_cam = poses_cam[initPoseIdx];
    posesQualities[initPoseIdx] = localPoseRefiner.refineUsingSilhouette(initialPose_cam, true, tablePlane);
  }
}

void PoseEstimator::refinePosesByTableOrientation(const cv::Vec4f &tablePlane,
                    const cv::Mat &centralEdges, const cv::Mat &silhouetteEdges,
                    vector<PoseRT> &poses_cam, vector<float> &initPosesQualities) const
{
  cout << "refine poses by table orientation" << endl;
  if (poses_cam.empty())
  {
    return;
  }

  initPosesQualities.resize(poses_cam.size());
  vector<float> rotationAngles(poses_cam.size());

  TermCriteria oldCriteria = params.lmParams.termCriteria;
  //TODO: move up
  TermCriteria newCriteria = TermCriteria(CV_TERMCRIT_ITER, 5, 0.0);
  params.lmParams.termCriteria = newCriteria;
  vector<Mat> jacobians;
  refineInitialPoses(centralEdges, silhouetteEdges, poses_cam, initPosesQualities, &jacobians);
  params.lmParams.termCriteria = oldCriteria;

  for (size_t initPoseIdx = 0; initPoseIdx < poses_cam.size(); ++initPoseIdx)
  {
#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
    showEdgels(centralEdges, edgeModel.points, poses_cam[initPoseIdx], kinectCamera, "before alignment to a table plane");

//    if (pt_pub != 0)
//    {
//      publishPoints(edgeModel.points, initialPose_cam.getRvec(), initialPose_cam.getTvec(), *pt_pub, 1, Scalar(0, 0, 255));
//      namedWindow("ready to align pose to a table plane");
//      waitKey();
//      destroyWindow("ready to align pose to a table plane");
//    }
#endif

    findTransformationToTable(poses_cam[initPoseIdx], tablePlane, rotationAngles[initPoseIdx], jacobians[initPoseIdx]);

#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
    showEdgels(centralEdges, edgeModel.points, poses_cam[initPoseIdx], kinectCamera, "after alignment to a table plane");
//    if (pt_pub != 0)
//    {
//      publishPoints(edgeModel.points, initialPose_cam.getRvec(), initialPose_cam.getTvec(), *pt_pub, 1, Scalar(0, 0, 255));
//      namedWindow("aligned pose to a table plane");
//      waitKey();
//      destroyWindow("aligned pose to a table plane");
//    }
    cout << "quality[" << initPoseIdx << "]: " << initPosesQualities[initPoseIdx] << endl;
    waitKey();
#endif
  }

  //TODO: move up
  newCriteria = TermCriteria(CV_TERMCRIT_ITER, 1, 0.0);
  params.lmParams.termCriteria = newCriteria;
  LocalPoseRefiner localPoseRefiner(edgeModel, centralEdges, kinectCamera.cameraMatrix, kinectCamera.distCoeffs, kinectCamera.extrinsics.getProjectiveMatrix(), params.lmParams);
  localPoseRefiner.setSilhouetteEdges(silhouetteEdges);
  for (size_t initPoseIdx = 0; initPoseIdx < poses_cam.size(); ++initPoseIdx)
  {
    PoseRT &initialPose_cam = poses_cam[initPoseIdx];
    initPosesQualities[initPoseIdx] = localPoseRefiner.refineUsingSilhouette(initialPose_cam, true, tablePlane);

#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
    showEdgels(centralEdges, edgeModel.points, initialPose_cam, kinectCamera, "central pose aligned to a table plane");
    showEdgels(centralEdges, edgeModel.stableEdgels, initialPose_cam, kinectCamera, "stable edgels alinged to a table plane");
    cout << "quality[" << initPoseIdx << "]: " << initPosesQualities[initPoseIdx] << endl;
    waitKey();
#endif
  }
  params.lmParams.termCriteria = oldCriteria;
  localPoseRefiner.setParams(params.lmParams);

  vector<bool> isPoseFilteredOut;
  filterOut3DPoses(initPosesQualities, poses_cam,
                   params.ratioToMinimum, params.neighborMaxRotation, params.neighborMaxTranslation,
                   isPoseFilteredOut);

  filterValues(poses_cam, isPoseFilteredOut);
  initPosesQualities.resize(poses_cam.size());
  cout << "suppression: " << isPoseFilteredOut.size() << " -> " << poses_cam.size() << endl;

  for (size_t initPoseIdx = 0; initPoseIdx < poses_cam.size(); ++initPoseIdx)
  {
    PoseRT &initialPose_cam = poses_cam[initPoseIdx];
    initPosesQualities[initPoseIdx] = localPoseRefiner.refineUsingSilhouette(initialPose_cam, true, tablePlane);

#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
    showEdgels(centralEdges, edgeModel.points, initialPose_cam, kinectCamera, "central pose refined by LM with a table plane");
    showEdgels(centralEdges, edgeModel.stableEdgels, initialPose_cam, kinectCamera, "stable edgels refined by LM with a table plane");
    cout << "quality[" << initPoseIdx << "]: " << initPosesQualities[initPoseIdx] << endl;
    waitKey();
#endif
  }

  vector<float>::iterator bestPoseIt = std::min_element(initPosesQualities.begin(), initPosesQualities.end());
  int bestPoseIdx = bestPoseIt - initPosesQualities.begin();
//  vector<float>::iterator bestPoseIt = std::min_element(rotationAngles.begin(), rotationAngles.end());
//  int bestPoseIdx = bestPoseIt - rotationAngles.begin();

  PoseRT bestPose = poses_cam[bestPoseIdx];
  float bestPoseQuality = initPosesQualities[bestPoseIdx];
  cout << "best quality: " << bestPoseQuality << endl;
  poses_cam.clear();
  poses_cam.push_back(bestPose);
  initPosesQualities.clear();
  initPosesQualities.push_back(bestPoseQuality);
}

void PoseEstimator::findTransformationToTable(PoseRT &pose_cam, const cv::Vec4f &tablePlane, float &rotationAngle, const Mat finalJacobian) const
{
  EdgeModel rotatedEdgeModel;
  edgeModel.rotate_cam(pose_cam, rotatedEdgeModel);

  const int dim = 3;

  Point3d rotatedAxis = rotatedEdgeModel.upStraightDirection;
  Point3d tableNormal(tablePlane[0], tablePlane[1], tablePlane[2]);

  Mat Rot_obj2cam = rotatedEdgeModel.Rt_obj2cam.clone();
  Rot_obj2cam(Range(0, 3), Range(3, 4)).setTo(0);
  Point3d tableNormal_obj, rotatedAxis_obj;
  transformPoint(Rot_obj2cam.inv(DECOMP_SVD), tableNormal, tableNormal_obj);
  transformPoint(Rot_obj2cam.inv(DECOMP_SVD), rotatedAxis, rotatedAxis_obj);

  //find rotation to align the rotation axis with the table normal
  double cosphi = rotatedAxis_obj.ddot(tableNormal_obj) / (norm(rotatedAxis_obj) * norm(tableNormal_obj));
  double phi = acos(cosphi);
  rotationAngle = std::min(phi, CV_PI - phi);
  Point3d rvecPt = rotatedAxis_obj.cross(tableNormal_obj);
  rvecPt = rvecPt * (phi / norm(rvecPt));
  Mat rvec_obj = Mat(rvecPt).reshape(1, dim);
  Mat R;
  Rodrigues(rvec_obj, R);
  Mat zeroVec = Mat::zeros(dim, 1, CV_64FC1);
  Mat R_Rt;
  createProjectiveMatrix(R, zeroVec, R_Rt);

  Point3d transformedTableAnchor;
  transformPoint(rotatedEdgeModel.Rt_obj2cam * R_Rt * rotatedEdgeModel.Rt_obj2cam.inv(DECOMP_SVD), rotatedEdgeModel.tableAnchor, transformedTableAnchor);

  //project transformedTableAnchor on the table plane
  double alpha = -(tablePlane[3] + tableNormal.ddot(transformedTableAnchor)) / tableNormal.ddot(tableNormal);
  Point3d anchorOnTable = transformedTableAnchor + alpha * tableNormal;

  Point3d transformedTableAnchor_obj, anchorOnTable_obj;
  transformPoint(rotatedEdgeModel.Rt_obj2cam.inv(DECOMP_SVD), transformedTableAnchor, transformedTableAnchor_obj);
  transformPoint(rotatedEdgeModel.Rt_obj2cam.inv(DECOMP_SVD), anchorOnTable, anchorOnTable_obj);

  Point3d tvecPt_obj = anchorOnTable_obj - transformedTableAnchor_obj;
  Mat tvec_obj = Mat(tvecPt_obj).reshape(1, dim);

  if (!finalJacobian.empty())
  {
    Mat Q = finalJacobian.t() * finalJacobian;
    const int constraintCount = 4;
    const int fullDim = 6;
    Mat E = (Mat_<double>(constraintCount, fullDim) <<
                0.0, 0.0, 0.0, tableNormal_obj.x, tableNormal_obj.y, tableNormal_obj.z,
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0
            );
    double d_tvec = tableNormal_obj.ddot(anchorOnTable_obj) - tableNormal_obj.ddot(transformedTableAnchor_obj);
    Mat d(constraintCount, 1, CV_64FC1);
    d.at<double>(0) = d_tvec;
    const int dim = 3;
    Mat dRows = d.rowRange(1, 1 + dim);
    rvec_obj.copyTo(dRows);

    Mat A(constraintCount + fullDim, constraintCount + fullDim, CV_64FC1, Scalar(0));
    Mat roiQ = A(Range(0, fullDim), Range(0, fullDim));
    Q.copyTo(roiQ);
    Mat roiE = A(Range(fullDim, fullDim + constraintCount), Range(0, fullDim));
    E.copyTo(roiE);
    Mat Et = E.t();
    Mat roiEt = A(Range(0, fullDim), Range(fullDim, fullDim + constraintCount));
    Et.copyTo(roiEt);
    Mat b(fullDim + constraintCount, 1, CV_64FC1, Scalar(0));
    Mat roiD = b.rowRange(fullDim, fullDim + constraintCount);
    d.copyTo(roiD);

    Mat solution;
    bool result = solve(A, b, solution);
    CV_Assert(result);

    CV_Assert(solution.cols == 1);
    rvec_obj = solution.rowRange(0, dim);
    tvec_obj = solution.rowRange(dim, 2*dim);
  }

  PoseRT pose_obj(rvec_obj, tvec_obj);

  Mat Rt_cam = rotatedEdgeModel.Rt_obj2cam * pose_obj.getProjectiveMatrix() * rotatedEdgeModel.Rt_obj2cam.inv(DECOMP_SVD);
  PoseRT transform2table_cam(Rt_cam);

  pose_cam = transform2table_cam * pose_cam;
}

void PoseEstimator::computeCentralEdges(const Mat &centralBgrImage, const Mat &glassMask, Mat &centralEdges, Mat &silhouetteEdges) const
{
  Mat centralGrayImage;
  cvtColor(centralBgrImage, centralGrayImage, CV_BGR2GRAY);
  Canny(centralGrayImage, centralEdges, params.cannyThreshold1, params.cannyThreshold2);
#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
  imshow("central edges before", centralEdges);
#endif

  Mat centralEdgesMask;
  dilate(glassMask, centralEdgesMask, Mat(), Point(-1, -1), params.dilationsForEdgesRemovalCount);
  centralEdges.setTo(0, ~centralEdgesMask);
#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
  imshow("central edges after", centralEdges);
#endif
  Mat glassMaskClone = glassMask.clone();
  vector<vector<Point> > glassMaskContours;
  findContours(glassMaskClone, glassMaskContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
  drawContours(centralEdges, glassMaskContours, -1, Scalar(255));
  silhouetteEdges = Mat(glassMask.size(), CV_8UC1, Scalar(0));
  drawContours(silhouetteEdges, glassMaskContours, -1, Scalar(255));

#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
  imshow("central edges even after", centralEdges);
  imshow("silhouette edges", silhouetteEdges);
  //waitKey();
#endif

}

PoseEstimator::BasisMatch::BasisMatch()
{
  confidence = 0.0f;
  trainBasis = Basis(0, 0);
  testBasis = Basis(0, 0);
  silhouetteIndex = 0;
}

void suppressNonMaximum(const cv::Mat &confidences, int windowSize, float absoluteSuppressionFactor, std::vector<cv::Point> &maxLocations)
{
  CV_Assert(windowSize % 2 == 1);
  CV_Assert(confidences.type() == CV_32FC1);
  double maxValDouble;
  minMaxLoc(confidences, 0, &maxValDouble);
  float maxVal = maxValDouble;

  int halfWindowSize = windowSize / 2;
  maxLocations.clear();
  Mat wideConfidences(confidences.rows + windowSize - 1, confidences.cols + windowSize - 1, confidences.type(), Scalar(0));
  Mat roi = wideConfidences(Rect(halfWindowSize, halfWindowSize, confidences.cols, confidences.rows));
  confidences.copyTo(roi);

  Mat isSuppressed(wideConfidences.size(), CV_8UC1, Scalar(0));

  for (int row = 0; row < confidences.rows; ++row)
  {
    for (int col = 0; col < confidences.cols; ++col)
    {
      float mainVal = wideConfidences.at<float>(row + halfWindowSize, col + halfWindowSize);
      if (mainVal * absoluteSuppressionFactor < maxVal)
      {
        isSuppressed.at<uchar>(row + halfWindowSize, col + halfWindowSize) = 255;
        continue;
      }

      for (int dy = -halfWindowSize; dy < halfWindowSize; ++dy)
      {
        for (int dx = -halfWindowSize; dx < halfWindowSize; ++dx)
        {
          int x = col + halfWindowSize + dx;
          int y = row + halfWindowSize + dy;

          float val = wideConfidences.at<float>(y, x);
          if (mainVal > val)
          {
            isSuppressed.at<uchar>(y, x) = 255;
          }
        }
      }
    }
  }

  for (int row = 0; row < confidences.rows; ++row)
  {
    for (int col = 0; col < confidences.cols; ++col)
    {
      if (isSuppressed.at<uchar>(row + halfWindowSize, col + halfWindowSize) == 0)
      {
        maxLocations.push_back(Point(col, row));
      }
    }
  }
}

void decomposeSimilarityTransformation(const cv::Mat &transformation, Point2f &translation, Point2f &rotationCosSin, float &scale)
{
  CV_Assert(transformation.type() == CV_32FC1);

  Mat scaledRotationMatrix = transformation(Range(0, 2), Range(0, 2));
  scale = sqrt(determinant(scaledRotationMatrix));
  const float eps = 1e-4;
  CV_Assert(scale > eps);
  rotationCosSin.x = scaledRotationMatrix.at<float>(0, 0) / scale;
  rotationCosSin.y = scaledRotationMatrix.at<float>(1, 0) / scale;

  translation.x = transformation.at<float>(0, 2);
  translation.y = transformation.at<float>(1, 2);
}

void compareSimilarityTransformations(const cv::Mat &transformation_1, const cv::Mat &transformation_2, float &translationDiff, float &rotationCosDiff, float &scaleChange)
{
  Point2f translation_1, cosSin_1;
  float scale_1;
  decomposeSimilarityTransformation(transformation_1, translation_1, cosSin_1, scale_1);

  Point2f translation_2, cosSin_2;
  float scale_2;
  decomposeSimilarityTransformation(transformation_2, translation_2, cosSin_2, scale_2);

  translationDiff = norm(translation_2 - translation_1);
  scaleChange = scale_2 / scale_1;
  rotationCosDiff = cosSin_1.dot(cosSin_2);
}

void PoseEstimator::findBasisMatches(const std::vector<cv::Point2f> &contour, const Basis &testBasis, std::vector<BasisMatch> &basisMatches) const
{
  Point2f firstPoint = contour.at(testBasis.first);
  Point2f secondPoint= contour.at(testBasis.second);
  const float testScale = norm(firstPoint - secondPoint);

  for (size_t i = 0; i < silhouettes.size(); ++i)
  {
    votes[i] = Scalar(0);
  }

  Mat similarityTransformation;
  findSimilarityTransformation(firstPoint, secondPoint, similarityTransformation);
  Mat contourFloatMat(contour);
  Mat transformedContour;
  transform(contourFloatMat, transformedContour, similarityTransformation);
  vector<Point2f> transformedContourVec = transformedContour;
  for (size_t i = 0; i < transformedContourVec.size(); ++i)
  {
    if (i == testBasis.first || i == testBasis.second)
    {
      continue;
    }

    float invertedGranularity = 1.0 / params.ghGranularity;
    Point pt = transformedContourVec[i] * invertedGranularity;
    GHKey key(pt.x, pt.y);

    std::pair<GHTable::iterator, GHTable::iterator> range = ghTable.equal_range(key);
//        std::pair<GHTable::const_iterator, GHTable::const_iterator> range = ghTable.equal_range(key);
    for(GHTable::iterator it = range.first; it != range.second; ++it)
    {
      GHValue value = it->second;
      votes[value[0]].at<int>(value[1], value[2]) += 1;
    }
  }

  for (size_t i = 0; i < votes.size(); ++i)
  {
    Mat currentVotes;
    votes[i].convertTo(currentVotes, CV_32FC1, 1.0 / silhouettes[i].size());
    Mat currentScale = testScale * canonicScales[i];
    currentVotes /= currentScale;

    vector<Point> maxLocations;
    suppressNonMaximum(currentVotes, params.votesWindowSize, params.votesConfidentSuppression, maxLocations);

    for (size_t j = 0; j < maxLocations.size(); ++j)
    {
      Point maxLoc = maxLocations[j];

      BasisMatch match;
      match.confidence = currentVotes.at<float>(maxLoc.y, maxLoc.x);

      if (currentScale.at<float>(maxLoc.y, maxLoc.x) < params.minScale)
      {
        continue;
      }

      match.testBasis = testBasis;
      match.trainBasis = Basis(maxLoc.y, maxLoc.x);
      match.silhouetteIndex = i;

      basisMatches.push_back(match);
    }
  }
}

void PoseEstimator::suppressBasisMatches(const std::vector<BasisMatch> &matches, std::vector<BasisMatch> &filteredMatches) const
{
  vector<float> errors(matches.size());
  for (size_t i = 0; i < matches.size(); ++i)
  {
    const float eps = 1e-3;
//    CV_Assert(matches[i].confidence > eps);
    errors[i] = 1.0 / (matches[i].confidence + eps);
  }

  vector<bool> isSuppressed;
  bool useNeighbors = false;
  suppressNonMinimum(errors, params.basisConfidentSuppression, isSuppressed, useNeighbors);

  filteredMatches.clear();
  for (size_t i = 0; i < matches.size(); ++i)
  {
    if (!isSuppressed[i])
    {
      filteredMatches.push_back(matches[i]);
    }
  }
}

void PoseEstimator::suppressSimilarityTransformations(const std::vector<BasisMatch> &matches, const std::vector<cv::Mat> &similarityTransformations_obj, std::vector<bool> &isSuppressed) const
{
  //TODO: move up
  const float maxTranslationDiff = 10.0f;
  const float minRotationCosDiff = 0.95f;
  const float maxScaleChange = 1.2f;
  const int maxSilhouetteNeighbor = 1;
  const float finalConfidentDomination = 1.5f;

  vector<vector<int> > neighbors(matches.size());
  for (size_t i = 0; i < matches.size(); ++i)
  {
    for (size_t j = i+1; j < matches.size(); ++j)
    {
      int silhouettesCount = static_cast<int>(silhouettes.size());
      //TODO: compare with
      //if (abs(matches[i].silhouetteIndex - matches[j].silhouetteIndex) > maxSilhouetteNeighbor)
      if ((silhouettesCount + matches[i].silhouetteIndex - matches[j].silhouetteIndex) % silhouettesCount > maxSilhouetteNeighbor)
      {
        continue;
      }

      float translationDiff, rotationCosDiff, scaleChange;
      compareSimilarityTransformations(similarityTransformations_obj[i], similarityTransformations_obj[j], translationDiff, rotationCosDiff, scaleChange);

      if (translationDiff < maxTranslationDiff && rotationCosDiff > minRotationCosDiff && std::max(scaleChange, 1.0f / scaleChange) < maxScaleChange)
      {
        neighbors[i].push_back(j);
        neighbors[j].push_back(i);
      }
    }
  }

  float maxVotes = 0.0f;
  for (size_t i = 0; i < matches.size(); ++i)
  {
    if (matches[i].confidence > maxVotes)
    {
      maxVotes = matches[i].confidence;
    }
  }

  isSuppressed.resize(matches.size(), false);
  for (size_t i = 0; i < matches.size(); ++i)
  {
    if (matches[i].confidence * finalConfidentDomination < maxVotes)
    {
      isSuppressed[i] = true;
      continue;
    }

    for (size_t j = 0; j < neighbors[i].size(); ++j)
    {
      if (matches[i].confidence > matches[neighbors[i][j]].confidence)
      {
        isSuppressed[neighbors[i][j]] = true;
      }
    }
  }
}

void PoseEstimator::filterOut3DPoses(const std::vector<float> &errors, const std::vector<PoseRT> &poses_cam,
                                     float ratioToMinimum, float neighborMaxRotation, float neighborMaxTranslation,
                                     std::vector<bool> &isFilteredOut) const
{
  CV_Assert(errors.size() == poses_cam.size());

  filterOutHighValues(errors, ratioToMinimum, isFilteredOut);

  vector<vector<int> > neighbors(poses_cam.size());
  for (size_t i = 0; i < poses_cam.size(); ++i)
  {
    if (isFilteredOut[i])
    {
      continue;
    }

    for (size_t j = i + 1; j < poses_cam.size(); ++j)
    {
      if (isFilteredOut[j])
      {
        continue;
      }

      double rotationDistance, translationDistance;
      //TODO: check symmetry of the distance
      PoseRT::computeDistance(poses_cam[i], poses_cam[j], rotationDistance, translationDistance, edgeModel.Rt_obj2cam);

      if (rotationDistance < neighborMaxRotation && translationDistance < neighborMaxTranslation)
      {
        neighbors[i].push_back(j);
        neighbors[j].push_back(i);
      }
    }
  }

  filterOutNonMinima(errors, neighbors, isFilteredOut);
}

void PoseEstimator::suppressBasisMatchesIn3D(std::vector<BasisMatch> &matches) const
{
  vector<float> errors(matches.size());
  vector<PoseRT> poses_cam(matches.size());
  for (size_t i = 0; i < matches.size(); ++i)
  {
    errors[i] = -matches[i].confidence;
    poses_cam[i] = matches[i].pose;
  }

  vector<bool> isFilteredOut;
  //TODO: change the parameter to 1.0/p
  filterOut3DPoses(errors, poses_cam, 1.0 / params.confidentSuppresion3D,
                   params.maxRotation3D, params.maxTranslation3D, isFilteredOut);
  filterValues(matches, isFilteredOut);
}


void PoseEstimator::estimateSimilarityTransformations(const std::vector<cv::Point> &contour, std::vector<BasisMatch> &matches) const
{
  for (size_t i = 0; i < matches.size(); ++i)
  {
    Mat testTransformation, trainTransformation;
    findSimilarityTransformation(contour[matches[i].testBasis.first], contour[matches[i].testBasis.second], testTransformation);
    Mat edgels;
    silhouettes[matches[i].silhouetteIndex].getEdgels(edgels);
    vector<Point2f> edgelsVec = edgels;
    findSimilarityTransformation(edgelsVec[matches[i].trainBasis.first], edgelsVec[matches[i].trainBasis.second], trainTransformation);

    Mat testHomography = affine2homography(testTransformation);
    Mat invertedTestTransformation(homography2affine(testHomography.inv()));
    Mat finalSimilarityTransformation;
    composeAffineTransformations(trainTransformation, invertedTestTransformation, finalSimilarityTransformation);
    matches[i].similarityTransformation_cam = finalSimilarityTransformation;
    silhouettes[matches[i].silhouetteIndex].camera2object(matches[i].similarityTransformation_cam, matches[i].similarityTransformation_obj);
  }
}

void PoseEstimator::estimatePoses(std::vector<BasisMatch> &matches) const
{
  for (size_t i = 0; i < matches.size(); ++i)
  {
    silhouettes[matches[i].silhouetteIndex].affine2poseRT(edgeModel, kinectCamera, matches[i].similarityTransformation_cam, params.useClosedFormPnP, matches[i].pose);
  }
}

//TODO: suppress poses in 3D
//TODO: refine estimate similarities by using all corresponding points
void PoseEstimator::getInitialPosesByGeometricHashing(const cv::Mat &glassMask, std::vector<PoseRT> &initialPoses, std::vector<float> &initialPosesQualities, std::vector<cv::Mat> *initialSilhouettes) const
{
  cout << "get initial poses by geometric hashing..." << endl;
  initialPoses.clear();
  initialPosesQualities.clear();

  Mat maskClone = glassMask.clone();
  vector<vector<Point> > allGlassContours;
  findContours(maskClone, allGlassContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

  cout << "Number of glass contours: " << allGlassContours.size() << endl;
  cout << "Silhouettes size: " << silhouettes.size() << endl;

  for (size_t contourIndex = 0; contourIndex < allGlassContours.size(); ++contourIndex)
  {
    vector<Point> &srcContour = allGlassContours[contourIndex];
    if (srcContour.size() < params.minGlassContourLength || contourArea(srcContour) < params.minGlassContourArea)
    {
      continue;
    }

    vector<Point> currentContour;
    for (size_t i = 0; i < srcContour.size(); i += params.ghTestBasisStep)
    {
      currentContour.push_back(srcContour[i]);
    }

    Mat currentContourMat = Mat(currentContour);
    Mat currentContourFloatMat;
    currentContourMat.convertTo(currentContourFloatMat, CV_32FC2);
    vector<Point2f> currentContourFloat(currentContourMat);

    vector<BasisMatch> bestMatches;
    const int ghIterationCount = ceil(log(1.0 - params.ghSuccessProbability) / log(1.0 - params.ghObjectContourProportion * params.ghObjectContourProportion));
    for (int iterationIndex = 0; iterationIndex < ghIterationCount; ++iterationIndex)
    {
      int firstIndex = rand() % currentContour.size();
      int secondIndex = rand() % currentContour.size();
      if (currentContour[firstIndex] == currentContour[secondIndex])
      {
        continue;
      }
      Basis testBasis(firstIndex, secondIndex);
      vector<BasisMatch> currentMatches;
      findBasisMatches(currentContourFloat, testBasis, currentMatches);
      std::copy(currentMatches.begin(), currentMatches.end(), std::back_inserter(bestMatches));
    }

    vector<BasisMatch> filteredCorrespondences;
    suppressBasisMatches(bestMatches, filteredCorrespondences);

    estimateSimilarityTransformations(currentContour, filteredCorrespondences);
    estimatePoses(filteredCorrespondences);

    cout << "before 3d: " << filteredCorrespondences.size() << endl;
    suppressBasisMatchesIn3D(filteredCorrespondences);
    cout << "after 3d: " << filteredCorrespondences.size() << endl;


//    vector<bool> isSimilaritySuppressed;
//    suppressSimilarityTransformations(filteredCorrespondences, similarityTransformations_obj, isSimilaritySuppressed);


    cout << "best correspondences size: " << bestMatches.size() << endl;
    cout << "filtered correspondences size: " << filteredCorrespondences.size() << endl;
    int remainedCorrespondences = 0;
    for (size_t i = 0; i < filteredCorrespondences.size(); ++i)
    {
//      if (isSimilaritySuppressed[i])
//      {
//        continue;
//      }
      ++remainedCorrespondences;

//      PoseRT pose;
//      silhouettes[filteredCorrespondences[i].silhouetteIndex].affine2poseRT(edgeModel, kinectCamera, similarityTransformations_cam[i], params.useClosedFormPnP, pose);
//      initialPoses.push_back(pose);
//      initialPosesQualities.push_back(-filteredCorrespondences[i].confidence);

      initialPoses.push_back(filteredCorrespondences[i].pose);
      initialPosesQualities.push_back(-filteredCorrespondences[i].confidence);

      if (initialSilhouettes != 0)
      {
        Mat edgels;
        silhouettes[filteredCorrespondences[i].silhouetteIndex].getEdgels(edgels);
        Mat transformedEdgels;
        transform(edgels, transformedEdgels, filteredCorrespondences[i].similarityTransformation_cam);
        initialSilhouettes->push_back(transformedEdgels);
      }

#ifdef VISUALIZE_GEOMETRIC_HASHING
      if (filteredCorrespondences[i].silhouetteIndex != 0)
      {
//        continue;
      }
      Mat visualization = glassMask.clone();
      silhouettes[filteredCorrespondences[i].silhouetteIndex].visualizeSimilarityTransformation(filteredCorrespondences[i].similarityTransformation_cam, visualization, Scalar(255, 0, 0));
      imshow("transformation by geometric hashing", visualization);

      cout << "votes: " << filteredCorrespondences[i].confidence << endl;
      cout << "idx: " << filteredCorrespondences[i].silhouetteIndex << endl;
      cout << "i: " << i << endl;

/*
      if (i != 0)
      {
        float translationDiff, rotationCosDiff, scaleChange;
        compareSimilarityTransformations(similarityTransformations_obj[i], similarityTransformations_obj[i-1], translationDiff, rotationCosDiff, scaleChange);
        cout << translationDiff << endl;
        cout << rotationCosDiff<< endl;
        cout << scaleChange<< endl;
        for (size_t j = 0; j < neighbors[i].size(); ++j)
        {
          cout << neighbors[i][j] << " ";
        }
        cout << endl;


//        for (size_t j = 0; j < neighbors[i-1].size(); ++j)
//        {
//          cout << neighbors[i-1][j] << " ";
//        }
//        cout << endl;

      }
*/

      cout << endl;
      waitKey();
#endif
    }
    cout << "remained correspondences: " << remainedCorrespondences << endl;

  }
  cout << "Initial pose count: " << initialPoses.size() << endl;
}

void PoseEstimator::suppressNonMinimum(std::vector<float> errors, float absoluteSuppressionFactor, std::vector<bool> &isSuppressed, bool useNeighbors)
{
  isSuppressed.resize(errors.size(), false);
  for (size_t i = 0; i < errors.size(); ++i)
  {
    for (size_t j = 0; j < errors.size(); ++j)
    {
      if (i == j)
      {
        continue;
      }

      if (errors[j] * absoluteSuppressionFactor < errors[i])
      {
        isSuppressed[i] = true;
        break;
      }
    }

    if (useNeighbors)
    {
      if (isSuppressed[i])
      {
        continue;
      }

      bool isNextBetter = errors[i] > errors[(i + 1) % errors.size()];
      bool isPreviousBetter = errors[i] > errors[(static_cast<int>(errors.size() + i) - 1) % errors.size()];

      if (isNextBetter || isPreviousBetter)
      {
        isSuppressed[i] = true;
      }
    }
  }
}

void PoseEstimator::getInitialPoses(const cv::Mat &glassMask, std::vector<PoseRT> &initialPoses, std::vector<float> &initialPosesQualities) const
{
  cout << "get initial poses..." << endl;
  initialPoses.clear();
  initialPosesQualities.clear();

//  vector<Point2f> glassContour;
//  mask2contour(glassMask, glassContour);
//  imshow("mask", glassMask);
//  waitKey();

  Mat maskClone = glassMask.clone();
  vector<vector<Point> > allGlassContours;
  findContours(maskClone, allGlassContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

  Mat edgesImage(glassMask.size(), CV_8UC1, Scalar(0));
  drawContours(edgesImage, allGlassContours, -1, Scalar(255));

//  imshow("edges for silhouettes match", edgesImage);
//  waitKey();

  Mat dt;
  distanceTransform(~edgesImage, dt, CV_DIST_L2, CV_DIST_MASK_PRECISE);

  vector<vector<PoseRT> > candidatePoses(allGlassContours.size());
  vector<vector<float> > candidateQualities(allGlassContours.size());
  cout << "Number of glass contours: " << allGlassContours.size() << endl;
  cout << "Silhouettes size: " << silhouettes.size() << endl;
  for (size_t contourIndex = 0; contourIndex < allGlassContours.size(); ++contourIndex)
  {
    if (allGlassContours[contourIndex].size() < params.minGlassContourLength)
    {
      continue;
    }

    if (contourArea(allGlassContours[contourIndex]) < params.minGlassContourArea)
    {
      continue;
    }

    vector<Mat> affineTransformations(silhouettes.size());
    vector<float> poseQualities(silhouettes.size());

    for (size_t i = 0; i < silhouettes.size(); ++i)
    {
      silhouettes[i].match(Mat(allGlassContours[contourIndex]), affineTransformations[i], params.icp2dIterationsCount, params.min2dScaleChange);
      Mat transformedEdgels;
      Mat edgels;
      silhouettes[i].getEdgels(edgels);
      transform(edgels, transformedEdgels, affineTransformations[i]);

      vector<Point2f> transformedEdgelsVec = transformedEdgels;
      double chamferDistance = 0.0;
      for (size_t j = 0; j < transformedEdgelsVec.size(); ++j)
      {
        Point pt = transformedEdgelsVec[j];
        if (isPointInside(dt, pt))
        {
          chamferDistance += dt.at<float>(pt);
        }
        else
        {
          int x = std::max(0, std::min(pt.x, dt.cols - 1));
          int y = std::max(0, std::min(pt.y, dt.rows - 1));
          Point ptOnImage = Point(x, y);
          chamferDistance += dt.at<float>(ptOnImage) + norm(ptOnImage - pt);
        }
      }
      poseQualities[i] = chamferDistance / transformedEdgelsVec.size();
    }

    vector<bool> isSuppressed;
    vector<PoseRT> poses(silhouettes.size());

    suppressNonMinimum(poseQualities, params.confidentDomination, isSuppressed);

    for (size_t i = 0; i < silhouettes.size(); ++i)
    {
      if (isSuppressed[i])
      {
        continue;
      }

      silhouettes[i].affine2poseRT(edgeModel, kinectCamera, affineTransformations[i], params.useClosedFormPnP, poses[i]);
    }

  //TODO: filter by distance
  //  const double translationThreshold = 0.02;
  //  const double rotationThreshold = CV_PI / 10.0;
  //  for (size_t i = 0; i < silhouettes.size(); ++i)
  //  {
  //    if (isDominated[i])
  //      continue;
  //
  //    for (size_t j = 0; j < silhouettes.size(); ++j)
  //    {
  //      if (isDominated[j] || i == j)
  //      {
  //        continue;
  //      }
  //
  //      double rotationDistance, translationDistance;
  //
  //      evaluatePoseWithRotation(*silhouettes[i].edgeModel, *silhouettes[i].edgeModel, poses[i], poses[j], "", 0, Mat(), "", false, &rotationDistance, &translationDistance);
  //      //PoseRT::computeDistance(poses[i], poses[j], rotationDistance, translationDistance, silhouettes[i].getRt_obj2cam());
  //      cout << rotationDistance << " " << translationDistance << endl;
  //
  //      isDominated[i] = rotationDistance < rotationThreshold && translationDistance < translationThreshold;
  //    }
  //  }

    for (size_t i = 0; i < silhouettes.size(); ++i)
    {
      if (isSuppressed[i])
        continue;

      candidatePoses[contourIndex].push_back(poses[i]);
      candidateQualities[contourIndex].push_back(poseQualities[i]);

      initialPoses.push_back(poses[i]);
      initialPosesQualities.push_back(poseQualities[i]);
    }
  }

  cout << "Number of initial poses: " << initialPoses.size() << endl;

//  int bestContourIndex = -1;
//  float bestQuality = std::numeric_limits<float>::max();
//  for (int i = 0; i < static_cast<int>(candidateQualities.size()); ++i)
//  {
//    if (candidateQualities[i].empty())
//    {
//      continue;
//    }
//
//    float quality = *std::min_element(candidateQualities[i].begin(), candidateQualities[i].end());
//    if (quality <= bestQuality)
//    {
//      bestQuality = quality;
//      bestContourIndex = i;
//    }
//  }
//
//  if (bestContourIndex >= 0)
//  {
//    initialPoses = candidatePoses[bestContourIndex];
//    initialPosesQualities = candidateQualities[bestContourIndex];
//  }
}

void PoseEstimator::refineInitialPoses(const cv::Mat &centralEdges, const cv::Mat &silhouetteEdges,
                                       vector<PoseRT> &initPoses_cam, vector<float> &initPosesQualities,
                                       vector<cv::Mat> *jacobians) const
{
  cout << "refine initial poses" << endl;
  if (initPoses_cam.empty())
  {
    return;
  }

  initPosesQualities.resize(initPoses_cam.size());
  if (jacobians != 0)
  {
    jacobians->resize(initPoses_cam.size());
  }

  LocalPoseRefiner localPoseRefiner(edgeModel, centralEdges, kinectCamera.cameraMatrix, kinectCamera.distCoeffs, kinectCamera.extrinsics.getProjectiveMatrix(), params.lmParams);
  localPoseRefiner.setSilhouetteEdges(silhouetteEdges);
  for (size_t initPoseIdx = 0; initPoseIdx < initPoses_cam.size(); ++initPoseIdx)
  {
    //cout << "Pose idx: " << initPoseIdx << endl;
    PoseRT &initialPose_cam = initPoses_cam[initPoseIdx];

#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
    showEdgels(centralEdges, edgeModel.points, initialPose_cam, kinectCamera, "central pose");
    showEdgels(centralEdges, edgeModel.stableEdgels, initialPose_cam, kinectCamera, "stable edgels");
//    waitKey();
#endif
    if (jacobians == 0)
    {
      initPosesQualities[initPoseIdx] = localPoseRefiner.refineUsingSilhouette(initialPose_cam, true);
    }
    else
    {
      initPosesQualities[initPoseIdx] = localPoseRefiner.refineUsingSilhouette(initialPose_cam, true, Vec4f::all(0.0f), &((*jacobians)[initPoseIdx]));
    }
#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
    showEdgels(centralEdges, edgeModel.points, initialPose_cam, kinectCamera, "central pose refined");
    showEdgels(centralEdges, edgeModel.stableEdgels, initialPose_cam, kinectCamera, "stable edges refined");
//    waitKey();
#endif
  }

//  vector<float>::iterator bestPoseIt = std::min_element(poseErrors.begin(), poseErrors.end());
//  int bestPoseIdx = bestPoseIt - poseErrors.begin();
//  PoseRT bestPose = initPoses_cam[bestPoseIdx];
//  initPoses_cam.clear();
//  initPoses_cam.push_back(bestPose);
}

void PoseEstimator::read(const std::string &filename)
{
  FileStorage fs(filename, FileStorage::READ);
  read(fs.root());
  fs.release();
}

void PoseEstimator::read(const cv::FileNode& fn)
{
  params.read(fn);
  kinectCamera.read(fn);
  edgeModel.read(fn);

  silhouettes.clear();
  cv::FileNode silhouettesFN = fn["silhouettes"];
  cv::FileNodeIterator it = silhouettesFN.begin(), it_end = silhouettesFN.end();
  for ( ; it != it_end; ++it)
  {
    Silhouette currentSilhouette;
    currentSilhouette.read(*it);
    silhouettes.push_back(currentSilhouette);
  }
}

void PoseEstimator::write(const std::string &filename) const
{
  FileStorage fs(filename, FileStorage::WRITE);
  write(fs);
  fs.release();
}

void PoseEstimator::write(cv::FileStorage& fs) const
{
  params.write(fs);
  kinectCamera.write(fs);
  edgeModel.write(fs);

  fs << "silhouettes" << "[";
  for (size_t i = 0; i < silhouettes.size(); ++i)
  {
    fs << "{";
    silhouettes[i].write(fs);
    fs << "}";
  }
  fs << "]";
}

void PoseEstimatorParams::read(const FileNode &fileNode)
{
  FileNode fn = fileNode["params"];

  minGlassContourLength = static_cast<int>(fn["minGlassContourLength"]);
  minGlassContourArea = fn["minGlassContourArea"];

  cannyThreshold1 = fn["cannyThreshold1"];
  cannyThreshold2 = fn["cannyThreshold2"];
  dilationsForEdgesRemovalCount = fn["dilationsForEdgesRemovalCount"];

  confidentDomination = fn["confidentDomination"];
}

void PoseEstimatorParams::write(cv::FileStorage &fs) const
{
  fs << "params" << "{";

  fs << "minGlassContourLength" << static_cast<int>(minGlassContourLength);
  fs << "minGlassContourArea" << minGlassContourArea;

  fs << "cannyThreshold1" << cannyThreshold1;
  fs << "cannyThreshold2" << cannyThreshold2;
  fs << "dilationsForEdgesRemovalCount" << dilationsForEdgesRemovalCount;

  fs << "confidentDomination" << confidentDomination;
  fs << "}";
}

void PoseEstimator::visualize(const PoseRT &pose, cv::Mat &image, cv::Scalar color) const
{
  image = drawEdgels(image, edgeModel.points, pose, kinectCamera, color);
}

#ifdef USE_3D_VISUALIZATION
void PoseEstimator::visualize(const PoseRT &pose, const boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, cv::Scalar color, const std::string &title) const
{
  vector<Point3f> object;
  project3dPoints(edgeModel.points, pose.getRvec(), pose.getTvec(), object);
  pcl::PointCloud<pcl::PointXYZ> pclObject;
  cv2pcl(object, pclObject);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> objectColor(pclObject.makeShared(), color[2], color[1], color[0]);
  viewer->addPointCloud<pcl::PointXYZ>(pclObject.makeShared(), objectColor, title);
}
#endif

cv::Size PoseEstimator::getValidTestImageSize() const
{
  return kinectCamera.imageSize;
}

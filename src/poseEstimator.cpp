/*
 * poseEstimator.cpp
 *
 *  Created on: Dec 2, 2011
 *      Author: Ilya Lysenkov
 */

#include "edges_pose_refiner/poseEstimator.hpp"
#include "edges_pose_refiner/localPoseRefiner.hpp"
#include "edges_pose_refiner/tableSegmentation.hpp"
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

//#define VISUALIZE_FINAL_REFINEMENT

//#define VERBOSE

using namespace cv;
using std::cout;
using std::endl;

namespace transpod
{
  PoseEstimator::PoseEstimator(const PinholeCamera &_camera, const PoseEstimatorParams &_params)
  {
    kinectCamera = _camera;
    params = _params;
    ghTable = 0;
  }

  void PoseEstimator::setModel(const EdgeModel &_edgeModel)
  {
    edgeModel = _edgeModel;

    Ptr<const PinholeCamera> centralCameraPtr = new PinholeCamera(kinectCamera);
    edgeModel.generateSilhouettes(centralCameraPtr, params.silhouetteCount, silhouettes, params.downFactor, params.closingIterationsCount);
    generateGeometricHashes();

  }

  EdgeModel PoseEstimator::getModel() const
  {
    return edgeModel;
  }


  void PoseEstimator::generateGeometricHashes()
  {
    ghTable = new GHTable();

    canonicScales.resize(silhouettes.size());
#ifdef VERBOSE
    cout << "number of train silhouettes: " << silhouettes.size() << endl;
#endif
    for (size_t i = 0; i < silhouettes.size(); ++i)
    {
      silhouettes[i].generateGeometricHash(i, *ghTable, canonicScales[i], params.ghGranularity, params.ghBasisStep, params.ghMinDistanceBetweenBasisPoints);
    }

    //These lines allocate memory for the created table more efficiently.
    //Without them detection is slower in ~4x times.
    //TODO: better investigate this issue and fix it in more elegant way.
    Ptr<GHTable> finalGHTable = new GHTable(*ghTable);
    ghTable = finalGHTable;

    //TODO: make key distribution in the table more uniform
  }

  void PoseEstimator::estimatePose(const cv::Mat &kinectBgrImage, const cv::Mat &glassMask, std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities, const cv::Vec4f *tablePlane,
                                   std::vector<cv::Mat> *initialSilhouettes, std::vector<PoseRT> *initialPoses) const
  {
    CV_Assert(kinectBgrImage.size() == glassMask.size());
    CV_Assert(kinectBgrImage.size() == getValidTestImageSize());

    if (silhouettes.empty())
    {
      std::cerr << "PoseEstimator is not initialized" << std::endl;
      return;
    }

  //  getInitialPoses(glassMask, poses_cam, posesQualities);
    getInitialPosesByGeometricHashing(glassMask, poses_cam, posesQualities, initialSilhouettes);
    if (initialPoses != 0)
    {
      *initialPoses = poses_cam;
    }

  //  refineInitialPoses(kinectBgrImage, glassMask, poses_cam, posesQualities);
    if (tablePlane != 0)
    {
      refinePosesBySupportPlane(kinectBgrImage, glassMask, *tablePlane, poses_cam, posesQualities);
    }
  }

  void PoseEstimator::refinePosesBySupportPlane(const cv::Mat &bgrImage, const cv::Mat &glassMask, const cv::Vec4f &tablePlane,
                                                std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities) const
  {
    Mat testEdges, silhouetteEdges;
    computeCentralEdges(bgrImage, glassMask, testEdges, silhouetteEdges);
    refinePosesByTableOrientation(tablePlane, bgrImage, testEdges, silhouetteEdges, poses_cam, posesQualities);
    refineFinalTablePoses(tablePlane, bgrImage, testEdges, silhouetteEdges, poses_cam, posesQualities);
  }

  void PoseEstimator::refineFinalTablePoses(const cv::Vec4f &tablePlane,
                      const cv::Mat &testBgrImage, const cv::Mat &testEdges, const cv::Mat &silhouetteEdges,
                      std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities) const
  {
#ifdef VERBOSE
    cout << "final refinement of poses by table orientation" << endl;
#endif
    if (poses_cam.empty())
    {
      return;
    }

    posesQualities.resize(poses_cam.size());
    LocalPoseRefiner localPoseRefiner(edgeModel, testBgrImage, testEdges, kinectCamera, params.lmFinalParams);
    localPoseRefiner.setSilhouetteEdges(silhouetteEdges);
    for (size_t initPoseIdx = 0; initPoseIdx < poses_cam.size(); ++initPoseIdx)
    {
      PoseRT &initialPose_cam = poses_cam[initPoseIdx];
#ifdef VISUALIZE_FINAL_REFINEMENT
      showEdgels(testEdges, edgeModel.points, initialPose_cam, kinectCamera, "before final refinement");
      showEdgels(testEdges, edgeModel.stableEdgels, initialPose_cam, kinectCamera, "stable before final refinement");
#endif
      posesQualities[initPoseIdx] = localPoseRefiner.refineUsingSilhouette(initialPose_cam, true, tablePlane);
#ifdef VISUALIZE_FINAL_REFINEMENT
      showEdgels(testEdges, edgeModel.points, initialPose_cam, kinectCamera, "after final refinement");
      showEdgels(testEdges, edgeModel.stableEdgels, initialPose_cam, kinectCamera, "stable after final refinement");
      waitKey();
#endif
    }
  }

  void PoseEstimator::refinePosesByTableOrientation(const cv::Vec4f &tablePlane,
                      const cv::Mat &testBgrImage, const cv::Mat &centralEdges, const cv::Mat &silhouetteEdges,
                      vector<PoseRT> &poses_cam, vector<float> &initPosesQualities) const
  {
#ifdef VERBOSE
    cout << "refine poses by table orientation" << endl;
#endif
    if (poses_cam.empty())
    {
      return;
    }

    initPosesQualities.resize(poses_cam.size());
    vector<float> rotationAngles(poses_cam.size());

    LocalPoseRefinerParams lmJacobianParams = params.lmInitialParams;
    lmJacobianParams.termCriteria = params.lmJacobianCriteria;
    vector<Mat> jacobians;
    refineInitialPoses(testBgrImage, centralEdges, silhouetteEdges, poses_cam, initPosesQualities, lmJacobianParams, &jacobians);

    for (size_t initPoseIdx = 0; initPoseIdx < poses_cam.size(); ++initPoseIdx)
    {
  #ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
      cout << "quality[" << initPoseIdx << "]: " << initPosesQualities[initPoseIdx] << endl;
      showEdgels(centralEdges, edgeModel.points, poses_cam[initPoseIdx], kinectCamera, "before alignment to a table plane");
  #endif

      findTransformationToTable(poses_cam[initPoseIdx], tablePlane, rotationAngles[initPoseIdx], jacobians[initPoseIdx]);

  #ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
      showEdgels(centralEdges, edgeModel.points, poses_cam[initPoseIdx], kinectCamera, "after alignment to a table plane");
      cout << "quality[" << initPoseIdx << "]: " << initPosesQualities[initPoseIdx] << endl;
      waitKey();
  #endif
    }

    LocalPoseRefinerParams lmErrorParams = params.lmInitialParams;
    lmErrorParams.termCriteria = params.lmErrorCriteria;
    LocalPoseRefiner localPoseRefiner(edgeModel, testBgrImage, centralEdges, kinectCamera, lmErrorParams);
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
    localPoseRefiner.setParams(params.lmInitialParams);

    vector<bool> isPoseFilteredOut;
    filterOutHighValues(initPosesQualities, params.ratioToMinimum, isPoseFilteredOut);
    suppress3DPoses(initPosesQualities, poses_cam,
                    params.neighborMaxRotation, params.neighborMaxTranslation,
                    isPoseFilteredOut);

    filterValues(poses_cam, isPoseFilteredOut);
    initPosesQualities.resize(poses_cam.size());
#ifdef VERBOSE
    cout << "table suppression: " << isPoseFilteredOut.size() << " -> " << poses_cam.size() << endl;
#endif


    for (size_t initPoseIdx = 0; initPoseIdx < poses_cam.size(); ++initPoseIdx)
    {
      PoseRT &initialPose_cam = poses_cam[initPoseIdx];
      initPosesQualities[initPoseIdx] = localPoseRefiner.refineUsingSilhouette(initialPose_cam, true, tablePlane);

  #ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
      showEdgels(centralEdges, edgeModel.points, initialPose_cam, kinectCamera, "central pose refined by LM with a table plane");
      showEdgels(centralEdges, edgeModel.stableEdgels, initialPose_cam, kinectCamera, "stable edgels refined by LM with a table plane");
      cout << "quality[" << initPoseIdx << "]: " << initPosesQualities[initPoseIdx] << endl;
      cout << "pose[" << initPoseIdx << "]: " << initialPose_cam << endl;
      waitKey();
  #endif
    }

    vector<float>::iterator bestPoseIt = std::min_element(initPosesQualities.begin(), initPosesQualities.end());
    int bestPoseIdx = bestPoseIt - initPosesQualities.begin();
  //  vector<float>::iterator bestPoseIt = std::min_element(rotationAngles.begin(), rotationAngles.end());
  //  int bestPoseIdx = bestPoseIt - rotationAngles.begin();

    PoseRT bestPose = poses_cam[bestPoseIdx];
    float bestPoseQuality = initPosesQualities[bestPoseIdx];
#ifdef VERBOSE
    cout << "best quality: " << bestPoseQuality << endl;
#endif
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
    waitKey();
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
    Mat wideConfidences;
    copyMakeBorder(confidences, wideConfidences, halfWindowSize, halfWindowSize, halfWindowSize, halfWindowSize, BORDER_CONSTANT, 0.0);

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

    vector<Mat> votes(silhouettes.size());
    for (size_t i = 0; i < votes.size(); ++i)
    {
      votes[i] = Mat(silhouettes[i].getDownsampledSize(), silhouettes[i].getDownsampledSize(), CV_32SC1, Scalar(0));
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

      std::pair<GHTable::const_iterator, GHTable::const_iterator> range = ghTable->equal_range(key);
      for(GHTable::const_iterator it = range.first; it != range.second; ++it)
      {
        GHValue value = it->second;
        votes[value[0]].at<int>(value[1], value[2]) += 1;
      }
    }

    for (size_t i = 0; i < votes.size(); ++i)
    {
      Mat currentVotes;
      votes[i].convertTo(currentVotes, CV_32FC1, 1.0 / silhouettes[i].getDownsampledSize());
      Mat currentScale = testScale * canonicScales[i];
      currentVotes /= currentScale;

      vector<Point> maxLocations;
      suppressNonMaximum(currentVotes, params.votesWindowSize, params.votesConfidentSuppression, maxLocations);

      for (size_t j = 0; j < maxLocations.size(); ++j)
      {
        Point maxLoc = maxLocations[j];

        BasisMatch match;
        match.confidence = currentVotes.at<float>(maxLoc.y, maxLoc.x);

        if (currentScale.at<float>(maxLoc.y, maxLoc.x) < params.minScale ||
            currentScale.at<float>(maxLoc.y, maxLoc.x) > params.maxScale)
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

  void PoseEstimator::suppressBasisMatches(std::vector<BasisMatch> &matches) const
  {
    vector<float> errors(matches.size());
    for (size_t i = 0; i < matches.size(); ++i)
    {
      errors[i] = matches[i].confidence;
    }

    vector<bool> isSuppressed;
    filterOutLowValues(errors, 1.0 / params.basisConfidentSuppression, isSuppressed);
    filterValues(matches, isSuppressed);
  }

  void PoseEstimator::suppress3DPoses(const std::vector<float> &errors, const std::vector<PoseRT> &poses_cam,
                                      float neighborMaxRotation, float neighborMaxTranslation,
                                      std::vector<bool> &isFilteredOut) const
  {
    CV_Assert(errors.size() == poses_cam.size());

    if (isFilteredOut.empty())
    {
      isFilteredOut.resize(errors.size(), false);
    }
    else
    {
      CV_Assert(isFilteredOut.size() == errors.size());
    }

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
        //TODO: use rotation symmetry
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
    suppress3DPoses(errors, poses_cam,
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
      silhouettes[matches[i].silhouetteIndex].getDownsampledEdgels(edgels);
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

  //TODO: refine estimate similarities by using all corresponding points
  void PoseEstimator::getInitialPosesByGeometricHashing(const cv::Mat &glassMask, std::vector<PoseRT> &initialPoses, std::vector<float> &initialPosesQualities, std::vector<cv::Mat> *initialSilhouettes) const
  {
#ifdef VERBOSE
    cout << "get initial poses by geometric hashing..." << endl;
#endif
    initialPoses.clear();
    initialPosesQualities.clear();

    Mat maskClone = glassMask.clone();
    vector<vector<Point> > allGlassContours;
    findContours(maskClone, allGlassContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

#ifdef VERBOSE
    cout << "Number of glass contours: " << allGlassContours.size() << endl;
#endif

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

      vector<BasisMatch> basisMatches;
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
        std::copy(currentMatches.begin(), currentMatches.end(), std::back_inserter(basisMatches));
      }

#ifdef VERBOSE
      cout << "before suppression: " << basisMatches.size() << endl;
#endif
      suppressBasisMatches(basisMatches);
#ifdef VERBOSE
      cout << "after suppression: " << basisMatches.size() << endl;
#endif
      estimateSimilarityTransformations(currentContour, basisMatches);
      estimatePoses(basisMatches);

#ifdef VERBOSE
      cout << "before 3d: " << basisMatches.size() << endl;
#endif
      suppressBasisMatchesIn3D(basisMatches);
#ifdef VERBOSE
      cout << "after 3d: " << basisMatches.size() << endl;
#endif


#ifdef VISUALIZE_GEOMETRIC_HASHING
      vector<std::pair<float, int> > sortedMatches;
      for (size_t i = 0; i < basisMatches.size(); ++i)
      {
        sortedMatches.push_back(std::make_pair(basisMatches[i].confidence, i));
      }

      std::sort(sortedMatches.begin(), sortedMatches.end());
      std::reverse(sortedMatches.begin(), sortedMatches.end());
#endif

      for (size_t i = 0; i < basisMatches.size(); ++i)
      {
  //      PoseRT pose;
  //      silhouettes[basisMatches[i].silhouetteIndex].affine2poseRT(edgeModel, kinectCamera, similarityTransformations_cam[i], params.useClosedFormPnP, pose);
  //      initialPoses.push_back(pose);
  //      initialPosesQualities.push_back(-basisMatches[i].confidence);

        initialPoses.push_back(basisMatches[i].pose);
        initialPosesQualities.push_back(-basisMatches[i].confidence);

        if (initialSilhouettes != 0)
        {
          Mat edgels;
          silhouettes[basisMatches[i].silhouetteIndex].getEdgels(edgels);
          Mat transformedEdgels;
          transform(edgels, transformedEdgels, basisMatches[i].similarityTransformation_cam);
          initialSilhouettes->push_back(transformedEdgels);
        }

  #ifdef VISUALIZE_GEOMETRIC_HASHING
        int matchIdx = sortedMatches[i].second;
        Mat visualization = glassMask.clone();
        silhouettes[basisMatches[matchIdx].silhouetteIndex].visualizeSimilarityTransformation(basisMatches[matchIdx].similarityTransformation_cam, visualization, Scalar(255, 0, 255));
        imshow("transformation by geometric hashing", visualization);

        cout << "votes: " << basisMatches[matchIdx].confidence << endl;
        cout << "idx: " << basisMatches[matchIdx].silhouetteIndex << endl;
        cout << "matchIdx" << matchIdx << endl;
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
    }
#ifdef VERBOSE
    cout << "Initial pose count: " << initialPoses.size() << endl;
#endif
  }

  void PoseEstimator::suppressNonMinimum(std::vector<float> errors, float absoluteSuppressionFactor, std::vector<bool> &isSuppressed, bool useNeighbors)
  {
    isSuppressed.resize(errors.size(), false);
    float minError = *std::min_element(errors.begin(), errors.end());

    for (size_t i = 0; i < errors.size(); ++i)
    {
      if (minError * absoluteSuppressionFactor < errors[i])
      {
        isSuppressed[i] = true;
      }
    }

    if (useNeighbors)
    {
      for (size_t i = 0; i < errors.size(); ++i)
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

/*
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
  */

  void PoseEstimator::refineInitialPoses(const cv::Mat &testBgrImage, const cv::Mat &centralEdges, const cv::Mat &silhouetteEdges,
                                         vector<PoseRT> &initPoses_cam, vector<float> &initPosesQualities,
                                         const LocalPoseRefinerParams &lmParams, vector<cv::Mat> *jacobians) const
  {
#ifdef VERBOSE
    cout << "refine initial poses" << endl;
#endif
    if (initPoses_cam.empty())
    {
      return;
    }

    initPosesQualities.resize(initPoses_cam.size());
    if (jacobians != 0)
    {
      jacobians->resize(initPoses_cam.size());
    }

    LocalPoseRefiner localPoseRefiner(edgeModel, testBgrImage, centralEdges, kinectCamera, lmParams);
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
    const cv::FileNode silhouettesFN = fn["silhouettes"];
    for (cv::FileNodeIterator it = silhouettesFN.begin(); it != silhouettesFN.end(); ++it)
    {
      Silhouette currentSilhouette;
      currentSilhouette.read(*it);
      silhouettes.push_back(currentSilhouette);
    }

    canonicScales.clear();
    const cv::FileNode canonicScalesFN = fn["canonicScales"];
    for (cv::FileNodeIterator it = canonicScalesFN.begin(); it != canonicScalesFN.end(); ++it)
    {
      Mat currentScale;
      (*it) >> currentScale;
      canonicScales.push_back(currentScale);
    }

/*
    votes.clear();
    const cv::FileNode votesFN = fn["votes"];
    for (cv::FileNodeIterator it = votesFN.begin(); it != votesFN.end(); ++it)
    {
      Mat currentVote;
      (*it) >> currentVote;
      votes.push_back(currentVote);
    }
*/

    ghTable = new GHTable();
    Mat hashTable;
    fn["hash_table"] >> hashTable;
    for (int elementIndex = 0; elementIndex < hashTable.rows; ++elementIndex)
    {
      Mat row = hashTable.row(elementIndex);
      std::pair<GHKey, GHValue> tableElement(std::make_pair(row.at<int>(0), row.at<int>(1)),
                                             GHValue(row.at<int>(2), row.at<int>(3), row.at<int>(4)));
      ghTable->insert(tableElement);
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

    fs << "canonicScales" << "[";
    for (size_t i = 0; i < canonicScales.size(); ++i)
    {
      fs << canonicScales[i];
    }
    fs << "]";

/*
    fs << "votes" << "[";
    for (size_t i = 0; i < votes.size(); ++i)
    {
      fs << votes[i];
    }
    fs << "]";
*/

    Mat hash_table(ghTable->size(), GH_KEY_DIMENSION + GHValue::channels, CV_32SC1);
    CV_Assert(GHValue::depth == CV_32S);
    int elementIndex = 0;
    for (GHTable::const_iterator it = ghTable->begin(); it != ghTable->end(); ++it, ++elementIndex)
    {
      hash_table.at<int>(elementIndex, 0) = it->first.first;
      hash_table.at<int>(elementIndex, 1) = it->first.second;
      for (int ch = 0; ch < it->second.channels; ++ch)
      {
        hash_table.at<int>(elementIndex, GH_KEY_DIMENSION + ch) = it->second[ch];
      }
    }
    fs << "hash_table" << hash_table;
  }

  void PoseEstimatorParams::read(const FileNode &fileNode)
  {
    FileNode fn = fileNode["params"];

    minGlassContourLength = static_cast<int>(fn["minGlassContourLength"]);
    minGlassContourArea = fn["minGlassContourArea"];

    cannyThreshold1 = fn["cannyThreshold1"];
    cannyThreshold2 = fn["cannyThreshold2"];
    dilationsForEdgesRemovalCount = fn["dilationsForEdgesRemovalCount"];

 //   confidentDomination = fn["confidentDomination"];
  }

  void PoseEstimatorParams::write(cv::FileStorage &fs) const
  {
    fs << "params" << "{";

    fs << "minGlassContourLength" << static_cast<int>(minGlassContourLength);
    fs << "minGlassContourArea" << minGlassContourArea;

    fs << "cannyThreshold1" << cannyThreshold1;
    fs << "cannyThreshold2" << cannyThreshold2;
    fs << "dilationsForEdgesRemovalCount" << dilationsForEdgesRemovalCount;

//    fs << "confidentDomination" << confidentDomination;
    fs << "}";
  }

  void PoseEstimator::visualize(const PoseRT &pose, cv::Mat &image,
                                cv::Scalar color, float blendingFactor) const
  {
    image = drawEdgels(image, edgeModel.points, pose, kinectCamera, color, blendingFactor);
  }

  float PoseEstimator::computeBlendingFactor(float error) const
  {
      //TODO: move up
      const float alpha = -5.5f;
      const float beta = -3.3f;

      //sigmoid function
      float blendingFactor = 1.0 / (1.0 + exp(-alpha * error + beta));
      return blendingFactor;
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
} //end of namespace transpod

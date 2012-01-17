/*
 * poseEstimator.cpp
 *
 *  Created on: Dec 2, 2011
 *      Author: Ilya Lysenkov
 */

#include "edges_pose_refiner/poseEstimator.hpp"
#include "edges_pose_refiner/localPoseRefiner.hpp"
#include "edges_pose_refiner/pclProcessing.hpp"

#ifdef USE_3D_VISUALIZATION
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#endif

#include <opencv2/opencv.hpp>
#include <boost/thread/thread.hpp>

//#define VISUALIZE_TABLE_ESTIMATION

//#define VISUALIZE_INITIAL_POSE_REFINEMENT

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
}

void PoseEstimator::estimatePose(const cv::Mat &kinectBgrImage, const cv::Mat &glassMask, std::vector<PoseRT> &poses_cam, std::vector<float> &posesQualities, const cv::Vec4f *tablePlane) const
{
  CV_Assert(kinectBgrImage.size() == glassMask.size());
  CV_Assert(kinectBgrImage.size() == getValitTestImageSize());

  if (silhouettes.empty())
  {
    std::cerr << "PoseEstimator is not initialized" << std::endl;
    return;
  }

  getInitialPoses(glassMask, poses_cam, posesQualities);
  refineInitialPoses(kinectBgrImage, glassMask, poses_cam, posesQualities);
  if (tablePlane != 0)
  {
    refinePosesByTableOrientation(*tablePlane, kinectBgrImage, glassMask, poses_cam, posesQualities);
  }
}

void PoseEstimator::refinePosesByTableOrientation(const cv::Vec4f &tablePlane, const cv::Mat &centralBgrImage, const cv::Mat &glassMask, vector<PoseRT> &poses_cam, vector<float> &initPosesQualities) const
{
  cout << "refine poses by table orientation" << endl;
  if (poses_cam.empty())
  {
    return;
  }

  Mat centralEdges, silhouetteEdges;
  computeCentralEdges(centralBgrImage, glassMask, centralEdges, silhouetteEdges);

  initPosesQualities.resize(poses_cam.size());
  vector<float> rotationAngles(poses_cam.size());
  for (size_t initPoseIdx = 0; initPoseIdx < poses_cam.size(); ++initPoseIdx)
  {
//    cout << "Pose idx: " << initPoseIdx << endl;
    LocalPoseRefiner localPoseRefiner(edgeModel, centralEdges, kinectCamera.cameraMatrix, kinectCamera.distCoeffs, kinectCamera.extrinsics.getProjectiveMatrix());
    localPoseRefiner.setSilhouetteEdges(silhouetteEdges);
    PoseRT &initialPose_cam = poses_cam[initPoseIdx];

    Mat finalJacobian;
    initPosesQualities[initPoseIdx] = localPoseRefiner.refineUsingSilhouette(initialPose_cam, true, Vec4f::all(0.0f), &finalJacobian);

#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
    displayEdgels(centralEdges, edgeModel.points, initialPose_cam, kinectCamera, "before alignment to a table plane");

//    if (pt_pub != 0)
//    {
//      publishPoints(edgeModel.points, initialPose_cam.getRvec(), initialPose_cam.getTvec(), *pt_pub, 1, Scalar(0, 0, 255));
//      namedWindow("ready to align pose to a table plane");
//      waitKey();
//      destroyWindow("ready to align pose to a table plane");
//    }
#endif

    findTransformationToTable(initialPose_cam, tablePlane, rotationAngles[initPoseIdx], finalJacobian);

#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
    displayEdgels(centralEdges, edgeModel.points, initialPose_cam, kinectCamera, "after alignment to a table plane");
//    if (pt_pub != 0)
//    {
//      publishPoints(edgeModel.points, initialPose_cam.getRvec(), initialPose_cam.getTvec(), *pt_pub, 1, Scalar(0, 0, 255));
//      namedWindow("aligned pose to a table plane");
//      waitKey();
//      destroyWindow("aligned pose to a table plane");
//    }
#endif

    initPosesQualities[initPoseIdx] = localPoseRefiner.refineUsingSilhouette(initialPose_cam, true, tablePlane);

#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
    displayEdgels(centralEdges, edgeModel.points, initialPose_cam, kinectCamera, "central pose refined by LM with a table plane");
    displayEdgels(centralEdges, edgeModel.stableEdgels, initialPose_cam, kinectCamera, "stable edgels refined by LM with a table plane");
    waitKey();
#endif
  }

  vector<float>::iterator bestPoseIt = std::min_element(initPosesQualities.begin(), initPosesQualities.end());
  int bestPoseIdx = bestPoseIt - initPosesQualities.begin();
//  vector<float>::iterator bestPoseIt = std::min_element(rotationAngles.begin(), rotationAngles.end());
//  int bestPoseIdx = bestPoseIt - rotationAngles.begin();

  PoseRT bestPose = poses_cam[bestPoseIdx];
  float bestPoseQuality = initPosesQualities[bestPoseIdx];
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

    vector<bool> isDominated(silhouettes.size(), false);
    vector<PoseRT> poses(silhouettes.size());
    for (size_t i = 0; i < silhouettes.size(); ++i)
    {
      for (size_t j = 0; j < silhouettes.size(); ++j)
      {
        if (i == j)
        {
          continue;
        }

        if (poseQualities[j] * params.confidentDomination < poseQualities[i])
        {
          isDominated[i] = true;
          break;
        }
      }

      if (isDominated[i])
      {
        continue;
      }

      bool isNextBetter = poseQualities[i] > poseQualities[(i + 1) % silhouettes.size()];
      bool isPreviousBetter = poseQualities[i] > poseQualities[(static_cast<int>(silhouettes.size() + i) - 1) % silhouettes.size()];

      if (isNextBetter || isPreviousBetter)
      {
        isDominated[i] = true;
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
      if (isDominated[i])
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

void PoseEstimator::refineInitialPoses(const cv::Mat &centralBgrImage, const cv::Mat &glassMask, vector<PoseRT> &initPoses_cam, vector<float> &initPosesQualities) const
{
  cout << "refine initial poses" << endl;
  if (initPoses_cam.empty())
  {
    return;
  }

  Mat centralEdges, silhouetteEdges;
  computeCentralEdges(centralBgrImage, glassMask, centralEdges, silhouetteEdges);

  initPosesQualities.resize(initPoses_cam.size());
  for (size_t initPoseIdx = 0; initPoseIdx < initPoses_cam.size(); ++initPoseIdx)
  {
    //cout << "Pose idx: " << initPoseIdx << endl;
    LocalPoseRefiner localPoseRefiner(edgeModel, centralEdges, kinectCamera.cameraMatrix, kinectCamera.distCoeffs, kinectCamera.extrinsics.getProjectiveMatrix());
    localPoseRefiner.setSilhouetteEdges(silhouetteEdges);
    PoseRT &initialPose_cam = initPoses_cam[initPoseIdx];

#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
    displayEdgels(centralEdges, edgeModel.points, initialPose_cam, kinectCamera, "central pose");
    displayEdgels(centralEdges, edgeModel.stableEdgels, initialPose_cam, kinectCamera, "stable edgels");
    waitKey();
#endif
    initPosesQualities[initPoseIdx] = localPoseRefiner.refineUsingSilhouette(initialPose_cam, true);
#ifdef VISUALIZE_INITIAL_POSE_REFINEMENT
    displayEdgels(centralEdges, edgeModel.points, initialPose_cam, kinectCamera, "central pose refined");
    displayEdgels(centralEdges, edgeModel.stableEdgels, initialPose_cam, kinectCamera, "stable edges refined");
    waitKey();
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
  image = displayEdgels(image, edgeModel.points, pose, kinectCamera, "", color);
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

cv::Size PoseEstimator::getValitTestImageSize() const
{
  return kinectCamera.imageSize;
}

#include "edges_pose_refiner/edgesPoseRefiner.hpp"
#include "edges_pose_refiner/localPoseRefiner.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iostream>
#include <fstream>
#include <set>
#include <opencv2/highgui/highgui.hpp>

using std::cout;
using std::endl;

using namespace cv;

#define VERBOSE

double estimatePoseQuality(const std::vector<double> &x, std::vector<double> &grad, void* f_data);

EdgesPoseRefiner::EdgesPoseRefiner(const EdgeModel &_edgeModel, const PinholeCamera &_camera, const EdgesPoseRefinerParams &_params)
{
  vector<PinholeCamera> _allCameras(1, _camera);
  initEdgesPoseRefiner(_edgeModel, _allCameras, _params);
}

EdgesPoseRefiner::EdgesPoseRefiner(const EdgeModel &_edgeModel, const std::vector<PinholeCamera> &_allCameras, const EdgesPoseRefinerParams &_params)
{
  initEdgesPoseRefiner(_edgeModel, _allCameras, _params);
}

void EdgesPoseRefiner::initEdgesPoseRefiner(const EdgeModel &_edgeModel, const std::vector<PinholeCamera> &_allCameras, const EdgesPoseRefinerParams &_params)
{
  dim = 3;
  edgeModel = _edgeModel;
  allCameras = _allCameras;
  params = _params;
  CV_Assert(params.maxTranslations.size() == params.maxRotationAngles.size());
}

void EdgesPoseRefiner::setParams(const EdgesPoseRefinerParams &_params)
{
  params = _params;
  CV_Assert(params.maxTranslations.size() == params.maxRotationAngles.size());
}

void EdgesPoseRefiner::setCenterMask(const PinholeCamera &_centerCamera, const cv::Mat &mask)
{
  centerCamera = _centerCamera;
  centerMask = mask;
}

void EdgesPoseRefiner::findTransformationToTable(cv::Mat &rvecFinal_cam, cv::Mat &tvecFinal_cam, const cv::Vec4f &tablePlane)
{
  Mat Rt_init;
  createProjectiveMatrix(rvecFinal_cam, tvecFinal_cam, Rt_init);
  Mat Rotation_init = Rt_init.clone();
  Rotation_init(Range(0, 3), Range(3, 4)).setTo(Scalar(0));

  Point3d rotatedAxis;
  transformPoint(Rt_init, edgeModel.rotationAxis, rotatedAxis);
  Point3d tableNormal(tablePlane[0], tablePlane[1], tablePlane[2]);

  double cosphi = rotatedAxis.ddot(tableNormal) / (norm(rotatedAxis) * norm(tableNormal));
  double phi = acos(cosphi);
  Point3d rvecPt = rotatedAxis.cross(tableNormal);
  rvecPt = rvecPt * (phi / norm(rvecPt));
  Mat rvec = Mat(rvecPt).reshape(1, dim);
  Mat R;
  Rodrigues(rvec, R);
  Mat zeroVec = Mat::zeros(dim, 1, CV_64FC1);
  Mat R_Rt;
  createProjectiveMatrix(R, zeroVec, R_Rt);

  Point3d transformedTableAnchor;
  transformPoint(R_Rt*Rt_init, edgeModel.tableAnchor, transformedTableAnchor);

  //project transformedTableAnchor on the table plane
  double alpha = -(tablePlane[3] + tableNormal.ddot(transformedTableAnchor)) / tableNormal.ddot(tableNormal);
  Point3d anchorOnTable = transformedTableAnchor + alpha * tableNormal;

  Point3d tvecPt = anchorOnTable - transformedTableAnchor;
  Mat tvec = Mat(tvecPt).reshape(1, dim);

  rvecFinal_cam = rvec;
  tvecFinal_cam = tvec;
}

double EdgesPoseRefiner::refine(const cv::Mat &testEdges, cv::Mat &rvec, cv::Mat &tvec, bool usePoseGuess) const
{
  Mat rvecGlobal_cam, tvecGlobal_cam;
  return refine(testEdges, rvec, tvec, rvecGlobal_cam, tvecGlobal_cam, usePoseGuess);
}

double EdgesPoseRefiner::refine(const cv::Mat &testEdges, cv::Mat &rvecFinal_cam, cv::Mat &tvecFinal_cam, cv::Mat &rvecGlobal_cam, cv::Mat &tvecGlobal_cam, bool usePoseGuess, const cv::Vec4f &tablePlane) const
{
  vector<Mat> allTestEdges(1, testEdges);
  return refine(allTestEdges, rvecFinal_cam, tvecFinal_cam, rvecGlobal_cam, tvecGlobal_cam, usePoseGuess);
}

double EdgesPoseRefiner::refine(const std::vector<cv::Mat> &testEdges, cv::Mat &rvecFinal_cam, cv::Mat &tvecFinal_cam, bool usePoseGuess, const cv::Vec4f &tablePlane) const
{
  Mat rvecGlobal_cam, tvecGlobal_cam;
  return refine(testEdges, rvecFinal_cam, tvecFinal_cam, rvecGlobal_cam, tvecGlobal_cam, usePoseGuess, tablePlane);
}

void EdgesPoseRefiner::addPoseQualityEstimator(const EdgeModel &edgeModel, const Mat &edges, const Mat &cameraMatrix, const Mat &distCoeffs, const Mat &extrinsicsRt, const Mat &rvec_cam, const Mat &tvec_cam, vector<Ptr<PoseQualityEstimator> > &poseQualityEstimators, bool usePoseGuess, const Mat centerMask) const
{
  Ptr<LocalPoseRefiner> localPoseRefiner = new LocalPoseRefiner(edgeModel, edges, cameraMatrix, distCoeffs, extrinsicsRt, params.localParams);
  if(!centerMask.empty())
  {
    localPoseRefiner->setCenterMask(centerMask);
  }
  Ptr<PoseQualityEstimator> poseQualityEstimator = new PoseQualityEstimator(localPoseRefiner, params.hTrimmedError);
  poseQualityEstimators.push_back(poseQualityEstimator);
}

double EdgesPoseRefiner::refine(const std::vector<cv::Mat> &testEdges, cv::Mat &rvecFinal_cam, cv::Mat &tvecFinal_cam, cv::Mat &rvecGlobal_cam, cv::Mat &tvecGlobal_cam, bool usePoseGuess, const cv::Vec4f &tablePlane) const
{
#ifdef VERBOSE
  std::cout << "Start refinement... " << endl;
#endif

  CV_Assert(testEdges.size() == allCameras.size());
  vector<Ptr<LocalPoseRefiner> > localPoseRefiners(testEdges.size());
  vector<Ptr<PoseQualityEstimator> > poseQualityEstimators;
  for(size_t i=0; i<testEdges.size(); i++)
  {
    addPoseQualityEstimator(edgeModel, testEdges[i], allCameras[i].cameraMatrix, allCameras[i].distCoeffs, allCameras[i].extrinsics.getProjectiveMatrix(), rvecFinal_cam, tvecFinal_cam, poseQualityEstimators, usePoseGuess);
  }
  if(!centerMask.empty())
  {
    addPoseQualityEstimator(edgeModel, testEdges[0], centerCamera.cameraMatrix, centerCamera.distCoeffs, centerCamera.extrinsics.getProjectiveMatrix(), rvecFinal_cam, tvecFinal_cam, poseQualityEstimators, usePoseGuess, centerMask);
  }

  rvecGlobal_cam = rvecFinal_cam.clone();
  tvecGlobal_cam = tvecFinal_cam.clone();
  double globalMinf = -1.0;
  for (size_t i = 0; i < params.maxTranslations.size(); ++i)
  {
    if(usePoseGuess)
    {
      for (size_t j = 0; j < poseQualityEstimators.size(); ++j)
      {
        poseQualityEstimators[j]->setInitialPose(PoseRT(rvecGlobal_cam, tvecGlobal_cam));
      }
    }
    globalMinf = runGlobalOptimization(&poseQualityEstimators, i, tablePlane, rvecGlobal_cam, tvecGlobal_cam);
  }

  rvecGlobal_cam.copyTo(rvecFinal_cam);
  tvecGlobal_cam.copyTo(tvecFinal_cam);
  if(!params.localParams.useViewDependentEdges)
  {
    //TODO: use several cameras
    //localPoseRefiner.refine(rvecFinal_cam, tvecFinal_cam, true);
    localPoseRefiners[0]->refine(rvecFinal_cam, tvecFinal_cam, true);
  }

  return globalMinf;
}

double estimatePoseQuality(const std::vector<double> &x, std::vector<double> &grad, void* f_data)
{
  vector<Ptr<PoseQualityEstimator> > *estimators = static_cast<vector<Ptr<PoseQualityEstimator> > *>(f_data);
  double result = 0.0;

  int lastIdx = static_cast<int>(estimators->size()) - 1;
  for(int i=lastIdx; i >= 0; i--)
  {
    double val = estimators->at(i)->evaluate(x);

    if(val >= std::numeric_limits<double>::max())
    {
      result = std::numeric_limits<double>::infinity();
      break;
    }

    result += val;
  }

  return result;
}

double EdgesPoseRefiner::runGlobalOptimization(std::vector<cv::Ptr<PoseQualityEstimator> > *estimators, size_t iteration, const cv::Vec4f &tablePlane, cv::Mat &rvec_cam, cv::Mat &tvec_cam) const
{
  const float eps = 1e-4;
  const bool useTable = cv::norm(tablePlane) > eps;
  const int problemDim = useTable ? dim : 2*dim;

  nlopt::opt opt(params.globalOptimizationAlgorithm, problemDim);
  setBounds(opt, iteration);

  opt.set_min_objective(estimatePoseQuality, estimators);
  opt.set_ftol_abs(params.absoluteToleranceOnFunctionValue);
  opt.set_maxeval(params.maxNumberOfFunctionEvaluations);
  opt.set_maxtime(params.maxTime);


  std::vector<double> xmin(problemDim, 0);
  double minf;
  //nlopt::result result = opt.optimize(xmin, minf);
  opt.optimize(xmin, minf);

#ifdef VERBOSE
//  cout << " NLOPT result: " << result << endl;
  cout << " minimum by NLOPT: " << minf << endl;
  cout << " min x: " << Mat(xmin) << endl;
#endif

  estimators->at(0)->obj2cam(xmin, rvec_cam, tvec_cam);

  return minf;
}

void EdgesPoseRefiner::setBounds(nlopt::opt &opt, size_t iteration) const
{
  const int problemDim = static_cast<int>(opt.get_dimension());
  CV_Assert(problemDim == dim || problemDim == 2*dim);
  //TODO: use polar coordinates
  std::vector<double> lb(problemDim), ub(problemDim);
  for(int i=0;i<dim;i++)
  {
    if(problemDim == 2*dim)
    {
      lb[i] = -params.maxRotationAngles[iteration];
      lb[i+dim] = -params.maxTranslations[iteration];
      ub[i] = params.maxRotationAngles[iteration];
      ub[i+dim] = params.maxTranslations[iteration];
    }

    if(problemDim == dim)
    {
      lb[i] = (i == dim-1) ? -params.maxTranslationZ : -params.maxTranslations[iteration];
      ub[i] = (i == dim-1) ? params.maxTranslationZ : params.maxTranslations[iteration];
    }
  }
  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
}


/*
 * poseError.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Ilya Lysenkov
 */

#include "edges_pose_refiner/poseError.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using std::cout;
using std::endl;

PoseRT PoseError::getPosesDifference() const
{
  return posesDifference;
}

double PoseError::getTranslationDifference() const
{
  return translationDiff;
}

double PoseError::getRotationDifference(bool useRadians) const
{
  const double rad2deg = 180.0 / CV_PI;
  return (useRadians ? rotationDifference : rotationDifference * rad2deg);
}

double PoseError::getDifference() const
{
  return totalDiff;
}

std::ostream& operator<<(std::ostream& output, const PoseError& poseError)
{
  const double rad2deg = 180.0 / CV_PI;
  output << "trans. (m): " << poseError.translationDiff << "   rot (deg): " << poseError.rotationDifference * rad2deg;
  return output;
}

PoseError::PoseError()
{
  init(PoseRT(), 0.0, 0.0);
}

void PoseError::init(const PoseRT &_posesDifference, double _rotationDifference, double _translationDifference)
{
  posesDifference = _posesDifference;
  //TODO: was it a bug?
//  rotationDifference = std::min(_rotationDifference, CV_PI - _rotationDifference);
  rotationDifference = std::min(_rotationDifference, 2 * CV_PI - _rotationDifference);
  CV_Assert(rotationDifference >= 0);
  translationDiff = _translationDifference;
  computeSingleCriteria();
}

bool PoseError::operator<(const PoseError &error) const
{
  return totalDiff < error.totalDiff;
}

void PoseError::computeSingleCriteria()
{
  const double rad2deg = 180.0 / CV_PI;
//  const double cm2deg = 30.0;
  const double deg2cm = 1.0 / 30.0;
  const double meter2cm = 100.0;

  double rotationDiffInDegs = rad2deg * rotationDifference;
  totalDiff = meter2cm * translationDiff  + deg2cm * rotationDiffInDegs;
}

PoseError PoseError::operator+(const PoseError &poseError) const
{
  PoseError result = *this;
  result += poseError;
  return result;
}

PoseError& PoseError::operator+=(const PoseError &poseError)
{
  translationDiff += poseError.translationDiff;
  rotationDifference += poseError.rotationDifference;
  totalDiff += poseError.totalDiff;
  return *this;
}

PoseError& PoseError::operator/=(int number)
{
  CV_Assert(number != 0);
  translationDiff /= number;
  rotationDifference /= number;
  totalDiff /= number;
  return *this;
}

void PoseError::computeStats(const vector<PoseError> &poses, double cmThreshold, PoseError &meanError, float &successRate, vector<bool> &isSuccessful)
{
  const double cm2meter = 0.01;
  double meterThreshold = cm2meter * cmThreshold;
  meanError = PoseError();

  int goodPoseCount = 0;
  isSuccessful.resize(poses.size());
  for (size_t i = 0; i < poses.size(); ++i)
  {
    isSuccessful[i] = poses[i].translationDiff < meterThreshold;
    if (isSuccessful[i])
    {
      meanError += poses[i];
      ++goodPoseCount;
    }
  }
  if (goodPoseCount != 0)
  {
    meanError /= goodPoseCount;
  }
  CV_Assert(poses.size() != 0);
  successRate = static_cast<float>(goodPoseCount) / poses.size();
}

void PoseError::evaluateErrors(const std::vector<PoseError> &poseErrors, double cmThreshold)
{
  cout << "Best poses (" << poseErrors.size() << "):" << endl;
  vector<PoseError> sortedBestPoses = poseErrors;
  std::sort(sortedBestPoses.begin(), sortedBestPoses.end());
  for (size_t i = 0; i < sortedBestPoses.size(); ++i)
  {
    cout << sortedBestPoses[i] << endl;
  }

  PoseError meanError;
  float successRate;
  vector<bool> isSuccessful;
  PoseError::computeStats(poseErrors, cmThreshold, meanError, successRate, isSuccessful);

  cout << "Success rate: " << successRate << endl;
  cout << "Mean error: " << meanError << endl;


  vector<PoseRT> poseDiffs;
  for (size_t i = 0; i < poseErrors.size(); ++i)
  {
    if (isSuccessful[i])
    {
      poseDiffs.push_back(poseErrors[i].getPosesDifference());
    }
  }


  PoseRT meanPose;
  PoseRT::computeMeanPose(poseDiffs, meanPose);
//  cout << "Mean pose: " << endl;
//  cout << meanPose << endl;


  double meanRvecError = 0.0;
  double meanTvecError = 0.0;
  int termsCount = 0;
  for (size_t i = 0; i < poseDiffs.size(); ++i)
  {
    double rotationDistance, translationDistance;
    //PoseRT::computeDistance(poseDiffs[i], meanPose, rotationDistance, translationDistance);
    //TODO: use rotation symmetry
    PoseRT::computeObjectDistance(poseDiffs[i], meanPose, rotationDistance, translationDistance);

    meanRvecError += rotationDistance;
    meanTvecError += translationDistance;
    ++termsCount;
  }
  if (termsCount != 0)
  {
    meanRvecError /= termsCount;
    meanTvecError /= termsCount;
  }

//  cout << "norm(mean rvec): " << norm(meanPose.rvec) << endl;
//  cout << "norm(mean tvec): " << norm(meanPose.tvec) << endl;

  cout << "Mean relative error: " << "trans. (m): " << meanTvecError << "  " << "rot (deg): " << meanRvecError * 180 / CV_PI << endl;
}

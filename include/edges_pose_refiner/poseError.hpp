/*
 * poseError.hpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef POSEERROR_HPP_
#define POSEERROR_HPP_

#include "edges_pose_refiner/poseRT.hpp"

class PoseError
{
public:
  PoseError();
  void init(const PoseRT &posesDifference, double rotationDifference, double translationDifference);

  bool operator<(const PoseError &error) const;

  friend std::ostream& operator<<(std::ostream& output, const PoseError& poseError);

  static void computeStats(const std::vector<PoseError> &poses, double cmThreshold, PoseError &meanError, float &successRate, std::vector<bool> &isSuccessful);
  static void evaluateErrors(const std::vector<PoseError> &poseErrors, double cmThreshold);

  PoseRT getPosesDifference() const;
  double getTranslationDifference() const;
  double getRotationDifference(bool useRadians = true) const;
  double getDifference() const;
private:
  void computeSingleCriteria();
  PoseError operator+(const PoseError &poseError) const;
  PoseError& operator+=(const PoseError &poseError);
  PoseError& operator/=(int number);

  double translationDiff;
  double rotationDifference;
  double totalDiff;
  PoseRT posesDifference;
};



#endif /* POSEERROR_HPP_ */

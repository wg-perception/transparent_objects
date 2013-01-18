/*
 * ilpProblem.hpp
 *
 *  Created on: 1/18/2013
 *      Author: ilysenkov
 */

#ifndef ILPPROBLEM_HPP
#define ILPPROBLEM_HPP

#include "edges_pose_refiner/pinholeCamera.hpp"

struct VolumeParams
{
  cv::Vec3f minBound, maxBound, step;

  VolumeParams()
  {
    minBound = cv::Vec3f(0.19f, -0.00f, -0.15f);
    maxBound = cv::Vec3f(0.30f,  0.14f,  0.0f);

//    step = cv::Vec3f::all(0.0025f);
    step = cv::Vec3f::all(0.005f);
//    step = cv::Vec3f::all(0.02f);
  }
};

struct VolumeVariable
{
    int ilpIndex;
    int volumeIndex;
    float label;
};

struct PixelVariable
{
    int ilpIndex;
    int imageIndex;
    int x, y;
    float label;
};

struct Constraint
{
    std::map<int, float> coefficients;
    int b;
};

class ILPProblem
{
public:
    ILPProblem(const VolumeParams &volumeParams, const PinholeCamera &camera);
    ILPProblem(const VolumeParams &volumeParams, const PinholeCamera &camera,
               const std::vector<PoseRT> &allPoses, const std::vector<cv::Mat> &allMasks);

    void getModel(std::vector<cv::Point3f> &model) const;
    void visualize(const std::vector<PoseRT> &allPoses, const std::vector<cv::Mat> &allMasks) const;

    void write(const std::string &filename) const;
    void read(const std::string &problemInstanceFilename, const std::string &solutionFilename);
private:
    bool isSolved;
    VolumeParams volumeParams;
    PinholeCamera camera;
    cv::Mat volumePoints, volumePoints_Vector;
    std::vector<VolumeVariable> volumeVariables;
    std::vector<PixelVariable> pixelVariables;
    std::vector<Constraint> constraints;
};

#endif // ILPPROBLEM_HPP

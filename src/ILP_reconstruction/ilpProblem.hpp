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
//    step = cv::Vec3f::all(0.005f);
    step = cv::Vec3f::all(0.02f);
  }
};

struct VolumeVariable
{
    int ilpIndex;
    int volumeIndex;
    float label;

    VolumeVariable()
    {
        label = 0.0f;
    }
};

struct PixelVariable
{
    int ilpIndex;
    int imageIndex;
    int x, y;
    float label;

    PixelVariable()
    {
        label = 0.0f;
    }
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
    void setGroundTruthModel(const std::vector<cv::Point3f> &groundTruthModel);
    void visualize(const std::vector<PoseRT> &allPoses, const std::vector<cv::Mat> &allMasks) const;
    void visualizeVolumeVariables() const;

    void write(const std::string &filename) const;
    void writeMPS(const std::string &filename) const;
    void writeLP(const std::string &filename) const;
    void read(const std::string &problemInstanceFilename, const std::string &solutionFilename);
private:
    void readProblemFormulation(const std::string &problemInstanceFilename);
    void readSolution(const std::string &solutionFilename);
    void supperimposeGroundTruth(const cv::Mat &image3d, cv::Mat &supperimposedImage) const;

    bool isSolved;
    VolumeParams volumeParams;
    PinholeCamera camera;
    cv::Mat volumePoints, volumePoints_Vector;

    std::vector<VolumeVariable> volumeVariables;
    std::vector<PixelVariable> pixelVariables;
    std::vector<Constraint> constraints;

    std::vector<cv::Point3f> groundTruthModel;
};

#endif // ILPPROBLEM_HPP

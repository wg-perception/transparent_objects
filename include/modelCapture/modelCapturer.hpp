#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/poseRT.hpp"
#include "edges_pose_refiner/pinholeCamera.hpp"
#include "../../src/mrf/GCoptimization.h"

struct VolumeParams
{
  cv::Vec3f minBound, maxBound, step;

  VolumeParams()
  {
    minBound = cv::Vec3f(0.1f, -0.1f, -0.3f);
    maxBound = cv::Vec3f(0.5f,  0.2f,  0.0f);
    step = cv::Vec3f::all(0.01f);
  }
};

class ModelCapturer
{
  public:
    struct Observation
    {
      cv::Mat bgrImage;
      cv::Mat initMask, mask;
      cv::Mat bgdProbabilities, fgdProbabilities;
      PoseRT pose;
    };

    ModelCapturer(const PinholeCamera &pinholeCamera);
    void setObservations(const std::vector<Observation> &observations, const std::vector<bool> *isObservationValid = 0);
    void setGroundTruthModel(const std::vector<cv::Point3f> &groundTruthModel);
//    void addObservation(const cv::Mat &objectMask, const PoseRT &pose_cam);
    //TODO: add clear()

    void createModel(std::vector<cv::Point3f> &modelPoints, const std::vector<cv::Point3f> *confidentModelPoints = 0) const;
  private:
    void findRepeatableVoxels(const cv::Mat &insideOfObjectRatios_Vector,
                              cv::Mat &isRepeatable,
                              const VolumeParams &volumeParams) const;
    void supperimposeGroundTruth(const cv::Mat &image3d, cv::Mat &supperimposedImage,
                                 const VolumeParams &volumeParams) const;
    void computeVisibleCounts(cv::Mat &volumePoints, cv::Mat &isRepeatable,
                              const VolumeParams &volumeParams,
                              const std::vector<cv::Point3f> *confidentModelPoints = 0) const;
    void computeGroundTruthEnergy(MRF* mrf, const cv::Mat &volumePoints,
                                            const VolumeParams &volumeParams) const;

    std::vector<Observation> observations;
    PinholeCamera camera;
    std::vector<cv::Point3f> groundTruthModel;
};

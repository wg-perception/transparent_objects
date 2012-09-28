#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/poseRT.hpp"
#include "edges_pose_refiner/pinholeCamera.hpp"

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
      cv::Mat mask;
      PoseRT pose;
    };

    ModelCapturer(const PinholeCamera &pinholeCamera);
    void setObservations(const std::vector<Observation> &observations, const std::vector<bool> *isObservationValid = 0);
    void setGroundTruthModel(const std::vector<cv::Point3f> &groundTruthModel);
//    void addObservation(const cv::Mat &objectMask, const PoseRT &pose_cam);
    //TODO: add clear()

    void createModel(std::vector<cv::Point3f> &modelPoints) const;
  private:
    void computeVisibleCounts(cv::Mat &volumePoints, cv::Mat &visibleCounts,
                              const VolumeParams &volumeParams) const;

    std::vector<Observation> observations;
    PinholeCamera camera;
    std::vector<cv::Point3f> groundTruthModel;
};

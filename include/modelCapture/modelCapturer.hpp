#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/poseRT.hpp"
#include "edges_pose_refiner/pinholeCamera.hpp"

struct VolumeParams
{
  float min_x, max_x;
  float min_y, max_y;
  float min_z, max_z;

  float step_x, step_y, step_z;

  VolumeParams()
  {
    min_x =  0.1f;
    max_x =  0.5f;

    min_y = -0.1f;
    max_y =  0.2f;

    min_z = -0.3f;
    max_z =  0.0f;

    step_x = 0.01f;
    step_y = 0.01f;
    step_z = 0.01f;
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
    void setObservations(const std::vector<Observation> &observations);
//    void addObservation(const cv::Mat &objectMask, const PoseRT &pose_cam);
    //TODO: add clear()

    void createModel(std::vector<cv::Point3f> &modelPoints) const;
  private:
    void computeVisibleCounts(cv::Mat &volumePoints, cv::Mat &visibleCounts,
                              const VolumeParams &volumeParams) const;

    std::vector<Observation> observations;
    PinholeCamera camera;
};

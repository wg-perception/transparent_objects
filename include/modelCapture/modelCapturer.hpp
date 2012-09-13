#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/poseRT.hpp"
#include "edges_pose_refiner/pinholeCamera.hpp"

class ModelCapturer
{
  public:
    ModelCapturer(const PinholeCamera &pinholeCamera);
    void addObservation(const cv::Mat &objectMask, const PoseRT &pose_cam);
    //TODO: add clear()

    void createModel() const;
  private:
    struct Observation
    {
      cv::Mat mask;
      PoseRT pose;
    };

    std::vector<Observation> observations;
    PinholeCamera camera;
};

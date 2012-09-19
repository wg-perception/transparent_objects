#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/poseRT.hpp"
#include "edges_pose_refiner/pinholeCamera.hpp"

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

    std::vector<Observation> observations;
    PinholeCamera camera;
};

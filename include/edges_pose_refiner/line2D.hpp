#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "edges_pose_refiner/edgeModel.hpp"

class Line2D : public cv::linemod::Detector
{
public:
  Line2D();
  Line2D(const std::vector<cv::Ptr<cv::linemod::Modality> >& modalities, const std::vector<int>& T_pyramid);

  int addTemplate(const std::vector<cv::Mat>& sources, const std::string& class_id,
                  const cv::Mat& object_mask, cv::Rect* bounding_box, const PoseRT *pose);

  //TODO: use const-method
  PoseRT getTrainPose(const cv::linemod::Match &match);
  PoseRT getTestPose(const PinholeCamera &camera, const cv::linemod::Match &match);
private:
  std::map<std::string, std::vector<PoseRT> > templatePoses;
  std::map<std::string, std::vector<cv::Point> > origins;

  using cv::linemod::Detector::addTemplate;
};

cv::Ptr<Line2D> getDefaultLine2D();
cv::Ptr<Line2D> trainLine2D(const std::string &baseFolder, const std::vector<std::string> &objectNames, std::vector<int> *testIndicesPtr = 0);
cv::Ptr<Line2D> trainLine2D(const PinholeCamera &camera, const EdgeModel &edgeModel, const std::string &objectName, const std::vector<PoseRT> &trainPoses);

#include "modelCapture/modelCapturer.hpp"

#include "edges_pose_refiner/utils.hpp"

using namespace cv;

ModelCapturer::ModelCapturer(const PinholeCamera &pinholeCamera)
{
  camera = pinholeCamera;
}

void ModelCapturer::addObservation(const cv::Mat &objectMask, const PoseRT &pose_cam)
{
  Observation newObservation;
  newObservation.mask = objectMask;
  newObservation.pose = pose_cam;

  observations.push_back(newObservation);
}

void initializeVolume(std::vector<cv::Point3f> &volumePoints)
{
  //TODO: move up
  const float min_x = -0.5f;
  const float max_x =  0.5f;
  const float min_y = -0.5f;
  const float max_y =  0.5f;
  const float min_z = -0.5f;
  const float max_z =  0.5f;

//  const float x_step = 0.01f;
//  const float y_step = 0.01f;
//  const float z_step = 0.01f;
  const float x_step = 0.0025f;
  const float y_step = 0.0025f;
  const float z_step = 0.0025f;

  volumePoints.clear();
  for (float x = min_x; x < max_x; x += x_step)
  {
    for (float y = min_y; y < max_y; y += y_step)
    {
      for (float z = min_z; z < max_z; z += z_step)
      {
        Point3f pt(x, y, z);
        volumePoints.push_back(pt);
      }
    }
  }
}

void ModelCapturer::createModel() const
{
  cout << "creating model... " << std::flush;
  //TODO: move up
  const float modelPointVisibility = 0.9f;

  vector<Point3f> volumePoints;
  initializeVolume(volumePoints);

  vector<int> visibleCounts(volumePoints.size(), 0);
  for (size_t observationIndex = 0; observationIndex < observations.size(); ++observationIndex)
  {
    cout << "observation: " << observationIndex << endl;

    vector<Point2f> projectedVolume;
    camera.projectPoints(volumePoints, observations[observationIndex].pose, projectedVolume);



//    showEdgels(glassMask)



    const Mat &mask = observations[observationIndex].mask;
    CV_Assert(mask.type() == CV_8UC1);
    for (size_t i = 0; i < projectedVolume.size(); ++i)
    {
      Point pt = projectedVolume[i];
      if (isPointInside(mask, pt) && mask.at<uchar>(pt))
      {
        visibleCounts[i] += 1;
      }
    }
  }


  int modelPointVisibleCount = observations.size() * modelPointVisibility;

  vector<Point3f> modelPoints;
  for (size_t i = 0; i < visibleCounts.size(); ++i)
  {
    if (visibleCounts[i] >= modelPointVisibleCount)
    {
      modelPoints.push_back(volumePoints[i]);
    }
  }
  cout << "done." << endl;

  publishPoints(modelPoints);
}

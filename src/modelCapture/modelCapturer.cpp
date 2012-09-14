#include "modelCapture/modelCapturer.hpp"

#include "edges_pose_refiner/utils.hpp"

using namespace cv;

ModelCapturer::ModelCapturer(const PinholeCamera &pinholeCamera)
{
  camera = pinholeCamera;
}

void ModelCapturer::setObservations(const std::vector<ModelCapturer::Observation> &_observations)
{
  observations = _observations;
}

/*
void ModelCapturer::addObservation(const cv::Mat &objectMask, const PoseRT &pose_cam)
{
  Observation newObservation;
  newObservation.mask = objectMask;
  newObservation.pose = pose_cam;

  observations.push_back(newObservation);
}
*/

//void initializeVolume(std::vector<cv::Point3f> &volumePoints)
void initializeVolume(Mat &volumePoints)
{
  //TODO: move up
//  const float min_x = -0.5f;
//  const float max_x =  0.5f;
//  const float min_y = -0.5f;
//  const float max_y =  0.5f;
//  const float min_z = -0.5f;
//  const float max_z =  0.5f;

  const float min_x =  0.1f;
  const float max_x =  0.5f;
  const float min_y = -0.1f;
  const float max_y =  0.2f;
  const float min_z = -0.3f;
  const float max_z =  0.0f;

//  const float x_step = 0.01f;
//  const float y_step = 0.01f;
//  const float z_step = 0.01f;
//  const float x_step = 0.0025f;
//  const float y_step = 0.0025f;
//  const float z_step = 0.0025f;

  const float step_x = 0.001f;
  const float step_y = 0.001f;
  const float step_z = 0.001f;
//  const float step_x = 0.01f;
//  const float step_y = 0.01f;
//  const float step_z = 0.01f;

  int dim_x = (max_x - min_x) / step_x;
  int dim_y = (max_y - min_y) / step_y;
  int dim_z = (max_z - min_z) / step_z;

  int dims[] = {dim_x, dim_y, dim_z};
  const int ndims= 3;
  volumePoints.create(ndims, dims, CV_32FC3);

  for (int z = 0; z < dim_z; ++z)
  {
    for (int x = 0; x < dim_x; ++x)
    {
      for (int y = 0; y < dim_y; ++y)
      {
        Vec3f pt(min_x + x * step_x, min_y + y * step_y, min_z + z * step_z);
        volumePoints.at<Vec3f>(z, x, y) = pt;
      }
    }
  }

/*
  volumePoints.clear();
  for (float z = min_z; z < max_z; z += z_step)
  {
    for (float x = min_x; x < max_x; x += x_step)
    {
      for (float y = min_y; y < max_y; y += y_step)
      {
        Point3f pt(x, y, z);
        volumePoints.push_back(pt);
      }
    }
  }
*/
}

void ModelCapturer::createModel() const
{
  cout << "creating model... " << std::flush;
  //TODO: move up
  const float modelPointVisibility = 0.9f;

//  vector<Point3f> volumePoints;
  Mat volumePoints;
  initializeVolume(volumePoints);

  Mat visibleCounts(volumePoints.dims, volumePoints.size.p, CV_32SC1, Scalar(0));
  Mat visibleCounts_Vector(1, visibleCounts.total(), CV_32SC1, visibleCounts.data);
  Mat volumePoints_Vector(1, volumePoints.total(), CV_32FC3, volumePoints.data);
  for (size_t observationIndex = 0; observationIndex < observations.size(); ++observationIndex)
  {
    cout << "observation: " << observationIndex << endl;
 //   showEdgels(observations[observationIndex].bgrImage, volumePoints, observations[observationIndex].pose,camera);
//    waitKey();


    vector<Point2f> projectedVolume;
//    camera.projectPoints(volumePoints, observations[observationIndex].pose, projectedVolume);
//    camera.projectPoints(tmpVolume, observations[observationIndex].pose, projectedVolume);
    camera.projectPoints(volumePoints_Vector, observations[observationIndex].pose, projectedVolume);






    const Mat &mask = observations[observationIndex].mask;
    CV_Assert(mask.type() == CV_8UC1);
    for (size_t i = 0; i < projectedVolume.size(); ++i)
    {
      Point pt = projectedVolume[i];
      if (isPointInside(mask, pt) && mask.at<uchar>(pt))
      {
        visibleCounts_Vector.at<int>(i) += 1;
      }
    }
  }



  int modelPointVisibleCount = observations.size() * modelPointVisibility;
  Mat isRepeatable = visibleCounts >= modelPointVisibleCount;

  CV_Assert(isRepeatable.type() == CV_8UC1);
  int previousLevelPointsCount = 0;
  int zLevelIndex;
  for (zLevelIndex = 0; zLevelIndex < isRepeatable.size.p[0]; ++zLevelIndex)
  {
    Mat zSlice(isRepeatable.size.p[1], isRepeatable.size.p[2], CV_8UC1, isRepeatable.ptr(zLevelIndex, 0, 0));
    int currentLevelPointsCount = countNonZero(zSlice);
    if (currentLevelPointsCount > 0 && currentLevelPointsCount <= previousLevelPointsCount)
    {
      break;
    }

    previousLevelPointsCount = currentLevelPointsCount;
  }
  int modelStartLevelIndex = max(zLevelIndex - 1, 1);

  vector<Point3f> modelPoints;
  for (int i = modelStartLevelIndex; i < isRepeatable.size.p[0] - 1; ++i)
  {
    for (int j = 1; j < isRepeatable.size.p[1] - 1; ++j)
    {
      for (int k = 1; k < isRepeatable.size.p[2] - 1; ++k)
      {
        if (!isRepeatable.at<uchar>(i, j, k))
        {
          continue;
        }

        if (isRepeatable.at<uchar>(i - 1, j, k) && isRepeatable.at<uchar>(i + 1, j ,k) &&
            isRepeatable.at<uchar>(i, j - 1, k) && isRepeatable.at<uchar>(i, j + 1 ,k) &&
            isRepeatable.at<uchar>(i, j, k - 1) && isRepeatable.at<uchar>(i, j ,k + 1))
        {
          continue;
        }

        modelPoints.push_back(volumePoints.at<Vec3f>(i, j, k));
      }
    }
  }




/*
  Mat plane;
  const Mat *arrays = &isRepeatable;
  NAryMatIterator it(&arrays, &plane, 1);
  cout << "sz: " << isRepeatable.size.p[0] << " " << isRepeatable.size.p[1] << " " << isRepeatable.size.p[2] << endl;
  cout << it.nplanes << endl;
  for(int p = 0; p < it.nplanes; p++, ++it)
  {
    cout << it.planes[0].dims << endl;
    cout << it.planes[0].rows << endl;
    cout << it.planes[0].cols << endl;
  }
*/

 /*
  CV_Assert(!modelPoints.empty());
  float currentModelZ = modelPoints[0].z;
  int previousLevelPointsCount = 0, currentLevelPointsCount = 0;
  int currentLevelStart = 0;
  int previousLevelStart = -1;
  size_t modelPointIndex;

  //TODO: move up
  const float z_step = 0.001f;
  for (modelPointIndex = 0; modelPointIndex < modelPoints.size(); ++modelPointIndex)
  {
    if (fabs(currentModelZ - modelPoints[modelPointIndex].z) < z_step / 2.0)
    {
      ++currentLevelPointsCount;
    }
    else
    {
      cout << currentLevelPointsCount << endl;
      if (previousLevelPointsCount >= currentLevelPointsCount)
      {
        break;
      }
      previousLevelPointsCount = currentLevelPointsCount;
      currentLevelPointsCount = 1;
      currentModelZ = modelPoints[modelPointIndex].z;

      previousLevelStart = currentLevelStart;
      currentLevelStart = modelPointIndex;
    }
  }
  int trueModelStart = previousLevelPointsCount;
  vector<Point3f> trueModel;
  std::copy(modelPoints.begin() + trueModelStart, modelPoints.end(), std::back_inserter(trueModel));

  cout << "done." << endl;
*/
  publishPoints(modelPoints);
//  publishPoints(trueModel);
}

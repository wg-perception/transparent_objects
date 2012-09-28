#include "modelCapture/modelCapturer.hpp"

#include "edges_pose_refiner/utils.hpp"

using namespace cv;

ModelCapturer::ModelCapturer(const PinholeCamera &pinholeCamera)
{
  camera = pinholeCamera;
}

void ModelCapturer::setObservations(const std::vector<ModelCapturer::Observation> &_observations, const std::vector<bool> *isObservationValid)
{
  if (isObservationValid == 0)
  {
    observations = _observations;
  }
  else
  {
    observations.clear();
    CV_Assert(_observations.size() == isObservationValid->size());
    for (size_t i = 0; i < isObservationValid->size(); ++i)
    {
      if ((*isObservationValid)[i])
      {
        observations.push_back(_observations[i]);
      }
    }
  }
}


void ModelCapturer::setGroundTruthModel(const std::vector<cv::Point3f> &_groundTruthModel)
{
  groundTruthModel = _groundTruthModel;
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


void initializeVolume(Mat &volumePoints, const VolumeParams &params = VolumeParams())
{
  Vec3f dimensions = (params.maxBound - params.minBound);
  Mat(dimensions, false) /= Mat(params.step);
  cout << "Volume dims: " << Mat(dimensions) << endl;

  int dims[] = {dimensions[2], dimensions[0], dimensions[1]};
  const int ndims= 3;
  volumePoints.create(ndims, dims, CV_32FC3);

  for (int z = 0; z < dims[0]; ++z)
  {
    for (int x = 0; x < dims[1]; ++x)
    {
      for (int y = 0; y < dims[2]; ++y)
      {
        Vec3f index(x, y, z);
        Vec3f pt = params.minBound + index.mul(params.step);
        volumePoints.at<Vec3f>(z, x, y) = pt;
      }
    }
  }
}

void ModelCapturer::computeVisibleCounts(cv::Mat &volumePoints, cv::Mat &visibleCounts,
                                         const VolumeParams &volumeParams) const
{
  initializeVolume(volumePoints, volumeParams);

  visibleCounts.create(volumePoints.dims, volumePoints.size.p, CV_32SC1);
  visibleCounts = Scalar(0);

  Mat visibleCounts_Vector(1, visibleCounts.total(), CV_32SC1, visibleCounts.data);
  Mat volumePoints_Vector(1, volumePoints.total(), CV_32FC3, volumePoints.data);
  for (size_t observationIndex = 0; observationIndex < observations.size(); ++observationIndex)
  {
    cout << "observation: " << observationIndex << endl;
 //   showEdgels(observations[observationIndex].bgrImage, volumePoints, observations[observationIndex].pose,camera);
//    waitKey();

    vector<Point2f> projectedVolume;
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
}

void ModelCapturer::createModel(std::vector<cv::Point3f> &modelPoints) const
{
  cout << "creating model... " << std::flush;

  VolumeParams volumeParams;
  Mat volumePoints, visibleCounts;
  computeVisibleCounts(volumePoints, visibleCounts, volumeParams);

  //TODO: move up
  const float modelPointVisibility = 0.9f;
  int modelPointVisibleCount = observations.size() * modelPointVisibility;
  Mat isRepeatable = visibleCounts >= modelPointVisibleCount;

  Vec3i minIndices = Vec3i::all(std::numeric_limits<int>::max());
  Vec3i maxIndices = Vec3i::all(-1);
  Mat minIndices_Mat(minIndices, false);
  Mat maxIndices_Mat(maxIndices, false);

  for (int i = 0; i < isRepeatable.size.p[0]; ++i)
  {
    for (int j = 0; j < isRepeatable.size.p[1]; ++j)
    {
      for (int k = 0; k < isRepeatable.size.p[2]; ++k)
      {
        if (isRepeatable.at<uchar>(i, j, k))
        {
          Mat indices(Vec3i(j, k, i));
          minIndices_Mat = min(minIndices_Mat, indices);
          maxIndices_Mat = max(maxIndices_Mat, indices);
        }
      }
    }
  }

  //TODO: move up
  const int uncertaintyWidth = 3;

  Vec3f minIndices_float = minIndices;
  Vec3f maxIndices_float = maxIndices;
  volumeParams.maxBound = volumeParams.minBound + (maxIndices_float + Vec3f::all(uncertaintyWidth)).mul(volumeParams.step);
  volumeParams.minBound = volumeParams.minBound + (minIndices_float - Vec3f::all(uncertaintyWidth)).mul(volumeParams.step);
  volumeParams.step = Vec3f::all(0.001f);
  //max_z should be equal to zero
  volumeParams.maxBound[2] = 0.0f;

  computeVisibleCounts(volumePoints, visibleCounts, volumeParams);
  isRepeatable = visibleCounts >= modelPointVisibleCount;


  CV_Assert(isRepeatable.type() == CV_8UC1);
  vector<float> levelCounts(isRepeatable.size.p[0]);
  int firstLevelIndex = -1, lastLevelIndex = -1;
  for (int zLevelIndex = 0; zLevelIndex < isRepeatable.size.p[0]; ++zLevelIndex)
  {
    Mat zSlice(isRepeatable.size.p[1], isRepeatable.size.p[2], CV_8UC1, isRepeatable.ptr(zLevelIndex, 0, 0));
    levelCounts[zLevelIndex] = countNonZero(zSlice);

    if (levelCounts[zLevelIndex] > 0)
    {
      lastLevelIndex = zLevelIndex;

      if (firstLevelIndex < 0)
      {
        firstLevelIndex = zLevelIndex;
        cout << "first level: " << zLevelIndex << endl;
      }
    }
  }

  Mat all_A(levelCounts.size(), 2, CV_32FC1, Scalar(1.0));
  for (int i = 0; i < all_A.rows; ++i)
  {
    all_A.at<float>(i, 1) = i;
  }
  Mat all_b(levelCounts);
  //TODO: move up
  const int windowWidth = 20;
//  const int windowWidth = 5;
  const int localMinWindow = windowWidth / 2;

  vector<float> errors(levelCounts.size(), std::numeric_limits<float>::max());
  const int firstLevelIndexToCheck = firstLevelIndex + max(all_A.cols, windowWidth);
  for (int i = firstLevelIndexToCheck; i <= lastLevelIndex; ++i)
  {
    Mat A = all_A.rowRange(i - windowWidth, i);
    Mat b = all_b.rowRange(i - windowWidth, i);
    Mat x;
    solve(A, b, x, DECOMP_SVD);
    errors[i] = norm(A * x - b);
    cout << i << " " << errors[i] << endl;
  }

  int modelStartLevelIndex = -1;
  for (int i = firstLevelIndexToCheck; i <= lastLevelIndex; ++i)
  {
    bool isLocalMin = true;
    for (int j = max(firstLevelIndexToCheck, i - localMinWindow); j <= min(i + localMinWindow, lastLevelIndex); ++j)
    {
      if (errors[j] < errors[i])
      {
        isLocalMin = false;
        break;
      }
    }

    if (isLocalMin)
    {
      modelStartLevelIndex = i;
      break;
    }
  }
  CV_Assert(modelStartLevelIndex >= 0);
  cout << "start: " << modelStartLevelIndex << endl;

  modelPoints.clear();
  for (int i = modelStartLevelIndex; i < isRepeatable.size.p[0]; ++i)
  {
    for (int j = 1; j < isRepeatable.size.p[1] - 1; ++j)
    {
      for (int k = 1; k < isRepeatable.size.p[2] - 1; ++k)
      {
        if (!isRepeatable.at<uchar>(i, j, k))
        {
          continue;
        }

        if (i != isRepeatable.size.p[0] - 1 &&   // add bottom
            isRepeatable.at<uchar>(i - 1, j, k) && isRepeatable.at<uchar>(i + 1, j ,k) &&
            isRepeatable.at<uchar>(i, j - 1, k) && isRepeatable.at<uchar>(i, j + 1 ,k) &&
            isRepeatable.at<uchar>(i, j, k - 1) && isRepeatable.at<uchar>(i, j ,k + 1))
        {
          continue;
        }

        modelPoints.push_back(volumePoints.at<Vec3f>(i, j, k));
      }
    }
  }
}

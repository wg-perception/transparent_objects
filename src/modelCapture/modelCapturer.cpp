#include "modelCapture/modelCapturer.hpp"

#include "edges_pose_refiner/utils.hpp"
#include "../mrf/GCoptimization.h"

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

//TODO: remove the global variable
Mat global_insideOfObjectRatios_Vector;

MRF::CostVal dCost(int pix, int label)
{
  CV_Assert(global_insideOfObjectRatios_Vector.type() == CV_32FC1);
  //TODO: move up
  const float powIndex = 4.67;
  float ratio = global_insideOfObjectRatios_Vector.at<float>(pix);
  float cost;
  if (label == 0)
  {
    cost = pow(ratio, powIndex);
  }
  else
  {
    cost = pow(1.0 - ratio, 1.0 / powIndex);
  }

  return cost;
}

MRF::CostVal fnCost(int pix_1, int pix_2, int label_1, int label_2)
{
  CV_Assert(global_insideOfObjectRatios_Vector.type() == CV_32FC1);
  if (label_1 == label_2)
  {
    return 0;
  }

  float ratio_1 = global_insideOfObjectRatios_Vector.at<float>(pix_1);
  float ratio_2 = global_insideOfObjectRatios_Vector.at<float>(pix_2);

  //TODO: move up
  const float smoothnessWeight = 0.5f;
  float cost = smoothnessWeight * exp(-fabs(ratio_1 - ratio_2));
  return cost;
}

void ModelCapturer::supperimposeGroundTruth(const cv::Mat &image3d, cv::Mat &supperimposedImage,
                                            const VolumeParams &volumeParams) const
{
  //TODO: move up
  const Vec3b groundTruthColor(0, 255, 0);
  CV_Assert(image3d.channels() == 1);
  Mat image3d_uchar = image3d;
  if (image3d.type() == CV_32FC1)
  {
    image3d.convertTo(image3d_uchar, CV_8UC1, 255);
  }
  CV_Assert(image3d_uchar.type() == CV_8UC1);

  cvtColor3d(image3d_uchar, supperimposedImage, CV_GRAY2BGR);

  Vec3f inverseStep;
  for (int i = 0; i < volumeParams.step.channels; ++i)
  {
    inverseStep[i] = 1.0 / volumeParams.step[i];
  }

  for (size_t i = 0; i < groundTruthModel.size(); ++i)
  {
    Vec3f pt(groundTruthModel[i]);
    Vec3i raw_indices = (pt - volumeParams.minBound).mul(inverseStep);
    Vec3i indices(raw_indices[2], raw_indices[0], raw_indices[1]);

    bool isInside = true;
    for (int j = 0; j < indices.channels; ++j)
    {
      isInside = isInside && indices[j] >= 0 && indices[j] < supperimposedImage.size.p[j];
    }

    if (isInside)
    {
      supperimposedImage.at<Vec3b>(indices) = groundTruthColor;
    }
  }
}

void ModelCapturer::findRepeatableVoxels(const cv::Mat &insideOfObjectRatios_Vector,
                                         cv::Mat &isRepeatable,
                                         const VolumeParams &volumeParams) const
{
  Mat isRepeatable_Vector(1, isRepeatable.total(), CV_8UC1, isRepeatable.data);
/*
  //TODO: move up
  const float minModelPointRepeatability = 0.9f;

  for (size_t i = 0; i < isRepeatable.total(); ++i)
  {
    if (insideOfImageCounts_Vector.at<int>(i) == 0)
    {
      continue;
    }

    if (visibleCounts_Vector.at<int>(i) == 0 ||
        static_cast<float>(insideOfObjectCounts_Vector.at<int>(i)) / visibleCounts_Vector.at<int>(i) > minModelPointRepeatability)
    {
      isRepeatable_Vector.at<uchar>(i) = 255;
    }
  }
*/

  global_insideOfObjectRatios_Vector = insideOfObjectRatios_Vector;
  float time;
  DataCost *data         = new DataCost(dCost);
  SmoothnessCost *smooth = new SmoothnessCost(fnCost);
  EnergyFunction *energy = new EnergyFunction(data,smooth);
  MRF* mrf = new Expansion(isRepeatable.total(), 2, energy);

  const int neighborWeight = 1;
  for (int i = 0; i < isRepeatable.size.p[0] - 1; ++i)
  {
    for (int j = 0; j < isRepeatable.size.p[1] - 1; ++j)
    {
      for (int k = 0; k < isRepeatable.size.p[2] - 1; ++k)
      {
        int current_index = k + j * isRepeatable.size.p[2] + i * isRepeatable.size.p[2] * isRepeatable.size.p[1];
        mrf->setNeighbors(current_index, current_index + 1, neighborWeight);
        mrf->setNeighbors(current_index, current_index + isRepeatable.size.p[2], neighborWeight);
        mrf->setNeighbors(current_index, current_index + isRepeatable.size.p[2] * isRepeatable.size.p[1], neighborWeight);
      }
    }
  }

  mrf->initialize();
  mrf->clearAnswer();
  //TODO: move up
  const int iterationsCount = 50;
  mrf->optimize(50, time);
  MRF::EnergyVal E_smooth = mrf->smoothnessEnergy();
  MRF::EnergyVal E_data   = mrf->dataEnergy();
  cout << "Total Energy = " << E_smooth + E_data << endl;
  cout << "Smoothness Energy = " << E_smooth << endl;
  cout << "Data Energy = " << E_data << endl;

  for (size_t i = 0; i < isRepeatable.total(); ++i)
  {
    isRepeatable_Vector.at<uchar>(i) = 255 * mrf->getLabel(i);
  }

  Mat vizImage;
  supperimposeGroundTruth(isRepeatable, vizImage, volumeParams);
  imshow3d("isRepeatable", vizImage);
  waitKey(0);

  delete mrf;
}

void ModelCapturer::computeVisibleCounts(cv::Mat &volumePoints, cv::Mat &isRepeatable,
                                         const VolumeParams &volumeParams,
                                         const std::vector<cv::Point3f> *confidentModelPoints) const
{
  initializeVolume(volumePoints, volumeParams);

  isRepeatable.create(volumePoints.dims, volumePoints.size.p, CV_8UC1);
  isRepeatable = Scalar(0);

  Mat isInsideOfObject(isRepeatable.total(), observations.size(), CV_8UC1, Scalar(0));
  Mat visibleCounts_Vector(1, isRepeatable.total(), CV_32SC1, Scalar(0));
  Mat insideOfImageCounts_Vector(1, isRepeatable.total(), CV_32SC1, Scalar(0));
  Mat insideOfObjectCounts_Vector(1, isRepeatable.total(), CV_32SC1, Scalar(0));

  Mat volumePoints_Vector(1, volumePoints.total(), CV_32FC3, volumePoints.data);
  for (size_t observationIndex = 0; observationIndex < observations.size(); ++observationIndex)
  {
    cout << "observation: " << observationIndex << endl;

    vector<Point2f> projectedVolume;
    vector<Point3f> rotatedVolume;
    project3dPoints(volumePoints_Vector, observations[observationIndex].pose, rotatedVolume);
    camera.projectPoints(rotatedVolume, PoseRT(), projectedVolume);



/*
    vector<Point2f> projectedConfidentModelPoints;
    Mat min_z(camera.imageSize, CV_32FC1, Scalar(std::numeric_limits<float>::max()));
    if (confidentModelPoints != 0)
    {
      vector<Point3f> rotatedConfidentModelPoints;
      project3dPoints(*confidentModelPoints, observations[observationIndex].pose, rotatedConfidentModelPoints);
      camera.projectPoints(rotatedConfidentModelPoints, PoseRT(), projectedConfidentModelPoints);
      for (size_t i = 0; i < projectedConfidentModelPoints.size(); ++i)
      {
        Point pt = projectedConfidentModelPoints[i];
        CV_Assert(isPointInside(min_z, pt));

        if (min_z.at<float>(pt) > (rotatedConfidentModelPoints)[i].z)
        {
          min_z.at<float>(pt) = (rotatedConfidentModelPoints)[i].z;
        }
      }
    }
*/


    const Mat &mask = observations[observationIndex].mask;
    CV_Assert(mask.type() == CV_8UC1);
    for (size_t i = 0; i < projectedVolume.size(); ++i)
    {
      Point pt = projectedVolume[i];
      if (isPointInside(mask, pt))
      {
        insideOfImageCounts_Vector.at<int>(i) += 1;
        // we use <= so confidentModelPoints can be updated too
        //TODO: use soft threshold
//        if (rotatedVolume[i].z <= min_z.at<float>(pt))
        {
          visibleCounts_Vector.at<int>(i) += 1;
          if (mask.at<uchar>(pt))
          {
            insideOfObjectCounts_Vector.at<int>(i) += 1;
//            isInsideOfObject.at<uchar>(i, observationIndex) = 255;
            isInsideOfObject.at<uchar>(i, observationIndex) = 1;
          }
        }
      }



#if 0
//      Vec3f interestingPoint(0.211, 0.049, -0.107);
      Vec3f interestingPoint(0.212, 0.067, -0.117);
      Vec3f interestingPoint(0.212, 0.067, -0.117);

      if (confidentModelPoints != 0 && isRepeatable.total() >= 1e6 && norm(volumePoints_Vector.at<Vec3f>(i) - interestingPoint) < 1e-4)
      {
        CV_Assert(isPointInside(mask, pt));
        CV_Assert(observations[observationIndex].bgdProbabilities.type() == CV_64FC1);
        CV_Assert(observations[observationIndex].fgdProbabilities.type() == CV_64FC1);
        double bgd = observations[observationIndex].bgdProbabilities.at<double>(pt);
        double fgd = observations[observationIndex].fgdProbabilities.at<double>(pt);
        cout << bgd << " " << fgd << " " << fgd / bgd << endl;

//        Mat viz = showEdgels(observations[observationIndex].bgrImage, *confidentModelPoints, observations[observationIndex].pose, camera);
        Mat viz = observations[observationIndex].bgrImage;
        Mat seg = drawSegmentation(viz, observations[observationIndex].mask);
        circle(seg, pt, 1, Scalar(255, 0, 0), -1);
/*
        cout << isPointInside(mask, pt) << " " << (rotatedVolume[i].z <= min_z.at<float>(pt)) << " " << int(mask.at<uchar>(pt)) << endl;

        if (observationIndex + 1 == observations.size())
        {
          cout << "Result:" << endl;
          cout << insideOfImageCounts_Vector.at<int>(i) << " " << visibleCounts_Vector.at<int>(i) << " " << insideOfObjectCounts_Vector.at<int>(i) << endl;
        }
*/

        imshow("seg", seg);
        waitKey();
      }
#endif

    }
  }

#if 0
  //TODO: move up
//  const float minModelPointRepeatability = 0.9f;
  const int width = 5;
  for (int i = 0; i < isInsideOfObject.rows; ++i)
  {
    int insideOfObjectCount = 0;

    int curSum = 0;
    for (int j = 0; j < width; ++j)
    {
      curSum += isInsideOfObject.at<uchar>(i, j);
    }

    for (int j = width / 2; j < isInsideOfObject.cols - width / 2; ++j)
    {
      if (i == 817788)
      {
        cout << int(isInsideOfObject.at<uchar>(i, j)) << " " << int(curSum > width / 2) << " " << curSum << endl;
      }
      if (curSum > width / 2)
      {
        insideOfObjectCount += 1;
      }
      curSum += isInsideOfObject.at<uchar>(i, j + width/2 + 1) - static_cast<int>(isInsideOfObject.at<uchar>(i, j - width/2));
    }

    insideOfObjectCounts_Vector.at<int>(i) = insideOfObjectCount;
/*
    if (insideOfObjectCount >= minModelPointRepeatability * (isInsideOfObject.cols - width - 1))
    {
      isRepeatable_Vector.at<uchar>(i) = 255;
    }
*/
  }
#endif

  Mat insideOfObjectRatios_Vector(insideOfObjectCounts_Vector.size(), CV_32FC1, Scalar(0));
  for (size_t i = 0; i < insideOfObjectCounts_Vector.total(); ++i)
  {
    if (insideOfImageCounts_Vector.at<int>(i) == 0)
    {
      continue;
    }

    float ratio = (visibleCounts_Vector.at<int>(i) == 0) ? 1.0f : static_cast<float>(insideOfObjectCounts_Vector.at<int>(i)) / visibleCounts_Vector.at<int>(i);
    insideOfObjectRatios_Vector.at<float>(i) = ratio;
  }

  Mat insideOfObjectRatios(isRepeatable.dims, isRepeatable.size.p, CV_32FC1, insideOfObjectRatios_Vector.data);
  Mat vizImage;
  supperimposeGroundTruth(insideOfObjectRatios, vizImage, volumeParams);
  imshow3d("ratios", vizImage);
  waitKey(0);

  findRepeatableVoxels(insideOfObjectRatios_Vector, isRepeatable, volumeParams);
}

void ModelCapturer::createModel(std::vector<cv::Point3f> &modelPoints, const std::vector<cv::Point3f> *confidentModelPoints) const
{
  cout << "creating model... " << std::flush;

  VolumeParams volumeParams;
  Mat volumePoints, isRepeatable;
  computeVisibleCounts(volumePoints, isRepeatable, volumeParams, confidentModelPoints);


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

  computeVisibleCounts(volumePoints, isRepeatable, volumeParams, confidentModelPoints);
  cout << "Volume:" << endl;
  cout << Mat(volumeParams.minBound) << endl;
  cout << Mat(volumeParams.maxBound) << endl;


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
  CV_Assert(isRepeatable.type() == CV_8UC1);
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

        Point3f pt = volumePoints.at<Vec3f>(i, j, k);
        modelPoints.push_back(pt);
      }
    }
  }

  cout << "Model is created." << endl;
  cout << "Number of points: " << modelPoints.size() << endl;
}

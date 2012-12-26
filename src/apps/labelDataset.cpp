#include "edges_pose_refiner/edgeModel.hpp"
#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/glassSegmentator.hpp"
#include "edges_pose_refiner/detector.hpp"
#include "edges_pose_refiner/poseEstimator.hpp"
#include "edges_pose_refiner/tableSegmentation.hpp"

using namespace cv;
using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
  if (argc != 5)
  {
    cout << argv[0] << " <modelsPath> <baseFoldler> <testObjectName> <occlusionName>" << endl;
    return -1;
  }

  const string trainedModelsPath = argv[1];
  const string baseFolder = argv[2];
  const string testObjectName = argv[3];
  const string occlusionName = argv[4];
  const string registrationMaskFilename = baseFolder + "/registrationMask.png";
  const string cameraFilename = baseFolder + "/center.yml";
  const string testFolder = baseFolder + "/" + testObjectName + "/";

  vector<string> occlusionObjectsNames = {occlusionName};

  TODBaseImporter baseImporter(baseFolder, testFolder);

  PinholeCamera camera;
  Mat registrationMask;
  vector<int> testIndices;
  vector<EdgeModel> occlusionEdgeModels;
  baseImporter.importAllData(&trainedModelsPath, &occlusionObjectsNames,
                             &camera, &registrationMask, &occlusionEdgeModels, &testIndices);

  transpod::PoseEstimator occlusionPoseEstimator(camera);
  occlusionPoseEstimator.setModel(occlusionEdgeModels[0]);

  GlassSegmentatorParams glassSegmentationParams;
  //good_clutter
  glassSegmentationParams.openingIterations = 15;
  glassSegmentationParams.closingIterations = 12;
  glassSegmentationParams.finalClosingIterations = 32;
  glassSegmentationParams.grabCutErosionsIterations = 4;
  GlassSegmentator glassSegmentator(glassSegmentationParams);

  vector<PoseRT> occlusionOffsets;
  cout << "offsets:" << endl;
  for (vector<int>::iterator testIterator = testIndices.begin(); testIterator != testIndices.end(); ++testIterator)
  {
    int testImageIndex = *testIterator;

    Mat bgrImage;
    baseImporter.importBGRImage(testImageIndex, bgrImage);

    Mat depthImage;
    baseImporter.importDepth(testImageIndex, depthImage);

    Mat sceneCloud;

    PoseRT model2test_ground;
    baseImporter.importGroundTruth(testImageIndex,model2test_ground, false);

    vector<PoseRT> poses_cam;
    vector<float> posesQualities;
    vector<string> objectNames;

    if (testIterator == testIndices.begin())
    {
      transpod::DetectorParams detectorParams;
      detectorParams.planeSegmentationMethod = transpod::FIDUCIALS;
      detectorParams.glassSegmentationMethod = transpod::MANUAL;
      transpod::Detector detector(camera, detectorParams);
      detector.addTrainObject(occlusionName, occlusionPoseEstimator);
      detector.detect(bgrImage, depthImage, registrationMask, sceneCloud, poses_cam, posesQualities, objectNames);
      detector.showResults(poses_cam, objectNames, bgrImage);
      waitKey(1000);
    }
    else
    {
      PoseRT initialPose = model2test_ground * occlusionOffsets[0];
      cout << "initial pose[" << testImageIndex << "]: " << initialPose << endl;

      Mat initialVisualization = bgrImage.clone();
      occlusionPoseEstimator.visualize(initialPose, initialVisualization);

      poses_cam.push_back(initialPose);
      //TODO: move up
      posesQualities.push_back(1.0f);
      objectNames.push_back(occlusionName);

      Mat glassMask;
      int numberOfComponents;
      glassSegmentator.segment(bgrImage, depthImage, registrationMask, numberOfComponents, glassMask);
      showSegmentation(bgrImage, glassMask);

      Vec4f tablePlane;
      computeTableOrientationByFiducials(camera, bgrImage, tablePlane);
      occlusionPoseEstimator.refinePosesBySupportPlane(bgrImage, glassMask, tablePlane, poses_cam, posesQualities);

      Mat finalVisualization = bgrImage.clone();
      occlusionPoseEstimator.visualize(poses_cam[0], finalVisualization);

      Mat fullVisualization;
      hconcat(initialVisualization, finalVisualization, fullVisualization);
      imshow("estimated poses", fullVisualization);
      waitKey(500);
    }
    CV_Assert(!poses_cam.empty());

    PoseRT currentOffset = model2test_ground.inv() * poses_cam[0];
    occlusionOffsets.push_back(currentOffset);
    cout << currentOffset << endl;
  }


  //TODO: use MCD
  vector<float> xs, ys;
  for (size_t i = 0; i < occlusionOffsets.size(); ++i)
  {
    Mat tvec = occlusionOffsets[i].getTvec();
    xs.push_back(tvec.at<double>(0));
    ys.push_back(tvec.at<double>(1));
    const float eps = 1e-3;
    CV_Assert(fabs(tvec.at<double>(2)) < eps);
  }

  int middleIndex = xs.size() / 2;
  std::nth_element(xs.begin(), xs.begin() + middleIndex, xs.end());
  float x_offset = xs[middleIndex];
  std::nth_element(ys.begin(), ys.begin() + middleIndex, ys.end());
  float y_offset = ys[middleIndex];

  cout << "final offset: " << x_offset << " " << y_offset << endl;

  PoseRT finalOffset;
  finalOffset.tvec.at<double>(0) = x_offset;
  finalOffset.tvec.at<double>(1) = y_offset;

  string offsetFilename = baseFolder + "/" + testObjectName + "/occlusion_" + occlusionName + ".xml";
  finalOffset.write(offsetFilename);

  for (vector<int>::iterator testIterator = testIndices.begin(); testIterator != testIndices.end(); ++testIterator)
  {
    int testImageIndex = *testIterator;

    Mat bgrImage;
    baseImporter.importBGRImage(testImageIndex, bgrImage);

    PoseRT model2test_ground;
    baseImporter.importGroundTruth(testImageIndex,model2test_ground, false);

    PoseRT initialPose = model2test_ground * finalOffset;

    Mat initialVisualization = bgrImage.clone();
    occlusionPoseEstimator.visualize(initialPose, initialVisualization);
    imshow("sanity check", initialVisualization);
    waitKey(200);
  }
  waitKey();

  return 0;
}

#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/line2D.hpp"
#include "edges_pose_refiner/poseError.hpp"
#include "edges_pose_refiner/glassSegmentator.hpp"
#include "edges_pose_refiner/nonMaximumSuppression.hpp"

#include <iostream>
#include <iterator>
#include <set>
#include <fstream>

//#define VERBOSE
//#define VISUALIZE_POSES


using namespace cv;
using std::cout;
using std::endl;

//TODO: remove code duplication
void suppress3DPoses(const EdgeModel &edgeModel, const std::vector<float> &confidences, const std::vector<PoseRT> &poses_cam,
                                    float neighborMaxRotation, float neighborMaxTranslation,
                                    std::vector<bool> &isFilteredOut)
{
  CV_Assert(confidences.size() == poses_cam.size());

  if (isFilteredOut.empty())
  {
    isFilteredOut.resize(confidences.size(), false);
  }
  else
  {
    CV_Assert(isFilteredOut.size() == confidences.size());
  }

  vector<vector<int> > neighbors(poses_cam.size());
  for (size_t i = 0; i < poses_cam.size(); ++i)
  {
    if (isFilteredOut[i])
    {
      continue;
    }

    for (size_t j = i + 1; j < poses_cam.size(); ++j)
    {
      if (isFilteredOut[j])
      {
        continue;
      }

      double rotationDistance, translationDistance;
      //TODO: use rotation symmetry
      //TODO: check symmetry of the distance
      PoseRT::computeDistance(poses_cam[i], poses_cam[j], rotationDistance, translationDistance, edgeModel.Rt_obj2cam);

      if (rotationDistance < neighborMaxRotation && translationDistance < neighborMaxTranslation)
      {
        neighbors[i].push_back(j);
        neighbors[j].push_back(i);
      }
    }
  }

  filterOutNonMaxima(confidences, neighbors, isFilteredOut);
}


void evaluatePoseWithRotation(const EdgeModel &originalEdgeModel, const PoseRT &est_cam, const PoseRT &ground_cam, PoseError &poseError)
{
  EdgeModel groundModel, estimatedModel;
  originalEdgeModel.rotate_cam(ground_cam, groundModel);
  originalEdgeModel.rotate_cam(est_cam, estimatedModel);

  const double eps = 1e-4;
  CV_Assert(groundModel.hasRotationSymmetry);
  CV_Assert(estimatedModel.hasRotationSymmetry);

  Point3d tvecPoint = groundModel.getObjectCenter() - estimatedModel.getObjectCenter();
  double tvecDiff = norm(tvecPoint);

  double cosAngle = groundModel.upStraightDirection.ddot(estimatedModel.upStraightDirection);
  cosAngle = std::min(cosAngle, 1.0);
  cosAngle = std::max(cosAngle, -1.0);
  double hartleyDiff = acos(cosAngle);

  PoseRT diff_cam = est_cam * ground_cam.inv();
  Mat Rt_diff_obj = groundModel.Rt_obj2cam.inv(DECOMP_SVD) * diff_cam.getProjectiveMatrix() * groundModel.Rt_obj2cam;
  Mat rvec_diff_obj, tvec_diff_obj;
  getRvecTvec(Rt_diff_obj, rvec_diff_obj, tvec_diff_obj);

/*
    point2col(tvecPoint, *tvec);
    Point3d rvecPoint = estimatedModel.rotationAxis.cross(groundModel.rotationAxis);
    rvecPoint *= hartleyDiff / norm(rvecPoint);
    point2col(rvecPoint, *rvec);
*/
  poseError.init(PoseRT(rvec_diff_obj, tvec_diff_obj), hartleyDiff, tvecDiff);

/*

  Point3d zRvecPoint_obj = estimatedModel.rotationAxis.cross(groundModel.rotationAxis);

  CV_Assert(norm(zRvecPoint_obj) > eps);
  zRvecPoint_obj *= hartleyDiff / norm(zRvecPoint_obj);
  Mat zRvec_obj;
  point2col(zRvecPoint_obj, zRvec_obj);
  const int dim = 3;
  Point3d tvecPoint_obj = groundModel.getObjectCenter() - estimatedModel.getObjectCenter();

  zRvec_obj = Mat::zeros(dim, 1, CV_64FC1);

  Mat zTvec_obj;
  point2col(tvecPoint_obj, zTvec_obj);

  //zTvec_obj = Mat::zeros(dim, 1, CV_64FC1);

  PoseRT zPose_obj = PoseRT(zRvec_obj, zTvec_obj);
  Mat withoutZRotation_Rt = estimatedModel.Rt_obj2cam * zPose_obj.getProjectiveMatrix() * estimatedModel.Rt_obj2cam.inv(DECOMP_SVD) * est_cam.getProjectiveMatrix();
  PoseRT withoutZRotationPose = PoseRT(withoutZRotation_Rt);


  double xyRotationDiff, xyTranslationDiff;
  PoseRT::computeDistance(ground_cam, withoutZRotationPose, xyRotationDiff, xyTranslationDiff, groundModel.Rt_obj2cam);
  //PoseRT::computeDistance(ground_cam, withoutZRotationPose, xyRotationDiff, xyTranslationDiff);
  cout << "xy: " << xyTranslationDiff << " " << xyRotationDiff * 180.0 / CV_PI<< endl;
*/
}

void drawResponse(const std::vector<cv::linemod::Template>& templates,
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T)
{
  static const cv::Scalar COLORS[5] = { CV_RGB(0, 0, 255),
                                        CV_RGB(0, 255, 0),
                                        CV_RGB(255, 255, 0),
                                        CV_RGB(255, 140, 0),
                                        CV_RGB(255, 0, 0) };

  for (int m = 0; m < num_modalities; ++m)
  {
    // NOTE: Original demo recalculated max response for each feature in the TxT
    // box around it and chose the display color based on that response. Here
    // the display color just depends on the modality.
    cv::Scalar color = COLORS[m];

    for (int i = 0; i < (int)templates[m].features.size(); ++i)
    {
      cv::linemod::Feature f = templates[m].features[i];
      cv::Point pt(f.x + offset.x, f.y + offset.y);
      cv::circle(dst, pt, T / 2, color);
    }
  }
}

void generatePoses(int rotationsPosesCount, std::vector<PoseRT> &poses)
{
  //TODO: move up
  const string rotationsFilename = "rotations.txt";
  const float minDistance = 0.4f;
  const float maxDistance = 0.9f;
  const float distanceStep = 0.03f;

  std::ifstream fin(rotationsFilename);
  CV_Assert(fin.is_open());
  vector<PoseRT> rotationPoses;
  for (size_t i = 0; i < rotationsPosesCount; ++i)
  {
    double x, y, z, w;
    fin >> x >> y >> z >> w;
    PoseRT currentPose;
    currentPose.setQuaternion(x, y, z, w);
    rotationPoses.push_back(currentPose);
  }

  poses.clear();
  for (float distance = minDistance; distance < maxDistance; distance += distanceStep)
  {
    PoseRT translationPose;
    translationPose.tvec.at<double>(2) = distance;
    for (size_t i = 0; i < rotationPoses.size(); ++i)
    {
      PoseRT generatedPose = translationPose * rotationPoses[i];
      poses.push_back(generatedPose);
    }
  }

/*
  //TODO: move up
  const bool hasRotationSymmetry = true;
  const int silhouetteCount = 20;

  for (float distance = minDistance; distance < maxDistance; distance += distanceStep)
  {
    for (int k = 0; k < silhouetteCount; ++k)
    {
      for(int i = 0; i < silhouetteCount; ++i)
      {
        for (int j = 0; j < silhouetteCount; ++j)
        {
          if (hasRotationSymmetry && j != 0)
          {
            continue;
          }

          //TODO: generate silhouettes uniformly on the viewing sphere
          //TODO: move up
          double xAngle = 2.0 * CV_PI / 3.0 + i * 2.0 * (CV_PI / 3.0) / (silhouetteCount - 1.0);
          double yAngle = j * (2 * CV_PI) / silhouetteCount;
          const int dim = 3;
          Mat x_rvec_obj = (Mat_<double>(dim, 1) << xAngle, 0.0, 0.0);
          Mat y_rvec_obj = (Mat_<double>(dim, 1) << 0.0, yAngle, 0.0);

          //TODO: move up
          double zAngle = -CV_PI/6 + (CV_PI / 3) * (k / (silhouetteCount - 1.0));
          Mat z_rvec_obj = (Mat_<double>(dim, 1) << 0.0, 0.0, zAngle);
          Mat zeroTvec = Mat::zeros(dim, 1, CV_64FC1);

          PoseRT rotationPose = PoseRT(z_rvec_obj, zeroTvec) * PoseRT(x_rvec_obj, zeroTvec) * PoseRT(y_rvec_obj, zeroTvec);
          PoseRT translationPose;
          translationPose.tvec.at<double>(2) = distance;

          PoseRT generatedPose = translationPose * rotationPose;
          poses.push_back(generatedPose);
        }
      }
    }
  }
*/
  cout << "all templates count: " << poses.size() << endl;
}

void generateTrainPoses(int rotationsPosesCount, const PinholeCamera &camera, const EdgeModel &edgeModel, std::map<int, PoseRT> &trainPoses)
{
  trainPoses.clear();

  vector<PoseRT> sampledPoses_obj;
  generatePoses(rotationsPosesCount, sampledPoses_obj);

  EdgeModel canonicalEdgeModel = edgeModel;
  PoseRT model2canonicalPose;
  canonicalEdgeModel.rotateToCanonicalPose(camera, model2canonicalPose, 0.0f);

  for (size_t i = 0; i < sampledPoses_obj.size(); ++i)
  {
    PoseRT silhouettePose_cam = sampledPoses_obj[i].obj2cam(canonicalEdgeModel.Rt_obj2cam);
    silhouettePose_cam = silhouettePose_cam * model2canonicalPose;
    trainPoses[i] = silhouettePose_cam;
  }
}

void getSortedTrainIndices(const EdgeModel &edgeModel, const std::map<int, PoseRT> &trainPoses, const std::map<int, PoseRT> &testPoses,
                           std::multimap<double, int> &sortedTrainIndices)
{
  for (std::map<int, PoseRT>::const_iterator trainIt = trainPoses.begin(); trainIt != trainPoses.end(); ++trainIt)
  {
    double minDistance = std::numeric_limits<double>::max();

    for (std::map<int, PoseRT>::const_iterator testIt = testPoses.begin(); testIt != testPoses.end(); ++testIt)
    {
      PoseError poseError;
      //TODO: measure distance in z-direction only
      evaluatePoseWithRotation(edgeModel, trainIt->second, testIt->second, poseError);
      double distance = poseError.getDifference();
      if (distance < minDistance)
      {
        minDistance = distance;
      }
    }
    sortedTrainIndices.insert(std::make_pair(minDistance, trainIt->first));
  }
}

int main(int argc, char *argv[])
{
  std::system("date");

  if (argc != 5)
  {
    cout << argv[0] << " <trainBaseFolder> <modelsPath> <baseFoldler> <testObjectName>" << endl;
    return -1;
  }
  string trainBaseFolder = argv[1];
  string modelsPath = argv[2];
  string baseFolder = argv[3];
  string testObjectName = argv[4];

  const string testFolder = baseFolder + "/" + testObjectName + "/";
  const string kinectCameraFilename = baseFolder + "/center.yml";
//  const vector<string> objectNames = {"bank", "bucket"};
//  const vector<string> objectNames = {"bank", "bottle", "bucket", "glass", "wineglass"};
  const string registrationMaskFilename = baseFolder + "/registrationMask.png";
  const vector<string> objectNames = {testObjectName};

  const string initialPosesFilename = "initialPoses_" + testObjectName + ".txt";

  const int matching_threshold = 70;
//  const int rotationsPosesCount = 2000;
  const int rotationsPosesCount = 60;
  const float templatesCountBase = 1.05f;
  const float matchingThresholdBase = 0.8f;

  bool simulateTrainPoses = true;
  bool simulateTrainImages = true;

  vector<string> initialPosesStrings;
  readLinesInFile(initialPosesFilename, initialPosesStrings);
  vector<int> initialPosesCount;
  for (size_t i = 0; i < initialPosesStrings.size(); ++i)
  {
    //TODO: fix
    initialPosesCount.push_back(atoi(initialPosesStrings[i].c_str()));
  }

  CV_Assert(objectNames.size() == 1);
  TODBaseImporter dataImporter(testFolder);
  vector<EdgeModel> edgeModels(objectNames.size());
  for (size_t i = 0; i < objectNames.size(); ++i)
  {
    dataImporter.importEdgeModel(modelsPath, objectNames[i], edgeModels[i]);
    cout << "All points in the model: " << edgeModels[i].points.size() << endl;
    cout << "Surface points in the model: " << edgeModels[i].stableEdgels.size() << endl;
  }

  Mat registrationMask;
  dataImporter.importRegistrationMask(registrationMaskFilename, registrationMask);

  PinholeCamera kinectCamera;
  if(!kinectCameraFilename.empty())
  {
    dataImporter.readCameraParams(kinectCameraFilename, kinectCamera, false);
    CV_Assert(kinectCamera.imageSize == Size(640, 480));
  }

  vector<int> testIndices;
  dataImporter.importTestIndices(testIndices);

  CV_Assert(edgeModels.size() == 1);
  std::map<int, PoseRT> trainPoses;
  if (simulateTrainPoses)
  {
    generateTrainPoses(rotationsPosesCount, kinectCamera, edgeModels[0], trainPoses);
  }
  else
  {
    TODBaseImporter trainDataImporter(trainBaseFolder + "/" + testObjectName + "/");
    trainDataImporter.importAllGroundTruth(trainPoses);
  }

  std::map<int, PoseRT> testPoses;
  dataImporter.importAllGroundTruth(testPoses);

  std::multimap<double, int> sortedTrainIndices;
  CV_Assert(edgeModels.size() == 1);
  getSortedTrainIndices(edgeModels[0], trainPoses, testPoses, sortedTrainIndices);
//  getBestTrainPoses(edgeModels[0], trainPoses, testPoses, templatesCount, bestTrainPoses);

  float step = 1.0f;
//  for (size_t templatesCount = 1; templatesCount < trainPoses.size(); templatesCount += cvRound(step))
  size_t templatesCount = trainPoses.size();
  {
    step *= templatesCountBase;

    vector<PoseRT> bestTrainPoses;
    vector<int> bestTrainIndices;
    int addedTemplatesCount = 0;
    for (std::multimap<double, int>::iterator it = sortedTrainIndices.begin();
         it != sortedTrainIndices.end(); ++it)
    {
      if (addedTemplatesCount < templatesCount)
      {
#ifdef VERBOSE
        cout << it->first << " -> " << it->second << endl;
#endif
        bestTrainPoses.push_back(trainPoses.find(it->second)->second);
        bestTrainIndices.push_back(it->second);
        ++addedTemplatesCount;
      }
    }

    std::cout << "Training Line2D...  " << std::flush;
    CV_Assert(edgeModels.size() == 1);
    cv::Ptr<Line2D> detector;
    if (simulateTrainImages)
    {
      detector = trainLine2D(kinectCamera, edgeModels[0], objectNames[0], bestTrainPoses);
    }
    else
    {
      detector = trainLine2D(trainBaseFolder, objectNames, &bestTrainIndices);
    }
    std::cout << "done." << std::endl;
    CV_Assert(detector->numTemplates() != 0);

    std::vector<std::string> ids = detector->classIds();
    int num_classes = detector->numClasses();
    printf("Loaded %s with %d classes and %d templates\n",
           argv[1], num_classes, detector->numTemplates());
    if (!ids.empty())
    {
      printf("Class ids:\n");
      std::copy(ids.begin(), ids.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    }
    int num_modalities = (int)detector->getModalities().size();


    vector<PoseError> bestPoses;
    vector<double> allRecognitionTimes;
    for(size_t testIdx = 0; testIdx < testIndices.size(); testIdx++)
    {
      srand(42);
      RNG &rng = theRNG();
      rng.state = 0xffffffff;

  #if defined(VISUALIZE_POSE_REFINEMENT) && defined(USE_3D_VISUALIZATION)
      boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("transparent experiments"));
  #endif
      int testImageIdx = testIndices[ testIdx ];
#ifdef VERBOSE
      cout << "Test: " << testIdx << " " << testImageIdx << endl;
#endif

      Mat kinectDepth, kinectBgrImage;
      if(!kinectCameraFilename.empty())
      {
        dataImporter.importBGRImage(testImageIdx, kinectBgrImage);
        dataImporter.importDepth(testImageIdx, kinectDepth);
      }

      std::vector<cv::Mat> sources;
      sources.push_back(kinectBgrImage);
      sources.push_back(kinectDepth);

      PoseRT model2test_ground;
      dataImporter.importGroundTruth(testImageIdx, model2test_ground);

      //TODO: move up
      //good clutter
      GlassSegmentatorParams glassSegmentationParams;
      glassSegmentationParams.openingIterations = 15;
      glassSegmentationParams.closingIterations = 12;
      glassSegmentationParams.finalClosingIterations = 32;
      glassSegmentationParams.grabCutErosionsIterations = 4;
      GlassSegmentator glassSegmentator(glassSegmentationParams);
      int numberOfComponents;
      Mat glassMask;
      glassSegmentator.segment(kinectBgrImage, kinectDepth, registrationMask, numberOfComponents, glassMask);

      //TODO: move up
      const int maskWidth = 40;
      dilate(glassMask, glassMask, Mat(), Point(-1, -1), maskWidth);
      vector<Mat> masks;
      masks.push_back(glassMask);
      masks.push_back(glassMask);

      TickMeter recognitionTime;
      recognitionTime.start();

      std::vector<cv::linemod::Match> matches;
      std::vector<std::string> class_ids;
      std::vector<cv::Mat> quantized_images;
      float matchingThreshold = matching_threshold;
      do
      {
        detector->match(sources, matchingThreshold, matches, class_ids, quantized_images, masks);
        matchingThreshold *= matchingThresholdBase;
      }
      while(matches.empty());
      allRecognitionTimes.push_back(recognitionTime.getTimeSec());
      CV_Assert(!matches.empty());

      int classes_visited = 0;
      std::set<std::string> visited;

      if (objectNames.size() == 1)
      {
        int objectIndex = 0;


        vector<PoseRT> allDetectedPoses;
        vector<float> confidences;
        for (size_t i = 0; i < matches.size(); ++i)
        {
          PoseRT pose_cam = detector->getTestPose(kinectCamera, matches[i]);
          allDetectedPoses.push_back(pose_cam);
          confidences.push_back(matches[i].similarity);
        }

#ifdef SUPPRESS_POSES
        //TODO: move up
        const float neighborMaxRotation = 0.1f;
        const float neighborMaxTranslation = 0.02f;
        vector<bool> isFilteredOut;
        suppress3DPoses(edgeModels[0], confidences, allDetectedPoses, neighborMaxRotation, neighborMaxTranslation, isFilteredOut);

        int remainedPosesCount = 0;
        for (size_t i = 0; i < isFilteredOut.size(); ++i)
        {
          if (!isFilteredOut[i])
             ++remainedPosesCount;
        }
        cout << "Suppression: " << matches.size() << " --> " << remainedPosesCount << endl;
#endif

        vector<PoseError> poseErrors;
        int addedPosesCount = 0;
        for (size_t i = 0; i < matches.size() && addedPosesCount < initialPosesCount[testIdx]; ++i)
        {
#ifdef SUPPRESS_POSES
          if (isFilteredOut[i])
            continue;
#endif

          PoseRT pose_cam = allDetectedPoses[i];
          PoseError currentPoseError;
          evaluatePoseWithRotation(edgeModels[objectIndex], pose_cam, model2test_ground, currentPoseError);
          poseErrors.push_back(currentPoseError);
          ++addedPosesCount;

  #ifdef VERBOSE
          cout << i << ":\t" << currentPoseError << endl;
  #endif

        cv::linemod::Match m = matches[i];

#ifdef VERBOSE
          printf("Similarity[%d]: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
                 i, m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
          std::flush(std::cout);
#endif

#ifdef VISUALIZE_POSES
          // Draw matching template
          cv::Mat display = kinectBgrImage.clone();
          const std::vector<cv::linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
          drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0));
          imshow("display", display);
          waitKey();
#endif
        }
        vector<PoseError>::iterator bestPoseIt = std::min_element(poseErrors.begin(), poseErrors.end());
        int bestPoseIdx = std::distance(poseErrors.begin(), bestPoseIt);
        cout << "Best pose: " << poseErrors.at(bestPoseIdx) << endl;
        bestPoses.push_back(poseErrors[bestPoseIdx]);
      }
    }

    if (objectNames.size() == 1)
    {
      cout << "templates count: " << bestTrainPoses.size() << endl;
//      const double cmThreshold = 3.0;
      const double cmThreshold = 6.0;
      PoseError::evaluateErrors(bestPoses, cmThreshold);
    }
  }

  return 0;
}

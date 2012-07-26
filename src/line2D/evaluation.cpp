#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/line2D.hpp"
#include "edges_pose_refiner/poseError.hpp"

#include <iostream>
#include <iterator>
#include <set>
#include <fstream>

using namespace cv;
using std::cout;
using std::endl;

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

void generatePoses(std::vector<PoseRT> &poses)
{
  //TODO: move up
  const int rotationsPosesCount = 100;
  const string rotationsFilename = "rotations.txt";
  const float minDistance = 0.5f;
  const float maxDistance = 0.7f;
  const float distanceStep = 0.01f;

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

void generateTrainPoses(const PinholeCamera &camera, const EdgeModel &edgeModel, std::map<int, PoseRT> &trainPoses)
{
  trainPoses.clear();

  vector<PoseRT> sampledPoses_obj;
  generatePoses(sampledPoses_obj);

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

void getBestTrainPoses(const EdgeModel &edgeModel, const std::map<int, PoseRT> &trainPoses, const std::map<int, PoseRT> &testPoses,
                       std::vector<PoseRT> &bestTrainPoses)
{
  //TODO: move up
//  int templatesCount = 5;
  int templatesCount = 50;
//  int templatesCount = 5000;
//  int templatesCount = 400;
//  const int templatesCount = 2000;

  std::multimap<double, int> sortedTrainIndices;
  for (std::map<int, PoseRT>::const_iterator trainIt = trainPoses.begin(); trainIt != trainPoses.end(); ++trainIt)
  {
    double minDistance = std::numeric_limits<double>::max();

    for (std::map<int, PoseRT>::const_iterator testIt = testPoses.begin(); testIt != testPoses.end(); ++testIt)
    {
      PoseError poseError;
      evaluatePoseWithRotation(edgeModel, trainIt->second, testIt->second, poseError);
      double distance = poseError.getDifference();
      if (distance < minDistance)
      {
        minDistance = distance;
      }
    }
    sortedTrainIndices.insert(std::make_pair(minDistance, trainIt->first));
  }


  int addedTemplatesCount = 0;
  bestTrainPoses.clear();
  for (std::multimap<double, int>::iterator it = sortedTrainIndices.begin();
       it != sortedTrainIndices.end(); ++it)
  {
    if (addedTemplatesCount < templatesCount)
    {
      cout << it->first << " -> " << it->second << endl;
      bestTrainPoses.push_back(trainPoses.find(it->second)->second);
      ++addedTemplatesCount;
    }
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

  const int matching_threshold = 80;

  bool simulateTrainData = true;

  CV_Assert(objectNames.size() == 1);
  TODBaseImporter dataImporter(testFolder);
  vector<EdgeModel> edgeModels(objectNames.size());
  for (size_t i = 0; i < objectNames.size(); ++i)
  {
    dataImporter.importEdgeModel(modelsPath, objectNames[i], edgeModels[i]);
    cout << "All points in the model: " << edgeModels[i].points.size() << endl;
    cout << "Surface points in the model: " << edgeModels[i].stableEdgels.size() << endl;
  }

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
  if (simulateTrainData)
  {
    generateTrainPoses(kinectCamera, edgeModels[0], trainPoses);
  }
  else
  {
    TODBaseImporter trainDataImporter(trainBaseFolder + "/" + testObjectName + "/");
    trainDataImporter.importAllGroundTruth(trainPoses);
  }

  std::map<int, PoseRT> testPoses;
  dataImporter.importAllGroundTruth(testPoses);

  vector<PoseRT> bestTrainPoses;
  CV_Assert(edgeModels.size() == 1);
  getBestTrainPoses(edgeModels[0], trainPoses, testPoses, bestTrainPoses);


  std::cout << "Training Line2D...  " << std::flush;
  CV_Assert(edgeModels.size() == 1);
  cv::Ptr<Line2D> detector = trainLine2D(kinectCamera, edgeModels[0], objectNames[0], bestTrainPoses);
  std::cout << "done." << std::endl;

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


  Mat registrationMask = imread(registrationMaskFilename, CV_LOAD_IMAGE_GRAYSCALE);
  CV_Assert(!registrationMask.empty());

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
    cout << "Test: " << testIdx << " " << testImageIdx << endl;

    Mat kinectDepth, kinectBgrImage;
    if(!kinectCameraFilename.empty())
    {
      dataImporter.importBGRImage(testImageIdx, kinectBgrImage);
      dataImporter.importDepth(testImageIdx, kinectDepth);
    }

    std::vector<cv::Mat> sources;
    sources.push_back(kinectBgrImage);
    sources.push_back(kinectDepth);
    cv::Mat display = kinectBgrImage.clone();




    PoseRT model2test_ground;
    dataImporter.importGroundTruth(testImageIdx, model2test_ground);

    pcl::PointCloud<pcl::PointXYZ> testPointCloud;
    dataImporter.importPointCloud(testImageIdx, testPointCloud);

    TickMeter recognitionTime;
    recognitionTime.start();

    std::vector<cv::linemod::Match> matches;
    std::vector<std::string> class_ids;
    std::vector<cv::Mat> quantized_images;
    detector->match(sources, (float)matching_threshold, matches, class_ids, quantized_images);

    recognitionTime.stop();
    allRecognitionTimes.push_back(recognitionTime.getTimeSec());

    int classes_visited = 0;
    std::set<std::string> visited;

    for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
    {
      cv::linemod::Match m = matches[i];

      if (visited.insert(m.class_id).second)
      {
        ++classes_visited;

        printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
               m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);

        // Draw matching template
//        const std::vector<cv::linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
//        drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0));
//        imshow("display", display);
//        waitKey();
      }
    }

    if (objectNames.size() == 1)
    {
      int objectIndex = 0;

      PoseError poseError;
      PoseRT pose_cam = detector->getTestPose(kinectCamera, matches[0]);
      evaluatePoseWithRotation(edgeModels[objectIndex], pose_cam, model2test_ground, poseError);
      cout << poseError << endl;

      bestPoses.push_back(poseError);
    }
  }

  if (objectNames.size() == 1)
  {
    const double cmThreshold = 2.0;
//    const double cmThreshold = 200.0;
    PoseError::evaluateErrors(bestPoses, cmThreshold);
  }

  return 0;
}

#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/line2D.hpp"
#include "edges_pose_refiner/poseError.hpp"

#include <iostream>
#include <iterator>
#include <set>

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


  std::cout << "Training Line2D...  " << std::flush;
  cv::Ptr<Line2D> detector = trainLine2D(trainBaseFolder, objectNames);
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



  TODBaseImporter dataImporter(testFolder);

  PinholeCamera kinectCamera;
  if(!kinectCameraFilename.empty())
  {
    dataImporter.readCameraParams(kinectCameraFilename, kinectCamera, false);
    CV_Assert(kinectCamera.imageSize == Size(640, 480));
  }

  vector<EdgeModel> edgeModels(objectNames.size());
  for (size_t i = 0; i < objectNames.size(); ++i)
  {
    dataImporter.importEdgeModel(modelsPath, objectNames[i], edgeModels[i]);
    cout << "All points in the model: " << edgeModels[i].points.size() << endl;
    cout << "Surface points in the model: " << edgeModels[i].stableEdgels.size() << endl;
  }

  vector<int> testIndices;
  dataImporter.importTestIndices(testIndices);

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

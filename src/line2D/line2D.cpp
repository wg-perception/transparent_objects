#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/line2D.hpp"

using namespace cv;
using std::cout;
using std::endl;

Line2D::Line2D()
{
}

Line2D::Line2D(const std::vector<cv::Ptr<linemod::Modality> >& modalities, const std::vector<int>& T_pyramid)
  : linemod::Detector(modalities, T_pyramid)
{
}

int Line2D::addTemplate(const std::vector<Mat> &sources, const std::string &class_id,
                        const Mat &object_mask, Rect *bounding_box, const PoseRT *pose)
{
  Rect box;
  Rect *boxPtr = bounding_box ? bounding_box : &box;
  int templateIndex = addTemplate(sources, class_id, object_mask, boxPtr);
  if (templateIndex < 0)
  {
    return templateIndex;
  }

  origins[class_id].push_back(boxPtr->tl());

  if (pose != 0)
  {
    if (templateIndex != templatePoses[class_id].size())
    {
      cout << templateIndex << endl;
      cout << templatePoses[class_id].size() << endl;
    }
    CV_Assert(templateIndex == templatePoses[class_id].size());
    templatePoses[class_id].push_back(*pose);
  }

  return templateIndex;
}

PoseRT Line2D::getTrainPose(const cv::linemod::Match &match)
{
  return templatePoses[match.class_id].at(match.template_id);
}

PoseRT Line2D::getTestPose(const PinholeCamera &camera, const cv::linemod::Match &match)
{
  Point pt = origins[match.class_id].at(match.template_id);
  Point detectionPt = Point(match.x, match.y);
  Point shift = detectionPt - pt;

  Mat affineTransformation = (Mat_<double>(2, 3) << 1.0, 0.0, shift.x, 0.0, 1.0, shift.y);
  Mat homography = affine2homography(affineTransformation);

  //TODO: which inversion method is better?
  Mat fullTransform = camera.cameraMatrix.inv(DECOMP_SVD) * homography * camera.cameraMatrix;

  PoseRT trainPose = getTrainPose(match);
  CV_Assert(trainPose.getTvec().type() == CV_64FC1);
  double meanZ = trainPose.getTvec().at<double>(2);
  const float eps = 1e-6;
  CV_Assert(meanZ > eps);

  Point3d tvec;
  tvec.z = 0.0;
  tvec.x = fullTransform.at<double>(0, 2) * meanZ;
  tvec.y = fullTransform.at<double>(1, 2) * meanZ;

  Mat tvecMat;
  point2col(tvec, tvecMat);
  PoseRT pose2d_cam;
  pose2d_cam.tvec = tvecMat;

  PoseRT pose_cam = camera.extrinsics.inv() * pose2d_cam * trainPose;
  return pose_cam;
}

Ptr<Line2D> getDefaultLine2D()
{
  const int T_DEFAULTS[] = {5, 8};
  std::vector< Ptr<linemod::Modality> > modalities;
  modalities.push_back(new linemod::ColorGradient);
//  modalities.push_back(new linemod::ColorGradient(10.0f, 63, 30.0f));
  return new Line2D(modalities, std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
}

Ptr<Line2D> trainLine2D(const std::string &baseFolder, const std::vector<std::string> &objectNames,
                        std::vector<int> *testIndicesPtr)
{
  Ptr<Line2D> line2D = getDefaultLine2D();
  for (size_t objectIndex = 0; objectIndex < objectNames.size(); ++objectIndex)
  {
    std::string currentObjectName = objectNames[objectIndex];
    const string testFolder = baseFolder + "/" + currentObjectName + "/";
    TODBaseImporter dataImporter(testFolder);

    vector<int> testIndices;
    if (testIndicesPtr != 0)
    {
      testIndices = *testIndicesPtr;
    }
    else
    {
      dataImporter.importTestIndices(testIndices);
    }

    for (size_t testIndex = 0; testIndex < testIndices.size(); ++testIndex)
    {
      int imageIndex = testIndices[testIndex];

      Mat bgrImage;
      dataImporter.importBGRImage(imageIndex, bgrImage);

      vector<Mat> sources;
      sources.push_back(bgrImage);

      PoseRT trainPose;
      dataImporter.importGroundTruth(imageIndex, trainPose);

      //TODO: invesigate which mask to use
      Mat objectMask;
      dataImporter.importRawMask(imageIndex, objectMask);
      //objectMask = (objectMask == 255);

      line2D->addTemplate(sources, currentObjectName, objectMask, 0, &trainPose);
    }
  }

  return line2D;
}

Ptr<Line2D> trainLine2D(const PinholeCamera &camera, const EdgeModel &edgeModel, const std::string &objectName, const std::vector<PoseRT> &trainPoses)
{
  Ptr<Line2D> line2D = getDefaultLine2D();
  for (size_t i = 0; i < trainPoses.size(); ++i)
  {
    Silhouette silhouette;
    //TODO: move up
    float downFactor = 1.0;
    int closingIterationsCount = 6;
    cv::Ptr<PinholeCamera> cameraPtr = new PinholeCamera(camera);
    edgeModel.getSilhouette(cameraPtr, trainPoses[i], silhouette, downFactor, closingIterationsCount);

    Mat mask(camera.imageSize, CV_8UC1, Scalar(0));
    silhouette.draw(mask, Scalar::all(255), -1);

    Mat colorMask;
    cvtColor(mask, colorMask, COLOR_GRAY2BGR);

    vector<Mat> sources;
    sources.push_back(colorMask);

    line2D->addTemplate(sources, objectName, mask, 0, &trainPoses[i]);
  }

  return line2D;
}

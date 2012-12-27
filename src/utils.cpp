/*
 * utils.cpp
 *
 *  Created on: Apr 23, 2011
 *      Author: Ilya Lysenkov
 */

#include "edges_pose_refiner/utils.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/poseRT.hpp"
#include "chamfer_matching/chamfer_matching.h"

#ifdef USE_3D_VISUALIZATION
#include <boost/thread/thread.hpp>
#endif

#include <fstream>

using std::cout;
using std::endl;

using namespace cv;

//#define VISUALIZE_POSE_REFINEMENT

//#define USE_3D_VISUALIZATION

//TODO: is Projective right name?
void createProjectiveMatrix(const cv::Mat &R, const cv::Mat &t, cv::Mat &Rt)
{
  CV_Assert(R.type() == CV_64FC1);
  CV_Assert(t.type() == CV_64FC1);

  Rt.create(4, 4, CV_64FC1);
  Rt.at<double>(3, 0) = 0.;
  Rt.at<double>(3, 1) = 0.;
  Rt.at<double>(3, 2) = 0.;
  Rt.at<double>(3, 3) = 1.;

  Mat roi_R = Rt(Range(0, 3), Range(0, 3));
  CV_Assert(roi_R.rows == 3 && roi_R.cols == 3);

  if(R.rows == 3 && R.cols == 3)
  {
    R.copyTo(roi_R);
  }
  else
  {
    Mat fullR;
    Rodrigues(R, fullR);
    fullR.copyTo(roi_R);
  }

  Mat roi_t = Rt(Range(0, 3), Range(3,4));
  t.copyTo(roi_t);
}

void getRvecTvec(const Mat &projectiveMatrix, Mat &rvec, Mat &tvec)
{
  CV_Assert(projectiveMatrix.rows == 4 && projectiveMatrix.cols == 4);

  Rodrigues(projectiveMatrix(Range(0, 3), Range(0, 3)), rvec);
  projectiveMatrix(Range(0, 3), Range(3, 4)).copyTo(tvec);

  CV_Assert(rvec.rows == 3 && rvec.cols == 1);
  CV_Assert(tvec.rows == 3 && tvec.cols == 1);
  CV_Assert(rvec.type() == CV_64FC1 && tvec.type() == CV_64FC1);
}


void getTransformationMatrix(const cv::Mat &R_obj2cam, const cv::Mat &t_obj2cam, const cv::Mat &rvec_Object, const cv::Mat &tvec_Object, cv::Mat &transformationMatrix)
{
  Mat Rt_obj2cam;
  createProjectiveMatrix(R_obj2cam, t_obj2cam, Rt_obj2cam);

  getTransformationMatrix(Rt_obj2cam, rvec_Object, tvec_Object, transformationMatrix);
}

void getTransformationMatrix(const cv::Mat &Rt_obj2cam, const cv::Mat &rvec_Object, const cv::Mat &tvec_Object, cv::Mat &transformationMatrix)
{
  Mat Rt_obj;
  createProjectiveMatrix(rvec_Object, tvec_Object, Rt_obj);

  transformationMatrix = Rt_obj2cam * Rt_obj * Rt_obj2cam.inv(DECOMP_SVD);
}

void getRotationTranslation(const cv::Mat &projectiveMatrix, cv::Mat &R, cv::Mat &t)
{
  projectiveMatrix(Range(0, 3), Range(0, 3)).copyTo(R);
  projectiveMatrix(Range(0, 3), Range(3, 4)).copyTo(t);
}


void transformPoint(const cv::Mat &Rt, const cv::Point3d &point, cv::Point3d &transformedPoint)
{
  Mat transformedPointMat;
  perspectiveTransform(Mat(vector<Point3d>(1, point)), transformedPointMat, Rt);
  vector<Point3d> transformedPointVec = transformedPointMat;
  transformedPoint = transformedPointVec[0];
}

void readLinesInFile(const string &filename, std::vector<string> &lines)
{
  lines.clear();
  std::ifstream file(filename.c_str());
  if(!file.is_open())
  {
    CV_Error(CV_StsBadArg, "Cannot open file " + filename);
  }

  while(!file.eof())
  {
    string curLine;
    file >> curLine;
    if(curLine.empty())
    {
      break;
    }

    lines.push_back(curLine);
  }
  file.close();
}

bool isPointInside(const cv::Mat &image, cv::Point pt)
{
  return (0 <= pt.x && pt.x < image.cols && 0 <= pt.y && pt.y < image.rows);
}

void mask2contour(const cv::Mat &mask, std::vector<cv::Point2f> &contour)
{
  Mat maskClone = mask.clone();
  vector<vector<Point> > allContours;
  findContours(maskClone, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  contour.clear();
  for (size_t i = 0; i < allContours.size(); ++i)
  {
    std::copy(allContours[i].begin(), allContours[i].end(), std::back_inserter(contour));
  }
}

void hcat(const Mat &A, const Mat &B, Mat &result)
{
  result = A.t();
  Mat bt = B.t();
  result.push_back(bt);
  result = result.t();
}

void readFiducial(const string &filename, Mat &blackBlobsObject, Mat &whiteBlobsObject, Mat &allBlobsObject)
{
  FileStorage fiducialFS(filename, FileStorage::READ);
  if (!fiducialFS.isOpened())
  {
    CV_Error(CV_StsBadArg, "Cannot read fiducials from " + filename);
  }
  fiducialFS["fiducial"]["templates"][0] >> whiteBlobsObject;
  fiducialFS["fiducial"]["templates"][1] >> blackBlobsObject;
  fiducialFS.release();

//  hcat(blackBlobsObject, whiteBlobsObject, allBlobsObject);
  allBlobsObject = blackBlobsObject.clone();
  allBlobsObject.push_back(whiteBlobsObject);

  CV_Assert(!blackBlobsObject.empty() && !whiteBlobsObject.empty());
}

void detectFiducial(const cv::Mat &bgrImage, cv::Mat &blackBlobs, cv::Mat &whiteBlobs)
{
  //TODO: move up
  const Size boardSize(4, 11);
  SimpleBlobDetector::Params params;
  params.filterByInertia = true;
  params.minArea = 10;
  params.minDistBetweenBlobs = 5;
//  params.minArea = 6;
//  params.minDistBetweenBlobs = 3;

  params.blobColor = 0;
  Ptr<FeatureDetector> blackBlobDetector = new SimpleBlobDetector(params);

  params.blobColor = 255;
  Ptr<FeatureDetector> whiteBlobDetector = new SimpleBlobDetector(params);

  bool isBlackFound = findCirclesGrid(bgrImage, boardSize, blackBlobs, CALIB_CB_ASYMMETRIC_GRID | CALIB_CB_CLUSTERING, blackBlobDetector);
  bool isWhiteFound = findCirclesGrid(bgrImage, boardSize, whiteBlobs, CALIB_CB_ASYMMETRIC_GRID | CALIB_CB_CLUSTERING, whiteBlobDetector);

  if (!isBlackFound)
  {
      blackBlobs = Mat();
  }

  if (!isWhiteFound)
  {
      whiteBlobs = Mat();
  }
}

cv::Mat drawSegmentation(const cv::Mat &image, const cv::Mat &mask, const Scalar &color, int thickness)
{
  CV_Assert(!image.empty() && !mask.empty());
  Mat drawImage;
  if (image.channels() == 3)
  {
    drawImage = image.clone();
  }
  else
  {
    cvtColor(image, drawImage, CV_GRAY2BGR);
  }

  Mat glassMask = mask.clone();
  vector<vector<Point> > contours;
  findContours(glassMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
  drawContours(drawImage, contours, -1, color, thickness);
  return drawImage;
}

vector<Mat> drawEdgels(const std::vector<cv::Mat> &images, const vector<Point3f> &edgels3d,
                          const PoseRT &pose_cam,
                          const std::vector<PinholeCamera> &cameras,
                          cv::Scalar color, float blendingFactor)
{
  const float eps = 1e-5;
  CV_Assert(-eps < blendingFactor && blendingFactor < 1.0f + eps);
  vector<Mat> drawImages(images.size());
  for(size_t i=0; i<images.size(); i++)
  {
    if (images[i].channels() == 1)
    {
      cvtColor(images[i], drawImages[i], CV_GRAY2BGR);
    }
    else
    {
      drawImages[i] = images[i].clone();
    }
    PoseRT curPose = cameras[i].extrinsics * pose_cam;
    vector<Point2f> projectedEdgels;
    projectPoints(Mat(edgels3d), curPose.getRvec(), curPose.getTvec(), cameras[i].cameraMatrix, cameras[i].distCoeffs, projectedEdgels);

    for(size_t j=0; j<projectedEdgels.size(); j++)
    {
      //circle(drawImages[i], projectedEdgels[j], 2, Scalar(0, 0, 255), -1);
      circle(drawImages[i], projectedEdgels[j], 1, color, -1);
    }

    drawImages[i] = blendingFactor * drawImages[i] + (1.0f - blendingFactor) * images[i];
  }

  return drawImages;
}

Mat drawEdgels(const cv::Mat &image, const vector<Point3f> &edgels3d, const PoseRT &pose_cam, const PinholeCamera &camera,
               cv::Scalar color, float blendingFactor)
{
  vector<Mat> images(1, image);
  vector<PinholeCamera> allCameras(1, camera);
  return drawEdgels(images, edgels3d, pose_cam, allCameras, color, blendingFactor)[0];
}

vector<Mat> showEdgels(const std::vector<cv::Mat> &images, const vector<Point3f> &edgels3d,
                       const PoseRT &pose_cam,
                       const std::vector<PinholeCamera> &cameras,
                       const string &title,
                       cv::Scalar color)
{
  vector<Mat> drawImages = drawEdgels(images, edgels3d, pose_cam, cameras, color);
  for (size_t i = 0; i < images.size(); ++i)
  {
    std::stringstream titleStream;
    titleStream << title << " " << i;
    imshow(titleStream.str(), drawImages[i]);
  }
  return drawImages;
}

Mat showEdgels(const cv::Mat &image, const vector<Point3f> &edgels3d, const PoseRT &pose_cam, const PinholeCamera &camera, const string &title, cv::Scalar color)
{
  Mat drawImage = drawEdgels(image, edgels3d, pose_cam, camera, color);
  imshow(title, drawImage);
  return drawImage;
}

/*
void publishTable(const Vec4f &tablePlane, int id, Scalar color, ros::Publisher *pt_pub)
{
  float bound = 0.5;
  float step = 0.01;
  vector<Point3f> points;
  for (float x = -bound; x < bound; x += step)
  {
    for (float y = -bound; y < bound; y += step)
    {
      float z = (-tablePlane[3] - x * tablePlane[0] - y * tablePlane[1]) / tablePlane[2];
      points.push_back(Point3f(x, y, z));
    }
  }

  publishPoints(points, *pt_pub, id, color);
}
*/

void writePointCloud(const string &filename, const std::vector<cv::Point3f> &pointCloud)
{
  std::ofstream fout(filename.c_str());
  fout << format(Mat(pointCloud), "csv");
  fout.close();
}

void readPointCloud(const string &filename, std::vector<cv::Point3f> &pointCloud, std::vector<cv::Point3f> *normals)
{
  pointCloud.clear();
  if(normals != 0)
    normals->clear();
  std::ifstream file(filename.c_str());
  if (!file.is_open())
  {
    CV_Error(CV_StsBadArg, "Cannot open the file " + filename);
  }

  const int extSz = 3;
  string ext = filename.substr(filename.size() - extSz, extSz);
  if(ext == "ply")
  {
    while(!file.eof())
    {
      const int sz = 1024;
      char line[sz];
      file.getline(line, sz);
      int res = strcmp("end_header", line);

      if(res == 0)
        break;
    }
  }

  while(!file.eof())
  {
    Point3f pt;
    file >> pt.x >> pt.y;
    if(file.eof())
      break;
    file >> pt.z;
    pointCloud.push_back(pt);

    if(normals != 0)
    {
      Point3f pt;
      file >> pt.x >> pt.y >> pt.z;
      normals->push_back(pt);
    }
  }

  if(normals != 0)
  {
    CV_Assert(normals->size() == pointCloud.size());
  }
}

void readPointCloud(const std::string &filename, std::vector<cv::Point3f> &pointCloud, std::vector<cv::Point3i> &colors, std::vector<cv::Point3f> &normals)
{
  pointCloud.clear();
  colors.clear();
  normals.clear();

  std::ifstream file(filename.c_str());
  CV_Assert(file.is_open());

  const int extSz = 3;
  string ext = filename.substr(filename.size() - extSz, extSz);
  CV_Assert(ext == "ply");

  bool isElementSet = false;
  int propertyCount = 0;
  bool arePropertiesCounted = false;
  while(!file.eof())
  {
    const int sz = 1024;
    char line_c[sz];
    file.getline(line_c, sz);
    string line = line_c;
//    cout << line << endl;
    if (!isElementSet)
    {
      if (line.find("element") != string::npos)
      {
        isElementSet = true;
      }
    }
    else
    {
      if (!arePropertiesCounted)
      {
        if (line.find("property") != string::npos)
        {
          ++propertyCount;
        }
        else
        {
          arePropertiesCounted = true;
        }
      }
    }

    int res = strcmp("end_header", line_c);

    if(res == 0)
      break;
  }

  const int pointCloutPropertiesCount = 3;
  const int allPropertiesCount = 9;

  CV_Assert(propertyCount == pointCloutPropertiesCount || propertyCount == allPropertiesCount);

  while(!file.eof())
  {
    Point3f pt;
    file >> pt.x >> pt.y;
    if(file.eof())
      break;
    file >> pt.z;
    pointCloud.push_back(pt);

    if (propertyCount == allPropertiesCount)
    {
      Point3i pti;
      file >> pti.x >> pti.y >> pti.z;
      colors.push_back(pti);

      Point3f ptf;
      file >> ptf.x >> ptf.y >> ptf.z;
      normals.push_back(ptf);
    }
  }

  if (propertyCount == allPropertiesCount)
  {
    CV_Assert(pointCloud.size() == colors.size());
    CV_Assert(pointCloud.size() == normals.size());
  }
}

void project3dPoints(const std::vector<cv::Point3f>& points, const PoseRT &pose, std::vector<cv::Point3f>& modif_points)
{
  project3dPoints(points, pose.getRvec(), pose.getTvec(), modif_points);
}

void project3dPoints(const vector<Point3f>& points, const Mat& rvec, const Mat& tvec, vector<Point3f>& modif_points)
{
  modif_points.clear();
  modif_points.resize(points.size());
  Mat R(3, 3, CV_64FC1);
  Rodrigues(rvec, R);
  for (size_t i = 0; i < points.size(); i++)
  {
    modif_points[i].x = R.at<double> (0, 0) * points[i].x + R.at<double> (0, 1) * points[i].y + R.at<double> (0, 2)
        * points[i].z + tvec.at<double> (0, 0);
    modif_points[i].y = R.at<double> (1, 0) * points[i].x + R.at<double> (1, 1) * points[i].y + R.at<double> (1, 2)
        * points[i].z + tvec.at<double> (1, 0);
    modif_points[i].z = R.at<double> (2, 0) * points[i].x + R.at<double> (2, 1) * points[i].y + R.at<double> (2, 2)
        * points[i].z + tvec.at<double> (2, 0);
  }
}

void saveToCache(const std::string &name, const cv::Mat &mat)
{
  FileStorage fs(name + ".xml", FileStorage::WRITE);
  fs << name << mat;
  fs.release();
}

cv::Mat getFromCache(const std::string &name)
{
  Mat result;
  /*
  FileStorage fs(name + ".xml", FileStorage::READ);
  if (fs.isOpened())
  {
    fs[name] >> result;
    fs.release();
  }
  */

  return result;
}

cv::Mat getInvalidDepthMask(const cv::Mat &depthMat, const cv::Mat &registrationMask)
{
  Mat invalidDepthMask;
  if (depthMat.type() == CV_32FC1 || depthMat.type() == CV_64FC1)
  {
    invalidDepthMask = (depthMat != depthMat);
  }
  else
  {
    invalidDepthMask = (depthMat == 0);
  }

  CV_Assert(!registrationMask.empty());
  CV_Assert(registrationMask.size() == depthMat.size());
  CV_Assert(registrationMask.type() == CV_8UC1);
  invalidDepthMask.setTo(0, registrationMask);
  return invalidDepthMask;
}

void computeOrientations(const cv::Mat &edges, cv::Mat &orientationsImage)
{
//  imshow("input edges", edges);
//  waitKey();

  //TODO: move up
  int testM = 5;


  Mat edgesImage = edges.clone();
  //otherwise chamfer matching doesn't work
  edgesImage.row(0).setTo(Scalar(0));
  edgesImage.row(edgesImage.rows - 1).setTo(Scalar(0));
  edgesImage.col(0).setTo(Scalar(0));
  edgesImage.col(edgesImage.cols - 1).setTo(Scalar(0));

//  Mat edgesWideImage(edgesImage.rows + 2 * margin, edgesImage.cols + 2*margin, edgesImage.type(), Scalar(0));
//  Rect roiRect(margin, margin, edgesImage.cols, edgesImage.rows);
//  Mat roi = edgesWideImage(roiRect);
//  edgesImage.copyTo(roi);
//  imshow("narrow", edgesImage);
//  imshow("wide", edgesWideImage);
//  waitKey();

  IplImage edge_img = edgesImage;
  IplImage *dist_img = cvCreateImage(cvSize(edge_img.width, edge_img.height), IPL_DEPTH_32F, 1);
  cvSetZero(dist_img);
  IplImage *annotated_img = cvCreateImage(cvSize(edge_img.width, edge_img.height), IPL_DEPTH_32S, 2);
  cvSetZero(annotated_img);

  IplImage *orientation_img = cvCreateImage(cvSize(edge_img.width, edge_img.height), IPL_DEPTH_32F, 1);
  cvSetZero(orientation_img);
  IplImage* edge_clone = cvCloneImage(&edge_img);
  computeEdgeOrientations(edge_clone, orientation_img, testM);
  cvReleaseImage(&edge_clone);

  Mat orientation_img_mat(orientation_img);
  CV_Assert(orientation_img_mat.type() == CV_32FC1);
  for (int i = 0; i < orientation_img_mat.rows; ++i)
  {
    for (int j = 0; j < orientation_img_mat.cols; ++j)
    {
      if (cvIsNaN(orientation_img_mat.at<float>(i, j)))
      {
        edgesImage.at<uchar>(i, j) = 0;
      }
    }
  }
//  imshow("edgesImage", edgesImage);
//  waitKey(200);

  //this param is not used because we don't use computed dist_img
  const float dtTruncation = -1;
  ::computeDistanceTransform(&edge_img, dist_img, annotated_img, dtTruncation);

  fillNonContourOrientations(annotated_img, orientation_img);

  orientationsImage = Mat(orientation_img).clone();

#ifdef OCM_VISUALIZE
  Mat orView = min(orientationImage, CV_PI - orientationImage);

  Mat view;
  orView.convertTo(view, CV_8UC1, 235. / (CV_PI / 2.0), 20);
  view.setTo(Scalar(0), ~edgesImage);
  imshow("orientation", view);
  waitKey();
  cout << "Orientation: " << orientationImage(Rect(0, 0, 10, 10)) << endl;
  cout << "Edges: " << edgesImage(Rect(0, 0, 10, 10)) << endl;

  Mat nonNan = Mat(orientationImage.size(), CV_8UC1, Scalar(255));
  for(int i=0; i<orientationImage.rows; i++)
  {
    for(int j=0; j<orientationImage.cols; j++)
    {
      if(isnan(orientationImage.at<float>(i, j)))
      {
        nonNan.at<uchar>(i, j) = 0;
      }
    }
  }

  imshow("orientation non-nan", nonNan);
  waitKey();
#endif

/*
  double minOrientation, maxOrientation;
  minMaxLoc(orientationsImage, &minOrientation, &maxOrientation);
  cout << "Orientations: " << minOrientation << " " << maxOrientation << endl;
*/

  cvReleaseImage(&annotated_img);
  cvReleaseImage(&dist_img);
  cvReleaseImage(&orientation_img);
}

struct Imshow3dData
{
  cv::Mat image3d;
  std::string windowName;
  int position;
};

void onTrackbarChange(int position, void *rawData)
{
  Imshow3dData *data = static_cast<Imshow3dData*>(rawData);

  Mat image3d = data->image3d;
  CV_Assert(image3d.isContinuous());
  CV_Assert(image3d.dims == 3);
  CV_Assert(position >= 0 && position < image3d.size.p[0]);

  //TODO: support different types
  void *slice = 0;
  switch(image3d.type())
  {
    case CV_8UC3:
      slice = image3d.ptr<Vec3b>(position, 0, 0);
      break;
    case CV_8UC1:
      slice = image3d.ptr<uchar>(position, 0, 0);
      break;
    default:
      CV_Assert(false);
  }
  Mat image2d(image3d.size.p[1], image3d.size.p[2], image3d.type(), slice);

  imshow(data->windowName, image2d);
}

void imshow3d(const std::string &windowName, const cv::Mat &image3d)
{
  namedWindow(windowName); //QT backend crashes when destroying non-existent window
  destroyWindow(windowName);

  CV_Assert(image3d.dims == 3);
  namedWindow(windowName, CV_WINDOW_NORMAL);
  int count = image3d.size.p[0] - 1;

  //TODO: clear data when closing a window
  static std::map<std::string, Imshow3dData> allWindows;
  Imshow3dData &data = allWindows[windowName];
  data.position = 0;
  data.image3d = image3d;
  data.windowName = windowName;

  createTrackbar("z", windowName, &data.position, count, onTrackbarChange, &data);
  onTrackbarChange(data.position, &data);
}

void cvtColor3d(const cv::Mat &src, cv::Mat &dst, int code)
{
  CV_Assert(src.dims == 3);
  Mat src_vector(1, src.total(), src.type(), src.data);
  Mat dst_vector;
  cvtColor(src_vector, dst_vector, code);
  //TODO: eliminate copy
  Mat(src.dims, src.size.p, dst_vector.type(), dst_vector.data).copyTo(dst);
}

struct ManualContourMarkingData
{
  bool isLButtonPressed;
  std::vector<cv::Point> *contour;
  cv::Mat displayedImage;
  std::string windowName;
};

static void onMouse(int event, int x, int y, int, void *srcData)
{
  ManualContourMarkingData *data = static_cast<ManualContourMarkingData*>(srcData);
  if (event == CV_EVENT_LBUTTONUP)
  {
    data->isLButtonPressed = false;
  }

  if (event == CV_EVENT_LBUTTONDOWN)
  {
    data->isLButtonPressed = true;
  }

  if (!data->isLButtonPressed)
  {
    return;
  }

  Point pt(x, y);
  data->contour->push_back(pt);
  circle(data->displayedImage, pt, 1, Scalar(255, 0, 0), -1);
  imshow(data->windowName, data->displayedImage);
}

void markContourByUser(const cv::Mat &image, std::vector<cv::Point> &contour,
                       const std::string &windowName)
{
  const char resetKey = 'r';
  contour.clear();

  ManualContourMarkingData data;
  data.contour = &contour;
  //TODO: what if button is pressed already?
  data.isLButtonPressed = false;
  data.displayedImage = image.clone();
  data.windowName = windowName;

  namedWindow(data.windowName, WINDOW_NORMAL);
  setMouseCallback(data.windowName, onMouse, &data);
  imshow(data.windowName, data.displayedImage);
  int key = waitKey();
  while (key == resetKey)
  {
    data.displayedImage = image.clone();
    data.contour->clear();
    key = waitKey();
  }
  destroyWindow(data.windowName);

  CV_Assert(!contour.empty());
}

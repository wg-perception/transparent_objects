#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>

#include "edges_pose_refiner/glassSegmentator.hpp"
#include "edges_pose_refiner/utils.hpp"

using namespace cv;
using std::cout;
using std::endl;

//#define VISUALIZE

//#define VISUALIZE_TABLE


void showGrabCutResults(const Mat &bgrImage, const Mat &mask, const string &title = "grabCut");


void createMasksForGrabCut(const cv::Mat &objectMask,
                           std::vector<cv::Rect> &allRois, std::vector<cv::Mat> &allRoiMasks,
                           const GlassSegmentatorParams &params)
{
  Mat objectMaskEroded;
  erode(objectMask, objectMaskEroded, Mat(), Point(-1, -1), params.grabCutErosionsIterations);
  Mat objectMaskDilated;
  dilate(objectMask, objectMaskDilated, Mat(), Point(-1, -1), params.grabCutDilationsIterations);

  Mat tmpObjectMask = objectMask.clone();
  vector<vector<Point> > contours;
  findContours(tmpObjectMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  allRois.clear();
  allRoiMasks.clear();
  for(size_t i = 0; i < contours.size(); ++i)
  {
    //TODO: move up
    const float minArea = 40.0f;
    if (contourArea(contours[i]) < minArea)
    {
      continue;
    }

    Rect roi = boundingRect(Mat(contours[i]));
    roi.x = std::max(0, roi.x - params.grabCutMargin);
    roi.y = std::max(0, roi.y - params.grabCutMargin);
    roi.width = std::min(objectMask.cols - roi.x, roi.width + 2*params.grabCutMargin);
    roi.height = std::min(objectMask.rows - roi.y, roi.height + 2*params.grabCutMargin);

    Mat currentMask(objectMask.size(), CV_8UC1, GC_BGD);
    currentMask(roi).setTo(GC_PR_BGD, objectMaskDilated(roi));
    currentMask(roi).setTo(GC_PR_FGD, objectMask(roi));
    currentMask(roi).setTo(GC_FGD, objectMaskEroded(roi));

    Mat roiMask = currentMask(roi);

    allRois.push_back(roi);
    allRoiMasks.push_back(roiMask);
  }
  CV_Assert(allRois.size() == allRoiMasks.size());
}

void refineSegmentationByGrabCut(const Mat &bgrImage, const Mat &rawMask, Mat &refinedMask, const GlassSegmentatorParams &params)
{
  CV_Assert(!rawMask.empty());
#ifdef VISUALIZE
  imshow("before grabcut", rawMask);
#endif

  refinedMask = Mat(rawMask.size(), CV_8UC1, Scalar(GC_BGD));

  vector<Rect> allRois;
  vector<Mat> allRoiMasks;
  createMasksForGrabCut(rawMask, allRois, allRoiMasks, params);

  for(size_t i = 0; i < allRois.size(); ++i)
  {
    Rect roi = allRois[i];
    Mat roiMask = allRoiMasks[i];
#ifdef VISUALIZE
    showGrabCutResults(bgrImage(roi), roiMask, "initial mask for GrabCut segmentation");
#endif

    Mat bgdModel, fgdModel;
    grabCut(bgrImage(roi), roiMask, Rect(), bgdModel, fgdModel, params.grabCutIterations, GC_INIT_WITH_MASK);

    Mat refinedMaskRoi = refinedMask(roi);
    roiMask.copyTo(refinedMaskRoi, (roiMask != GC_BGD) & (roiMask != GC_PR_BGD));

#ifdef VISUALIZE
    showGrabCutResults(bgrImage(roi), roiMask, "grab cut segmentation");
    waitKey();
#endif
  }

  Mat prFgd = (refinedMask == GC_PR_FGD);
  Mat fgd = (refinedMask == GC_FGD);
  refinedMask = prFgd | fgd;
#ifdef VISUALIZE
  showSegmentation(bgrImage, refinedMask, "final GrabCut segmentation");
  imshow("final GrabCut mask", refinedMask);
#endif
}

void snakeImage(const Mat &image, vector<Point> &points)
{
  float alpha = 10.0f;
  float beta = 30.0f;
//  float gamma = 10000.0f;
  float gamma = 10.0f;
  const CvSize searchWindow = cvSize(15, 15);

/*
  vector<CvPoint> cvPoints(points.size());
  for(size_t i = 0; i < points.size(); ++i)
  {
    cvPoints[i] = points[i];
  }
*/

  vector<CvPoint> cvPoints;
  for(size_t i = 0; i < points.size(); ++i)
  {
    if(i % 2 == 0)
      cvPoints.push_back(points[i]);
  }


  Mat grayImage;
  if(image.channels() == 3)
  {
    cvtColor(image, grayImage, CV_BGR2GRAY);
  }
  else
  {
    grayImage = image;
  }

  IplImage imageForSnake = grayImage;
  cvSnakeImage(&imageForSnake, cvPoints.data(), cvPoints.size(), &alpha, &beta, &gamma, CV_VALUE, searchWindow, cvTermCriteria(CV_TERMCRIT_ITER, 1, 0.0), 1);


  for (size_t i = 0; i < points.size(); ++i)
  {
    points[i] = cvPoints[i / 2];
  }
}

void refineSegmentationBySnake(const Mat &bgrImage, const Mat &rawMask, Mat &refinedMask)
{
  refinedMask = Mat(rawMask.size(), CV_8UC1, Scalar(0));

  Mat tmpRawMask = rawMask.clone();
  vector<vector<Point> > contours;
  findContours(tmpRawMask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  for(size_t i = 0; i < contours.size(); ++i)
  {
    for(int j=0; j<10000; ++j)
    {
      Mat drawImage = bgrImage.clone();
      drawContours(drawImage, contours, i, Scalar(0, 255, 0), 1);
      imshow("snake", drawImage);
      waitKey();

      snakeImage(bgrImage, contours[i]);
    }
  }
}


void showGrabCutResults(const Mat &bgrImage, const Mat &mask, const string &title)
{
  Mat result(mask.size(), CV_8UC3, Scalar::all(0));
  result.setTo(Scalar(255, 0, 0), mask == GC_BGD);
  result.setTo(Scalar(128, 0, 0), mask == GC_PR_BGD);
  result.setTo(Scalar(0, 0, 255), mask == GC_FGD);
  result.setTo(Scalar(0, 0, 128), mask == GC_PR_FGD);

  imshow(title, result);

  //TODO: move up
  Mat mergedMask = 0.7 * bgrImage + 0.3 * result;
  imshow(title + " merged", mergedMask);
}

void showSegmentation(const Mat &image, const Mat &mask, const string &title)
{
  Mat drawImage = drawSegmentation(image, mask);
  imshow(title, drawImage);
}

GlassSegmentator::GlassSegmentator(const GlassSegmentatorParams &_params)
{
  params = _params;
}

void refineGlassMaskByTableHull(const std::vector<cv::Point2f> &tableHull, cv::Mat &glassMask)
{
#ifdef VISUALIZE_TABLE
  Mat visualizedGlassMask;
  cvtColor(glassMask, visualizedGlassMask, CV_GRAY2BGR);
  for (size_t i = 0; i < projectedHull.size(); ++i)
  {
    circle(visualizedGlassMask, projectedHull[i], 2, Scalar(0, 0, 255), -1);
    line(visualizedGlassMask, projectedHull[i], projectedHull[(i + 1) % projectedHull.size()], Scalar(255, 0, 0));
  }
  imshow("table hull", visualizedGlassMask);
#endif

  vector<vector<Point> > contours;
  Mat copyGlassMask = glassMask.clone();
  findContours(copyGlassMask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  for (size_t i = 0; i < contours.size(); ++i)
  {
    Moments moms = moments(contours[i]);
    Point2f centroid(moms.m10 / moms.m00, moms.m01 / moms.m00);
    if (pointPolygonTest(tableHull, centroid, false) < 0)
    {
      drawContours(glassMask, contours, i, Scalar(0, 0, 0), -1);
    }
  }
}

void GlassSegmentator::segment(const cv::Mat &bgrImage, const cv::Mat &depthMat, const cv::Mat &registrationMask, int &numberOfComponents,
                               cv::Mat &glassMask, const std::vector<cv::Point2f> *tableHull)
{
  Mat srcMask = getInvalidDepthMask(depthMat, registrationMask);
#ifdef VISUALIZE
  imshow("mask without registration errors", srcMask);
#endif

  if (tableHull != 0)
  {
    refineGlassMaskByTableHull(*tableHull, srcMask);
  }

#ifdef VISUALIZE
  imshow("mask with table", srcMask);
#endif

  Mat mask = srcMask.clone();
  morphologyEx(mask, mask, MORPH_CLOSE, Mat(), Point(-1, -1), params.closingIterations);
#ifdef VISUALIZE
  imshow("mask after closing", mask);
#endif

  morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), params.openingIterations);
#ifdef VISUALIZE
  imshow("mask after openning", mask);
#endif
  vector<vector<Point> > contours;
  findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  numberOfComponents = static_cast<int>(contours.size());

  Mat glassImage(mask.size(), CV_8UC1, Scalar(0));
  drawContours(glassImage, contours, -1, Scalar(255), -1);

  int elementSize = params.finalClosingIterations * 2 + 1;
  Mat structuringElement = getStructuringElement(MORPH_ELLIPSE, Size(elementSize, elementSize), Point(params.finalClosingIterations, params.finalClosingIterations));
  morphologyEx(srcMask, srcMask, MORPH_CLOSE, structuringElement, Point(params.finalClosingIterations, params.finalClosingIterations));

//  morphologyEx(srcMask, srcMask, MORPH_CLOSE, Mat(), Point(-1, -1), params.finalClosingIterations);
#ifdef VISUALIZE
  imshow("final closing", srcMask);
#endif
  uchar foundComponentColor = 128;

  for(int i = 0; i < glassImage.rows; ++i)
  {
    for(int j = 0; j < glassImage.cols; ++j)
    {
      if(glassImage.at<uchar>(i, j) == 255 && srcMask.at<uchar>(i, j) == 255)
      {
        floodFill(srcMask, Point(j, i), Scalar(foundComponentColor));
      }
    }
  }

  glassMask = (srcMask == foundComponentColor);

#ifdef VISUALIZE
  imshow("before convex", glassMask);
#endif

  if (params.fillConvex)
  {

    Mat tmpGlassMask = glassMask.clone();
    vector<vector<Point> > srcContours;
    findContours(tmpGlassMask, srcContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    for(size_t i = 0; i < srcContours.size(); ++i)
    {
      vector<Point> hull;
      //convexHull(Mat(srcContours[i]), hull, false, true);
      convexHull(Mat(srcContours[i]), hull);
      fillConvexPoly(glassMask, hull.data(), hull.size(), Scalar(255));
    }
  }


  if (params.useGrabCut)
  {
    Mat refinedGlassMask;
    refineSegmentationByGrabCut(bgrImage, glassMask, refinedGlassMask, params);
//    Mat prFgd = (refinedGlassMask == GC_PR_FGD);
//    Mat fgd = (refinedGlassMask == GC_FGD);
//    glassMask = prFgd | fgd;
    glassMask = refinedGlassMask;
  }

#ifdef VISUALIZE
  showSegmentation(bgrImage, glassMask, "grabCut segmentation");
  waitKey();
#endif
}

void segmentGlassManually(const cv::Mat &image, cv::Mat &glassMask)
{
  vector<vector<Point> > contours(1);
  markContourByUser(image, contours[0], "manual glass segmentation");

  glassMask = Mat(image.size(), CV_8UC1, Scalar(0));
  drawContours(glassMask, contours, -1, Scalar::all(255), -1);
}

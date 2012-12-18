/*
 * reconstructRotationalModel.cpp
 *
 *  Created on: 12/16/2012
 *      Author: ilysenkov
 */

#include <opencv2/opencv.hpp>
//#include <opencv2/rgbd/rgbd.hpp>
#include "edges_pose_refiner/utils.hpp"

using namespace cv;
using std::cout;
using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
    const string profileMaskFilename = "image_profile.png.user_mask.png";
    const string topMaskFilename = "image_top.png.user_mask.png";
    const string topDepthFilename = "depth_top.xml.gz";

    CV_Assert(argc == 3);
    const string basePath = argv[1];
    const string objectName = argv[2];

    Mat profileMask = imread(basePath + "/" + objectName + "/" + profileMaskFilename, CV_LOAD_IMAGE_GRAYSCALE);
    CV_Assert(!profileMask.empty());
    CV_Assert(profileMask.type() == CV_8UC1);

    Mat drawProfileMask;
    cvtColor(profileMask, drawProfileMask, CV_GRAY2BGR);

    vector<int> widths;
    for (int i = 0; i < profileMask.rows; ++i)
    {
        int xStart = -1;
        int xEnd = -1;
        for (int j = 0; j < profileMask.cols; ++j)
        {
            if (xStart < 0 && profileMask.at<uchar>(i, j))
            {
                xStart = j;
            }

            if (xStart > 0 && xEnd < 0 && !profileMask.at<uchar>(i, j))
            {
                xEnd = j;
            }
        }
        CV_Assert(xStart < 0 && xEnd < 0 || xStart > 0 && xEnd > 0);
        if (xStart > 0 && xEnd > 0)
        {
            widths.push_back(xEnd - xStart);
        }
    }
    std::reverse(widths.begin(), widths.end());

    Mat topMask = imread(basePath + "/" + objectName + "/" + topMaskFilename, CV_LOAD_IMAGE_GRAYSCALE);
    CV_Assert(!topMask.empty());
    string topDepthFullFilename = basePath + "/" + objectName + "/" + topDepthFilename;
    Mat depth;
    FileStorage fs(topDepthFullFilename, FileStorage::READ);
    CV_Assert(fs.isOpened());
    fs["depth_image"] >> depth;
    fs.release();
    CV_Assert(!depth.empty());

    Mat validDepthMask = (depth == depth);

    Mat dilatedTopMask;
    dilate(topMask, dilatedTopMask, Mat(), Point(-1, -1), 10);
    Mat nearTableMask = dilatedTopMask & (~topMask) & validDepthMask;

    double meanDepth = mean(depth, nearTableMask)[0];
    //TODO: fix
    //meanDepth = meanDepth - 0.15;
    cout << "mean depth: " << meanDepth << endl;

    Moments moms = moments(topMask, true);
    double area = moms.m00;
    double radiusPixels = sqrt(area / CV_PI);
    Mat K = (cv::Mat_<double>(3, 3) << 525.0,   0.0, 319.5,
                                         0.0, 525.0, 239.5,
                                         0.0,   0.0,   1.0);

    Mat origin = (cv::Mat_<double>(3, 1) << 319.5, 239.5, 1.0);
    Mat shifted = (cv::Mat_<double>(3, 1) << 319.5 + radiusPixels, 239.5, 1.0);
    Mat originRay = K.inv() * origin;
    Mat shiftedRay = K.inv() * shifted;

    originRay *= (meanDepth / originRay.at<double>(2));
    shiftedRay *= (meanDepth / shiftedRay.at<double>(2));
    cout << originRay << endl;
    cout << shiftedRay << endl;

    double radiusMeters = shiftedRay.at<double>(0) - originRay.at<double>(0);
    cout << radiusMeters << endl;


    int max_width = *std::max_element(widths.begin(), widths.end());
    vector<double> allRadiuses;
    for (size_t i = 0; i < widths.size(); ++i)
    {
        allRadiuses.push_back((widths[i] * radiusMeters) / max_width);
    }

    vector<Point3f> model;
    for (double x = -allRadiuses[0]; x <= allRadiuses[0]; x += 0.0005)
    {
      for (double y = -allRadiuses[0]; y <= allRadiuses[0]; y += 0.0005)
      {
        if (x*x + y*y <= allRadiuses[0] * allRadiuses[0])
        {
          model.push_back(Point3f(x, y, 0.0f));
        }
      }
    }

    for (size_t i = 1; i < widths.size(); ++i)
    {
        cout << allRadiuses[i] << endl;
        for (double phi = 0.0; phi < 2 * CV_PI; phi += CV_PI / 100)
        {
          double x = allRadiuses[i] * cos(phi);
          double y = allRadiuses[i] * sin(phi);

          model.push_back(Point3f(x, y, (i * 2.0 * radiusMeters) / max_width));
        }
    }
    cout << model.size() << endl;
    writePointCloud("model.asc", model);

    return 0;
}


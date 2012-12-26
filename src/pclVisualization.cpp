/*
 * pclVisualization.cpp
 *
 *  Created on: 12/24/2012
 *      Author: ilysenkov
 */

#include "edges_pose_refiner/pclVisualization.hpp"
#include "edges_pose_refiner/pcl.hpp"

#ifdef USE_3D_VISUALIZATION
#include <boost/thread/thread.hpp>
#endif

using namespace cv;
using std::cout;
using std::endl;

#ifdef USE_3D_VISUALIZATION
void publishPoints(const std::vector<cv::Point3f>& points, const boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer, cv::Scalar color, const std::string &title, const PoseRT &pose)
{
  vector<Point3f> rotatedPoints;
  project3dPoints(points, pose.getRvec(), pose.getTvec(), rotatedPoints);
  pcl::PointCloud<pcl::PointXYZ> pclPoints;
  cv2pcl(rotatedPoints, pclPoints);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pointsColor(pclPoints.makeShared(), color[2], color[1], color[0]);
  viewer->addPointCloud<pcl::PointXYZ>(pclPoints.makeShared(), pointsColor, title);
}
#endif

void publishPoints(const std::vector<cv::Point3f>& points, cv::Scalar color, const std::string &id, const PoseRT &pose)
{
#ifdef USE_3D_VISUALIZATION
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("id"));
  publishPoints(points, viewer, color, id, pose);

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
#else
  CV_Assert(false);
#endif
}

void publishPoints(const std::vector<std::vector<cv::Point3f> >& points)
{
#ifdef USE_3D_VISUALIZATION
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("id"));

//  const int minVal = 128;
//  const int maxVal = 255;
//  const int colorDim = 3;
  for (size_t i = 0; i < points.size(); i++)
  {
    cout << "size: " << points[i].size() << endl;
    Scalar color;
    switch (i)
    {
      case 0:
        color = cv::Scalar(0, 0, 255);
        break;
      case 1:
        color = cv::Scalar(0, 255, 0);
        break;
      case 2:
        color = cv::Scalar(255, 0, 255);
        break;
      case 3:
        color = cv::Scalar(255, 0, 0);
        break;
      default:
        color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
    }
/*
    for (int j = 0; j < colorDim; j++)
    {
      color[j] = minVal + rand() % (maxVal - minVal + 1);
    }
*/

    std::stringstream str;
    str << i;
    publishPoints(points[i], viewer, color, str.str());
  }

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
#else
  CV_Assert(false);
#endif
}



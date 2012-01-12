/*
 * registerPointClouds.cpp
 *
 *  Created on: Mar 15, 2011
 *      Author: ilysenkov
 */

#include "edges_pose_refiner/registerPointClouds.hpp"
#include <pcl/registration/icp_nl.h>
#include <opencv2/contrib/contrib.hpp>
#include "edges_pose_refiner/utils.hpp"

#define VERBOSE
//#define DOWNSAMPLE

using std::vector;
using std::cout;
using std::endl;

MultiViewRegistrator::MultiViewRegistrator(const MultiViewRegistratorParams &_params)
{
  params = _params;
}

//void MultiViewRegistrator::getPCLPointClouds(Object *object, vector<pcl::PointCloud<pcl::PointXYZ> > &pointClouds)
//{
//  pointClouds.resize(object->imagesNum);
//
//  vector<vector<cv::Point3f> > cvPointClouds;
//  getPointClouds(object, cvPointClouds);
//  for(size_t i=0; i<cvPointClouds.size(); i++)
//  {
//    cv2pcl(cvPointClouds[i], pointClouds[i]);
//  }
//}

void MultiViewRegistrator::setParams(const MultiViewRegistratorParams &_params)
{
  params = _params;
}

/*
void publishPose(const ros::Publisher &points_pub, TrainingSet &tr, const pcl::PointCloud<pcl::PointXYZ> &targetCloud, const pcl::PointCloud<pcl::PointXYZ> &beforeCloud, const pcl::PointCloud<pcl::PointXYZ> &afterCloud)
{
  vector<cv::Point3f> resultCloud;
  vector<int> indices;
  for(size_t i=0; i<targetCloud.points.size(); i++)
  {
    const pcl::PointXYZ &pt = targetCloud.points[i];
    resultCloud.push_back(cv::Point3f(pt.x, pt.y, pt.z));
    indices.push_back(0);
  }

  for(size_t i=0; i<beforeCloud.points.size(); i++)
  {
    const pcl::PointXYZ &pt = beforeCloud.points[i];
    resultCloud.push_back(cv::Point3f(pt.x, pt.y, pt.z));
    indices.push_back(1);
  }

  for(size_t i=0; i<afterCloud.points.size(); i++)
  {
    const pcl::PointXYZ &pt = afterCloud.points[i];
    resultCloud.push_back(cv::Point3f(pt.x, pt.y, pt.z));
    indices.push_back(2);
  }


  tr.publishPoints(resultCloud, points_pub, indices);
}
*/

float getNearestNeighborDistance(cv::flann::Index &flannIndex, const pcl::PointXYZ &pt, const MultiViewRegistratorParams &params)
{
  vector<float> query;
  query.push_back(pt.x);
  query.push_back(pt.y);
  query.push_back(pt.z);

  int knn = 1;
  vector<int> indices(knn);
  vector<float> dists(knn);
  flannIndex.knnSearch(query, indices, dists, knn, cv::flann::SearchParams(params.flannSearchesCount));
  return dists[0];
}

void MultiViewRegistrator::filterPointCloudByNearestNeighborDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr &targetPtr, const pcl::PointCloud<pcl::PointXYZ> &inputCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr &filteredCloudPtr) const
{
  vector<cv::Point3f> cvTargetCloud;
  pcl2cv(*targetPtr, cvTargetCloud);
  cv::flann::KDTreeIndexParams indexParams;
  cv::flann::Index flannIndex(cv::Mat(cvTargetCloud).reshape(1), indexParams);

  vector<float> allDists;
  for(size_t i=0; i<inputCloud.points.size(); i++)
  {
    allDists.push_back(getNearestNeighborDistance(flannIndex, inputCloud.points[i], params));
  }
  vector<float> sortedDists = allDists;
  std::sort(sortedDists.begin(), sortedDists.end());

  int maxInlierIdx = params.inliersRatio * sortedDists.size();
  float maxInlierDist = sortedDists.at(maxInlierIdx);
  float distThreshold = params.maxInlierDistFactor * maxInlierDist;

  for(size_t i=0; i<inputCloud.points.size(); i++)
  {
    float dist = allDists[i];
    //float dist = getNearestNeighborDistance(flannIndex, inputCloud.points[i]);
    if(dist < distThreshold)
    {
      filteredCloudPtr->points.push_back(inputCloud.points[i]);
    }
  }
#ifdef VERBOSE
      //cout << "filtration: " << inputCloud.points.size() << " vs. " << filteredCloudPtr->points.size() << endl;
#endif
}

void MultiViewRegistrator::align(const std::vector<std::vector<cv::Point3f> > &inputPointClouds, std::vector<std::vector<cv::Point3f> > &outputPointClouds) const
{
  vector<pcl::PointCloud<pcl::PointXYZ> > inputPointCloudsPCL(inputPointClouds.size());
  for(size_t i=0; i<inputPointClouds.size(); i++)
  {
    cv2pcl(inputPointClouds[i], inputPointCloudsPCL[i]);
  }

  vector<pcl::PointCloud<pcl::PointXYZ> > outputPointCloudsPCL;
  align(inputPointCloudsPCL, outputPointCloudsPCL);
  CV_Assert(inputPointCloudsPCL.size() == outputPointCloudsPCL.size());

  outputPointClouds.resize(outputPointCloudsPCL.size());
  for(size_t i=0; i<outputPointClouds.size(); i++)
  {
    pcl2cv(outputPointCloudsPCL[i], outputPointClouds[i]);
  }
}

void MultiViewRegistrator::align(const std::vector<pcl::PointCloud<pcl::PointXYZ> > &inputPointClouds, std::vector<pcl::PointCloud<pcl::PointXYZ> > &outputPointClouds) const
{
  outputPointClouds = inputPointClouds;
  //vector<pcl::PointCloud<pcl::PointXYZ> > outputPointClouds = inputPointClouds;
  //getPCLPointClouds(object, pointClouds);

  Eigen::Matrix4f zero2result;

  pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;

  for(int round=0; round<params.roundsCount; round++)
  {
#ifdef VERBOSE
    double beginScore = 0.;
    double endScore = 0.;
#endif
    for(size_t pcIdx=0; pcIdx<outputPointClouds.size(); pcIdx++)
    {
#ifdef VERBOSE
      cv::TickMeter registeringTime;
      registeringTime.start();
#endif

      pcl::PointCloud<pcl::PointXYZ>::Ptr targetPtr(new pcl::PointCloud<pcl::PointXYZ>());
      for(size_t i=0; i<outputPointClouds.size(); i++)
      {
        if(i != pcIdx)
          (*targetPtr) += outputPointClouds[i];
      }

      pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloudPtr(new pcl::PointCloud<pcl::PointXYZ>());
      filterPointCloudByNearestNeighborDistance(targetPtr, outputPointClouds[pcIdx], inputCloudPtr);

      icp.setInputTarget(targetPtr);
      icp.setInputCloud(inputCloudPtr);
      icp.setMaximumIterations(params.maxIterations);


#ifdef VERBOSE
      double initialScore = icp.getFitnessScore();
      //cout << "initial fitnessScore: " << icp.getFitnessScore() << endl;
      //cout << "Aligning pointCloud " << pcIdx << ": " << inputCloudPtr->points.size() << " vs. " << targetPtr->points.size() << endl;
      //system("date");
#endif

      pcl::PointCloud<pcl::PointXYZ> rotatedPointCloud;
      icp.align(rotatedPointCloud);

      //publishPose(points_pub, tr, *targetPtr, pointClouds[pcIdx], rotatedPointCloud);
      //cv::namedWindow("pose has been published");

      Eigen::Matrix4f finalTransformation = icp.getFinalTransformation();
      pcl::transformPointCloud(outputPointClouds[pcIdx], rotatedPointCloud, finalTransformation );
      outputPointClouds[pcIdx] = rotatedPointCloud;

      if(pcIdx == 0)
      {
        zero2result = (round == 0) ? finalTransformation : finalTransformation * zero2result;
      }

#ifdef VERBOSE
      registeringTime.stop();
      double finalScore = icp.getFitnessScore();
      beginScore += initialScore;
      endScore += finalScore;
      //cout << "registering time: " << registeringTime.getTimeSec() << "s" << endl;
      //cout << "fitnessScore: " << icp.getFitnessScore() << endl;
#endif
    }

#ifdef VERBOSE
    cout << "Score = " << beginScore - endScore << " = " << beginScore << " - " << endScore << endl;
#endif
  }


  for(size_t pcIdx=0; pcIdx<inputPointClouds.size(); pcIdx++)
  {
    Eigen::Matrix4f inv = zero2result.inverse();
    pcl::PointCloud<pcl::PointXYZ> rotatedPointCloud;
    pcl::transformPointCloud(outputPointClouds[pcIdx], rotatedPointCloud, inv);
    outputPointClouds[pcIdx] = rotatedPointCloud;
  }




//  outputPointClouds.resize(outputPointClouds.size());
//  for(size_t i=0; i<outputPointClouds.size(); i++)
//  {
//    pcl2cv(outputPointClouds[i], outputPointClouds[i]);
//  }
}

//void getPointClouds(Object *object, std::vector<std::vector<cv::Point3f> > &pointClouds)
//{
//  const int partsCount = object->imagesNum;
//  pointClouds.clear();
//  pointClouds.resize(partsCount);
//
//  for(int imgIdx=0; imgIdx<partsCount; imgIdx++)
//  {
//    vector<cv::Point3f> srcEdgels;
//    for(size_t edgelIdx=0; edgelIdx<object->points.at(imgIdx).size(); edgelIdx++)
//    {
//#ifdef DOWNSAMPLE
//      if(rand() % 10 != 0)
//        continue;
//#endif
//
//      int pointIdx = object->edgelsIndices.at(imgIdx).at(edgelIdx);
//      const pcl::PointXYZ &pointXYZ = object->clouds.at(imgIdx).points.at( pointIdx );
//      srcEdgels.push_back( cv::Point3f( pointXYZ.x, pointXYZ.y, pointXYZ.z ) );
//    }
//
//    project3dPoints(srcEdgels, object->rvecs[imgIdx], object->tvecs[imgIdx], pointClouds[imgIdx]);
//  }
//}

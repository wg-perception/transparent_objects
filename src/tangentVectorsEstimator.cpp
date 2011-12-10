/*
 * tangentVectorsEstimator.cpp
 *
 *  Created on: Apr 5, 2011
 *      Author: ilysenkov
 */

#include <edges_pose_refiner/tangentVectorsEstimator.hpp>
#include <opencv2/flann/flann.hpp>
#include <iostream>

using namespace cv;
using std::list;
using std::cout;
using std::endl;

cv::Point3f TangentVectorEstimator::getRobustCentroid(const std::vector<cv::Point3f> &points)
{
  flann::LinearIndexParams flannParams;
  flann::Index flannIndex(Mat(points).reshape(1), flannParams);
  int knn = cvRound(params.robustCentroidH * points.size());

  vector<float> scatters;
  scatters.reserve(points.size());

  for(size_t i=0; i<points.size(); i++)
  {
    Mat query = Mat(points[i]).reshape(1, 1);
    Mat dists(1, knn, CV_32FC1);
    Mat indices(1, knn, CV_32SC1);
    flannIndex.knnSearch(query, indices, dists, knn, flann::SearchParams());
    float scatter = sum(dists)[0];
    scatters.push_back(scatter);
  }

  float minScatter = std::numeric_limits<float>::max();
  int minScatterIdx = -1;
  for(size_t i=0; i<scatters.size(); i++)
  {
    if(scatters[i] < minScatter)
    {
      minScatter = scatters[i];
      minScatterIdx = i;
    }
  }


  Point3f estimate = points[minScatterIdx];

  //run C-steps:
  const size_t cStepsCount = 10;
  for(size_t i=0; i<cStepsCount; i++)
  {
    Mat query = Mat(estimate).reshape(1, 1);
    Mat dists(1, knn, CV_32FC1);
    Mat indices(1, knn, CV_32SC1);
    flannIndex.knnSearch(query, indices, dists, knn, flann::SearchParams());

    Point3d meanPoint(0.0, 0.0, 0.0);
    CV_Assert(indices.type() == CV_32SC1);
    for(int col=0; col<indices.cols; col++)
    {
      Point3d pt = points.at(indices.at<int>(0, col));
      meanPoint += pt;
    }
    meanPoint *= 1.0 / indices.cols;
    estimate = meanPoint;
  }

  return estimate;
}

void TangentVectorEstimator::estimateFinalOrientations(const std::vector<cv::Point3f> &contour, std::vector<cv::Point3f> &orientations)
{
  orientations.clear();
  orientations.reserve(contour.size());

  for(int pointIdx=0; pointIdx<static_cast<int>(contour.size()); pointIdx++)
  {
    if(pointIdx < params.minHalfContourPoints || static_cast<int>(contour.size()) - pointIdx - 1 < params.minHalfContourPoints)
    {
      orientations.push_back(params.nanOrientation);
      continue;
    }
    int firstIdx = std::max(0, pointIdx - params.maxHalfContourPoints);
    int lastIdx = std::min(static_cast<int>(contour.size()) - 1, pointIdx + params.maxHalfContourPoints);

    Point3f currentPoint = contour.at(pointIdx);
    vector<Point3f> estimations;
    for(int i=firstIdx; i<=lastIdx; i++)
    {
      if(i == pointIdx)
        continue;

      Point3f vec = currentPoint - contour.at(i);
      vec *= 1.0 / norm(vec);
      estimations.push_back(vec);
    }
    Point3d orientation = getRobustCentroid(estimations);
    Point3f result = orientation * (1.0 / norm(orientation));
    orientations.push_back(result);
  }

  CV_Assert(contour.size() == orientations.size());
}

cv::Point3f TangentVectorEstimator::estimateOrientation(const std::vector<cv::Point3f> &pointCloud, const std::vector<int> &contour)
{
  vector<Point3f> points;
  points.reserve(contour.size());
  int startIdx = static_cast<int>(contour.size()) - params.maxEstimationPointsCount - 1;

  int idx = 0;
  vector<int>::const_iterator it=contour.begin();
  while(idx < startIdx)
  {
    idx++;
    it++;
  }

  Point3f currentPoint = pointCloud.at(contour.back());
  for(; it!=contour.end(); it++)
  {
    Point3f vec = currentPoint - pointCloud[*it];
    vec *= 1.0 / norm(vec);
    points.push_back(vec);
  }

  //remove currentPoint
  points.pop_back();

  Point3d orientation = getRobustCentroid(points);
  Point3f result = orientation * (1.0 / norm(orientation));
  return result;
}



TangentVectorEstimator::TangentVectorEstimator(const TangentVectorEstimatorParams &_params)
{
  params = _params;
}

bool isPointProcessed(const std::vector<bool> &isTangentVectorEstimated, const std::vector<int> &visitedPoints, int pointIdx)
{
  vector<int>::const_iterator it = std::find(visitedPoints.begin(), visitedPoints.end(), pointIdx);
  return (isTangentVectorEstimated.at(pointIdx)) || (it != visitedPoints.end());
}

/*
class ContoursSorter
{
public:
  ContoursSorter(const vector<vector<int> > *_contours)
  {
    contours = _contours;
  }

  bool operator()(size_t a, size_t b)
  {
    return contours->at(a).size() > contours->at(b).size();
  }

private:
  const vector<vector<int> > *contours;
};
*/

void TangentVectorEstimator::constructKNNGraph(const std::vector<cv::Point3f> &pointCloud, Graph &graph, float &distanceQuantile)
{
  //flann::KDTreeIndexParams indexParams;
  flann::LinearIndexParams indexParams;
  flann::Index flannIndex(Mat(pointCloud).reshape(1), indexParams);

  graph.nearestNeighborsOut.resize(pointCloud.size());
  graph.nearestNeighborsIn.resize(pointCloud.size());
  vector<float> allDists;
  allDists.reserve(params.knn * pointCloud.size());
  for(size_t i=0; i<pointCloud.size(); i++)
  {
    vector<float> query = Mat(pointCloud[i]);
    vector<int> indices(params.knn);
    vector<float> dists(params.knn);

    flannIndex.knnSearch(query, indices, dists, params.knn, flann::SearchParams());
    for(int j=0; j<params.knn; j++)
    {
      graph.nearestNeighborsOut[i].push_back(indices[j]);
      graph.nearestNeighborsIn[indices[j]].push_back(i);
    }

    std::copy(dists.begin(), dists.end(), std::back_inserter(allDists));
  }

  std::sort(allDists.begin(), allDists.end());

  distanceQuantile = sqrt(allDists.at(cvRound(allDists.size() * params.distanceQuantile)));
}


void TangentVectorEstimator::followContourForward(const std::vector<cv::Point3f> &pointCloud, const std::vector<std::vector<int> > &edges, const std::vector<bool> &isTangentVectorEstimated, float distanceQuantile, std::vector<int> &contour, std::list<cv::Point3f> &orientations)
{
  CV_Assert(!contour.empty());

  int currentPointIdx = contour.back();
  bool isPointAdded = true;
  while(isPointAdded)
  {
    Point3f lastPoint = pointCloud.at(currentPointIdx);
    Point3f orientation(0, 0, 0);
    bool isOrienationEstimated = false;
    if(contour.size() >= static_cast<size_t>(params.minEstimationPointsCount))
    {
      orientation = estimateOrientation(pointCloud, contour);
      isOrienationEstimated = true;
    }
    orientations.push_back(orientation);


    isPointAdded = false;
    vector<float> dists;
    for(size_t i=0; i<edges.at(currentPointIdx).size(); i++)
    {
      int nextPointIdx = edges.at(currentPointIdx).at(i);
      if(!isPointProcessed(isTangentVectorEstimated, contour, nextPointIdx))
      {
        Point3f pt = pointCloud.at(nextPointIdx);
        Point3f vec = pt - lastPoint;
        double dist = norm(vec);
        if(dist > distanceQuantile * params.distanceQuantileFactor)
        {
          dists.push_back(std::numeric_limits<float>::max());
          continue;
        }
        if(isOrienationEstimated)
        {
          double cosPhi = vec.ddot(orientation) / norm(vec);
          dist *= (2.0 - cosPhi);
          //dist *= (1.1 - cosPhi);
        }
        dists.push_back(dist);
//        currentPointIdx = nextPointIdx;
//        contour.push_back(currentPointIdx);
//        isPointAdded = true;
//        break;
      }
      else
      {
        dists.push_back(std::numeric_limits<float>::max());
      }
    }

    vector<float>::iterator minElement = std::min_element(dists.begin(), dists.end());
    int minIdx = std::distance(dists.begin(), minElement);
    //cout << dists.at(minIdx) << " <-> " << *minElement << endl;

    float minDist = dists.at(minIdx);
    if(minDist != std::numeric_limits<float>::max())
    {
      currentPointIdx = edges.at(currentPointIdx).at(minIdx);
      contour.push_back(currentPointIdx);
      isPointAdded = true;
    }
  }
}

void TangentVectorEstimator::estimate(const std::vector<cv::Point3f> &pointCloud, std::vector<cv::Point3f> &tangentVectors)
{
//  Point3f centroid = getRobustCentroid(pointCloud);
//  cout << centroid<< endl;

  vector<bool> isTangentVectorEstimated(pointCloud.size(), false);

  Graph graph;
  float distanceQuantile;
  constructKNNGraph(pointCloud, graph, distanceQuantile);

  vector<vector<int> > contours;
  vector<list<Point3f> > orientations;
  for(size_t ptIdx=0; ptIdx<isTangentVectorEstimated.size(); ptIdx++)
  {
    if(isTangentVectorEstimated[ptIdx] == true)
      continue;

    vector<int> currentContour;
    list<Point3f> currentOrientations;
    currentContour.push_back(ptIdx);

    followContourForward(pointCloud, graph.nearestNeighborsOut, isTangentVectorEstimated, distanceQuantile, currentContour, currentOrientations);
    std::reverse(currentContour.begin(), currentContour.end());
    std::reverse(currentOrientations.begin(), currentOrientations.end());
    followContourForward(pointCloud, graph.nearestNeighborsIn, isTangentVectorEstimated, distanceQuantile, currentContour, currentOrientations);

    std::reverse(currentContour.begin(), currentContour.end());
    std::reverse(currentOrientations.begin(), currentOrientations.end());

    //followContour(true, graph, isTangentVectorEstimated, currentContour);

//    if(currentContour.size() >= minEstimationPointsCount)
//    {
//      cout << estimateOrientation(pointCloud, currentContour, maxEstimationPointsCount) << endl;
//    }


    //followContour(false, pointCloud, graph, isTangentVectorEstimated, currentContour);
    //followContourForward(graph, isTangentVectorEstimated, currentContour);
    //followContourBackward(graph, isTangentVectorEstimated, currentContour);

    for(vector<int>::iterator it = currentContour.begin(); it != currentContour.end(); it++)
    {
      isTangentVectorEstimated[*it] = true;
    }
    //TODO: don't copy memory
    contours.push_back(currentContour);
    orientations.push_back(currentOrientations);
  }




  //int numberOfContours = 20;
  //int numberOfContours = 40;
  //int numberOfContours = contours.size();
  //publishContours(pointCloud, contours, orientations, numberOfContours);
//  namedWindow("contours have been printed");
//  waitKey();

  //cout << "Size: " << contours.size() << endl;

  tangentVectors.resize(pointCloud.size());
  for(size_t i=0; i<contours.size(); i++)
  {
    if(contours[i].size() <= 2*static_cast<size_t>(params.minHalfContourPoints) + 5)
    {
      for(vector<int>::iterator it = contours[i].begin(); it != contours[i].end(); it++)
      {
        tangentVectors.at(*it) = params.nanOrientation;
      }
    }
    else
    {
      vector<Point3f> currentContour, currentOrientations;
      currentContour.reserve(contours[i].size());
      for(vector<int>::iterator it = contours[i].begin(); it != contours[i].end(); it++)
      {
        currentContour.push_back(pointCloud.at(*it));
      }
      estimateFinalOrientations(currentContour, currentOrientations);
      for(size_t j=0; j<contours[i].size(); j++)
      {
        tangentVectors.at(contours[i][j]) = currentOrientations[j];
      }
    }
  }
}

/*
void TangentVectorEstimator::publishContours(const std::vector<cv::Point3f> &pointCloud, const std::vector<std::vector<int> > &contours, const std::vector<std::list<cv::Point3f> > &orientations, int numberOfContours)
{
  vector<size_t> contourIndices;
  contourIndices.reserve(contours.size());
  for(size_t i=0; i<contours.size(); i++)
  {
    contourIndices.push_back(i);
  }

  ContoursSorter contoursSorter(&contours);
  std::sort(contourIndices.begin(), contourIndices.end(), contoursSorter);

  vector<Point3f> points = pointCloud;
  vector<int> indices(pointCloud.size(), 0);


  //publishPointsSlowly(*pt_pub, pointCloud, vector<Point3f>(), Scalar(0, 255, 0), "model", 0.002f, false);
  //namedWindow("wait");
  //waitKey();
  const size_t minNumberOfPoints = 2*static_cast<size_t>(params.minHalfContourPoints);
  for(int i=0; i<numberOfContours; i++)
  {
    size_t contourIdx = contourIndices[i];
    if(contours[contourIdx].size() < minNumberOfPoints)
      continue;

    vector<Point3f> currentContour;
    for(vector<int>::const_iterator it = contours[contourIdx].begin(); it != contours[contourIdx].end(); it++)
    {
      currentContour.push_back(pointCloud[*it]);
      points.push_back(pointCloud[*it]);
      indices.push_back(i+1);
    }


//    vector<Point3f> currentOrientations;
//    for(list<Point3f>::const_iterator it = orientations[contourIdx].begin(); it != orientations[contourIdx].end(); it++)
//    {
//      currentOrientations.push_back(*it);
//    }

    //publishPointsSlowly(*pt_pub, currentContour, Scalar(0, 0, 255), "contours", 0.003f, true);
    //publishPointsSlowly(*pt_pub, currentContour, currentOrientations, Scalar(255, 0, 0), "contours", 0.003f, true);
  }
}
*/

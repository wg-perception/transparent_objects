/*
 * kPartiteGraph.cpp
 *
 *  Created on: Apr 21, 2011
 *      Author: Ilya Lysenkov
 */

#include "edges_pose_refiner/kPartiteGraph.hpp"
#include "opencv2/flann/flann.hpp"

using namespace cv;
//using KPartiteGraph::VertexID;
//using KPartiteGraph::PartID;
//using KPartiteGraph::FullVertexIdx;

using std::pair;
using std::map;

KPartiteGraph::EdgeForQueue::EdgeForQueue(PartID _p1, VertexID _v1, PartID _p2, VertexID _v2, float _weight)
{
  p1 = _p1;
  v1 = _v1;
  p2 = _p2;
  v2 = _v2;
  weight = _weight;
}

KPartiteGraph::Edge::Edge(float _weight)
{
  weight = _weight;
}

KPartiteGraph::Vertex::Vertex(cv::Point3f _pt, size_t partsCount)
{
  pt = _pt;
  incidentEdges.resize(partsCount);
}

KPartiteGraph::FullVertexIdx::FullVertexIdx(PartID partID, VertexID vertexID)
{
  p = partID;
  v = vertexID;
}

KPartiteGraph::KPartiteGraph(const std::vector<std::vector<cv::Point3f> > &pointClouds)
{
  const size_t partsCount = pointClouds.size();
  parts.resize(partsCount);

  for(size_t partIdx=0; partIdx<parts.size(); partIdx++)
  {
    for(size_t vIdx=0; vIdx<pointClouds[partIdx].size(); vIdx++)
    {
      parts[partIdx].insert(pair<VertexID, Vertex>(vIdx, Vertex(pointClouds[partIdx][vIdx], partsCount)));
    }
  }

/*
  flann::LinearIndexParams indexParams;
  flannIndices.resize(partsCount);
  for(size_t i=0; i<partsCount; i++)
  {
    flannIndices[i] = new flann::Index(Mat(pointClouds[i]).reshape(1), indexParams);
  }
*/
}

bool KPartiteGraph::doesVertexExist(PartID p, VertexID v)
{
  CV_Assert(p < parts.size());
  return parts[p].find(v) != parts[p].end();
}

bool KPartiteGraph::doesVerticesAdjacent(PartID p1, VertexID v1, PartID p2, VertexID v2)
{
  CV_Assert(doesVertexExist(p1, v1) && doesVertexExist(p2, v2));

  std::map<VertexID, Edge> &edges = parts[p1].find(v1)->second.incidentEdges[p2];
  return edges.find(v2) != edges.end();
}

float KPartiteGraph::getEdgeWeight(PartID p1, VertexID v1, PartID p2, VertexID v2)
{
  CV_Assert(doesVerticesAdjacent(p1, v1, p2, v2));

  std::map<VertexID, Edge> &edges = parts[p1].find(v1)->second.incidentEdges[p2];
  return edges.find(v2)->second.weight;
}

void KPartiteGraph::addEdge(PartID p1, VertexID v1, PartID p2, VertexID v2, float weight)
{
  CV_Assert(doesVertexExist(p1, v1) && doesVertexExist(p2, v2));

  parts.at(p1).find(v1)->second.incidentEdges.at(p2).insert(pair<VertexID, Edge>(v2, Edge(weight)));
  parts.at(p2).find(v2)->second.incidentEdges.at(p1).insert(pair<VertexID, Edge>(v1, Edge(weight)));

  EdgeForQueue edge(p1, v1, p2, v2, weight);
  allEdges.push(edge);
}

void KPartiteGraph::removeVertex(PartID p, VertexID v)
{
  CV_Assert(doesVertexExist(p, v));
  vector<std::map<VertexID, Edge> > &edges = parts[p].find(v)->second.incidentEdges;

  for(size_t partIdx=0; partIdx < parts.size(); partIdx++)
  {
    for(std::map<VertexID, Edge>::iterator it=edges[partIdx].begin(); it!=edges[partIdx].end(); it++)
    {
      parts[partIdx].find(it->first)->second.incidentEdges[p].erase(v);
    }
  }

  parts[p].erase(v);
}

bool KPartiteGraph::getMinEdge(PartID &p1, VertexID &v1, PartID &p2, VertexID &v2, float &weight)
{
  //cout << allEdges.top().weight << endl;

  while(true)
  {
    if(allEdges.empty())
      return false;
    const EdgeForQueue &top = allEdges.top();
    bool isCorrect = doesVertexExist(top.p1, top.v1) && doesVertexExist(top.p2, top.v2) && doesVerticesAdjacent(top.p1, top.v1, top.p2, top.v2);
    if(isCorrect)
      break;
    else
      allEdges.pop();
  }


  const EdgeForQueue &top = allEdges.top();
  p1 = top.p1;
  v1 = top.v1;
  p2 = top.p2;
  v2 = top.v2;
  weight = top.weight;
  return true;







  //CV_Assert(false);
  weight = std::numeric_limits<float>::max();
  bool isFound = false;

  for(size_t i=0; i<parts.size(); i++)
  {
    for(std::map<VertexID, Vertex>::iterator it1 = parts[i].begin(); it1!=parts[i].end(); it1++)
    {
      for(size_t j=i+1; j<parts.size(); j++)
      {
        for(std::map<VertexID, Edge>::iterator it2 = it1->second.incidentEdges[j].begin(); it2!=it1->second.incidentEdges[j].end(); it2++)
        {
          if(it2->second.weight < weight)
          {
            weight = it2->second.weight;
            p1 = i;
            v1 = it1->first;
            p2 = j;
            v2 = it2->first;
            isFound = true;
          }
        }
      }
    }
  }
  return isFound;
}

void addEdges(int knn, const vector<vector<Point3f> > &pointClouds, vector<Ptr<flann::Index> > &flannIndices, size_t p1, size_t p2, Ptr<KPartiteGraph> graph)
{
  const size_t dim = 3;

  for(size_t ptIdx=0; ptIdx < pointClouds[p1].size(); ptIdx++)
  {
    vector<float> query = Mat(pointClouds[p1][ptIdx]);
    CV_Assert(query.size() == dim);

    vector<int> indices(knn);
    vector<float> dists(knn);
    flannIndices[p2]->knnSearch(query, indices, dists, knn, flann::SearchParams());

    for(size_t neighbor=0; neighbor < indices.size(); neighbor++)
    {
      graph->addEdge(p1, ptIdx, p2, indices[neighbor], dists[neighbor]);
    }
  }
}

void constructKNNG(int knn, const vector<vector<Point3f> > &pointClouds, Ptr<KPartiteGraph> &graph)
{
  const size_t partsCount = pointClouds.size();
  vector<size_t> verticesCounts;
  for(size_t i=0; i<partsCount; i++)
  {
    verticesCounts.push_back(pointClouds[i].size());
  }

  graph = new KPartiteGraph(pointClouds);

  cv::flann::LinearIndexParams indexParams;
  vector<Ptr<cv::flann::Index> > flannIndices(partsCount);
  for(size_t i=0; i<partsCount; i++)
  {
    flannIndices[i] = new flann::Index(Mat(pointClouds[i]).reshape(1), indexParams);
  }

  for(size_t i=0; i<partsCount; i++)
  {
    for(size_t j=0; j<partsCount; j++)
    {
      if(i != j)
      {
        addEdges(knn, pointClouds, flannIndices, i, j, graph);
      }
    }
  }
}

float getRij(const vector<vector<Point3f> > &pointClouds, size_t p1, size_t v1, size_t p2, size_t v2)
{
  return norm(pointClouds[p1][v1] - pointClouds[p2][v2]);
}

KPartiteGraph::VertexForQueue::VertexForQueue(PartID _p, VertexID _v, float _sum)
{
  p = _p;
  v = _v;
  representativity = _sum;
}

double KPartiteGraph::getVertexRepresentativity(size_t neighborsCount, PartID p, VertexID v)
{
  vector<EdgeForQueue> nearestEdges;
  getNearestEdges(p, v, nearestEdges);

  const vector<map<VertexID, Edge> > &edges = parts[p].find(v)->second.incidentEdges;
  vector<float> weights;
  for(size_t i=0; i<parts.size(); i++)
  {
    if(edges[i].empty())
      continue;
    float minWeight = std::numeric_limits<float>::max();
    for(map<VertexID, Edge>::const_iterator eIt=edges[i].begin(); eIt != edges[i].end(); eIt++)
    {
      if(eIt->second.weight < minWeight)
        minWeight = eIt->second.weight;
    }
    weights.push_back(minWeight);
  }
  if(weights.size() < neighborsCount)
    return std::numeric_limits<double>::max();


  std::sort(weights.begin(), weights.end());
  double sum = 0.;
  for(size_t i=0; i<neighborsCount; i++)
    sum += weights[i];

  return sum;
}

void KPartiteGraph::computeAllRepresentativites(size_t neighborsCount)
{
  for(size_t partIdx=0; partIdx<parts.size(); partIdx++)
  {
    for(map<VertexID, Vertex>::iterator vIt=parts[partIdx].begin(); vIt!=parts[partIdx].end(); vIt++)
    {
      double sum = getVertexRepresentativity(neighborsCount, partIdx, vIt->first);
      VertexForQueue v(partIdx, vIt->first, sum);
      allVertices.push(v);
    }
  }
}

cv::Point3d KPartiteGraph::getRobustCentroid(cv::Point3f currentCentroid, std::vector<EdgeForQueue> &edges, size_t neighborsCount)
{
  CV_Assert(neighborsCount <= edges.size());
  std::sort(edges.begin(), edges.end(), EdgeComparator());
  std::reverse(edges.begin(), edges.end());

  Point3d centroid = currentCentroid;
  for(size_t i=0; i<neighborsCount; i++)
  {
    Point3d pt = parts[edges[i].p2].find(edges[i].v2)->second.pt;
    centroid += pt;
  }
  centroid *= (1. / (neighborsCount+1));
  return centroid;
}

/*
void KPartiteGraph::estimateBestMCD(vector<EdgeForQueue> &edges, size_t minEdgesCount, size_t &bestEdgesCount, double &bestMCD)
{
  //CV_Assert(minEdgesCount <= edges.size());
  //  size_t bestEdgesCount = 0;
  //  double bestMCD = std::numeric_limits<double>::max();
  bestEdgesCount = 0;
  bestMCD = std::numeric_limits<double>::max();

  if(edges.size() < minEdgesCount)
  {
    return;
  }

  std::sort(edges.begin(), edges.end(), EdgeComparator());
  std::reverse(edges.begin(), edges.end());


  vector<Point3f> samplesVector;
  for(size_t i=0; i<edges.size(); i++)
  {
    samplesVector.push_back(parts[edges[i].p2].find(edges[i].v2)->second.pt);
  }
  Mat samples = Mat(samplesVector).reshape(1);

  for(size_t i=minEdgesCount; i<=edges.size(); i++)
  {
    Mat covar;
    Mat mean;
    calcCovarMatrix(samples.rowRange(Range(0, i)), covar, mean, CV_COVAR_NORMAL + CV_COVAR_SCALE + CV_COVAR_ROWS);

    const int dim = 3;
    CV_Assert(covar.rows == dim && covar.cols == dim);

    double det = determinant(covar) * pow(i, -dim);
    if(det < bestMCD)
    {
      bestEdgesCount = i;
      bestMCD = det;
    }
  }
  CV_Assert(minEdgesCount <= bestEdgesCount && bestEdgesCount <= edges.size());

  cout << "BestEdgesCount: " << bestEdgesCount << endl;
  //return bestEdgesCount;
}
*/

void KPartiteGraph::getNearestEdges(PartID partID, VertexID vertexID, vector<EdgeForQueue> &nearestEdges)
{
  nearestEdges.clear();
  for(size_t i=0; i<parts.size(); i++)
  {
    const map<VertexID, Edge> &edges = parts.at(partID).find(vertexID)->second.incidentEdges.at(i);
    if(edges.empty())
      continue;

    EdgeForQueue minEdge(-1, -1, -1, -1, std::numeric_limits<float>::max());
    for(map<VertexID, Edge>::const_iterator eIt=edges.begin(); eIt != edges.end(); eIt++)
    {
      if(eIt->second.weight < minEdge.weight)
      {
        minEdge = EdgeForQueue(partID, vertexID, i, eIt->first, eIt->second.weight);
      }
    }
    nearestEdges.push_back(minEdge);
  }
}

bool KPartiteGraph::getBestRepresentativeVertex(size_t neighborsCount, VertexForQueue &top)
{
  const float eps = 1e-12;

  bool isBest = false;
  while(!isBest)
  {
    do
    {
      if(allVertices.empty())
        return false;
      top = allVertices.top();
      allVertices.pop();
    }
    while(!doesVertexExist(top.p, top.v));

    float representativity = getVertexRepresentativity(neighborsCount, top.p, top.v);

    isBest = fabs(representativity - top.representativity) <= eps;
    if(!isBest && representativity < std::numeric_limits<float>::max())
    {
      allVertices.push(VertexForQueue(top.p, top.v, representativity));
    }
  }
  return true;
}

void KPartiteGraph::runCStep(size_t neighborsCount, std::vector<EdgeForQueue> &nearestEdges, cv::Point3f &centroid)
{
  centroid = getRobustCentroid(centroid, nearestEdges, neighborsCount);
  nearestEdges.clear();
  for(size_t i=0; i<parts.size(); i++)
  {
    int bestIdx = -1;
    double bestDist = std::numeric_limits<float>::max();
    for(map<VertexID, Vertex>::iterator it = parts.at(i).begin(); it!=parts.at(i).end(); it++)
    {
      double dist = norm(centroid - it->second.pt);
      if(dist < bestDist)
      {
        bestDist = dist;
        bestIdx = it->first;
      }
    }

    nearestEdges.push_back(EdgeForQueue(-1, -1, i, bestIdx, bestDist));
  }

}

void KPartiteGraph::removeInliersNeighbors(const std::vector<FullVertexIdx> &inliers, size_t minInliers)
{
  for(size_t inlierIdx=0; inlierIdx<inliers.size(); inlierIdx++)
  {
    vector<map<VertexID, Edge> > edges = parts[inliers[inlierIdx].p].find(inliers[inlierIdx].v)->second.incidentEdges;
    for(size_t partIdx=0; partIdx<edges.size(); partIdx++)
    {
      for(map<VertexID, Edge>::iterator it = edges[partIdx].begin(); it != edges[partIdx].end(); it++)
      {
        VertexID v = it->first;

        //bool remove = false;
        size_t neighbors = 0;
        for(size_t i=0; i<inliers.size(); i++)
        {
          if(partIdx == inliers[i].p && v == inliers[i].v)
          {
            neighbors = 0;
            break;
          }

          if(doesVerticesAdjacent(partIdx, v, inliers[i].p, inliers[i].v))
            neighbors++;
        }

        if(neighbors >= minInliers)
        {
          removeVertex(partIdx, v);
        }
      }
    }
  }
}

bool KPartiteGraph::getBestCentroid(size_t cstepsCount, size_t inliersNeighbrosCount, size_t neighborsCount, cv::Point3f &centroid)
{
  VertexForQueue top(-1, -1, -1);
  bool isBest = getBestRepresentativeVertex(neighborsCount, top);
  if(!isBest)
    return false;
  centroid = parts[top.p].find(top.v)->second.pt;

  vector<EdgeForQueue> nearestEdges;
  getNearestEdges(top.p, top.v, nearestEdges);
  if(nearestEdges.size() < neighborsCount)
  {
    return false;
  }


  for(size_t iterIdx = 0; iterIdx < cstepsCount; iterIdx++)
  {
    runCStep(neighborsCount, nearestEdges, centroid);
  }

  std::sort(nearestEdges.begin(), nearestEdges.end(), EdgeComparator());
  std::reverse(nearestEdges.begin(), nearestEdges.end());

  //vertices used in centroid estimating
  vector<FullVertexIdx> inliers;
  for(size_t i=0; i<neighborsCount; i++)
  {
    inliers.push_back(FullVertexIdx(nearestEdges[i].p2, nearestEdges[i].v2));
  }

  //remove vertices which are neighbors with at least inliersNeighbrosCount inliers
  removeInliersNeighbors(inliers, inliersNeighbrosCount);
  for(size_t i=0; i<inliers.size(); i++)
  {
    removeVertex(inliers[i].p, inliers[i].v);
  }

  return true;
}


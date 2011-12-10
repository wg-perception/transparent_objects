/*
 * kPartiteGraph.hpp
 *
 *  Created on: Apr 21, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef KPARTITEGRAPH_HPP_
#define KPARTITEGRAPH_HPP_

#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/registerPointClouds.hpp"
#include <queue>

#define MCD_EDGE_MODEL

/** \brief This class represent k-partite graphs for internal using */
class KPartiteGraph
{
public:
  typedef size_t VertexID;
  typedef size_t PartID;
  struct Edge
  {
    float weight;
    Edge(float weight = GET_DEFAULT_EDGE_WEIGHT());
  };

  struct Vertex
  {
    std::vector<std::map<VertexID, Edge> >incidentEdges;
#ifdef MCD_EDGE_MODEL
    cv::Point3f pt;

    Vertex(cv::Point3f pt, size_t partsCount);
#else
    Vertex(size_t partsCount);
#endif
  };

  struct FullVertexIdx
  {
    PartID p;
    VertexID v;

    FullVertexIdx(PartID partID, VertexID vertexID);
  };

  KPartiteGraph(const std::vector<std::vector<cv::Point3f> > &pointClouds);
  void addEdge(PartID p1, VertexID v1, PartID p2, VertexID v2, float weight = GET_DEFAULT_EDGE_WEIGHT());
  void removeVertex(PartID p, VertexID v);

  bool getMinEdge(PartID &p1, VertexID &v1, PartID &p2, VertexID &v2, float &weight);
  bool doesVerticesAdjacent(PartID p1, VertexID v1, PartID p2, VertexID v2);

  float getEdgeWeight(PartID p1, VertexID v1, PartID p2, VertexID v2);

//private:
  bool doesVertexExist(PartID p, VertexID v);
  static float GET_DEFAULT_EDGE_WEIGHT(){ return 1.; }

  std::vector<std::map<VertexID, Vertex> > parts;

  struct EdgeForQueue
  {
    float weight;
    PartID p1, p2;
    VertexID v1, v2;

    EdgeForQueue(PartID p1, VertexID v1, PartID p2, VertexID v2, float weight);
  };

  struct EdgeComparator
  {
    bool operator()(const EdgeForQueue &edge1, const EdgeForQueue &edge2)
    {
      return edge1.weight > edge2.weight;
    }
  };

  std::priority_queue<EdgeForQueue, std::vector<EdgeForQueue>, EdgeComparator> allEdges;


public:
  bool getBestCentroid(size_t cstepsCount, size_t inliersNeighbrosCount, size_t neighborsCount, cv::Point3f &centroid);
  void computeAllRepresentativites(size_t neighborsCount);

private:
  struct VertexForQueue
  {
    float representativity;
    PartID p;
    VertexID v;

    VertexForQueue(PartID p, VertexID v, float sumOfDistances);
  };

  bool getBestRepresentativeVertex(size_t neighborsCount, VertexForQueue &v);

  void runCStep(size_t neighborsCount, std::vector<EdgeForQueue> &nearestEdges, cv::Point3f &centroid);
  void removeInliersNeighbors(const std::vector<FullVertexIdx> &inliers, size_t minInliers);
  cv::Point3d getRobustCentroid(cv::Point3f currentCentroid, std::vector<EdgeForQueue> &edges, size_t neighborsCount);
  //void estimateBestMCD(vector<EdgeForQueue> &edges, size_t minEdgesCount, size_t &bestEdgesCount, double &bestMCD);
  double getVertexRepresentativity(size_t neighborsCount, PartID p, VertexID v);
  void getNearestEdges(PartID partID, VertexID vertexID, std::vector<EdgeForQueue> &nearestEdges);

  struct VertexComparator
  {
    bool operator()(const VertexForQueue &v1, const VertexForQueue &v2)
    {
      return v1.representativity > v2.representativity;
    }
  };

  std::priority_queue<VertexForQueue, std::vector<VertexForQueue>, VertexComparator> allVertices;
  //vector<cv::Ptr<cv::flann::Index> > flannIndices;
};

void constructKNNG(int knn, const std::vector<std::vector<cv::Point3f> > &pointClouds, cv::Ptr<KPartiteGraph> &graph);

#endif /* KPARTITEGRAPH_HPP_ */

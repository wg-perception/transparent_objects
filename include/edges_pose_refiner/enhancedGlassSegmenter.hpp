#ifndef ENHANCED_GLASS_SEGMENTER_HPP__
#define ENHANCED_GLASS_SEGMENTER_HPP__

#include <opencv2/opencv.hpp>
#include <numeric>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

#include "edges_pose_refiner/utils.hpp"
#include "edges_pose_refiner/geodesicActiveContour.hpp"
#include "edges_pose_refiner/region.hpp"
#include "edges_pose_refiner/segmentedImage.hpp"

struct VertexProperties
{
  cv::Point pt;
  float orientation;
  bool isRegion;
  int regionIndex;
};

struct EdgeProperties
{
  float length;
  float maximumAngle;

  EdgeProperties();
};

typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS, VertexProperties, EdgeProperties> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor VertexDescriptor;
typedef boost::graph_traits<Graph>::edge_descriptor EdgeDescriptor;


void computeColorSimilarity(const Region &region_1, const Region &region_2, float &distance);
void computeTextureDistortion(const Region &region_1, const Region &region_2, float &distance);
void computeOverlayConsistency(const Region &region_1, const Region &region_2, float &slope, float &intercept);

//TODO: is it possible to use the index to access a vertex directly?
VertexDescriptor getRegionVertex(const Graph &graph, int regionIndex);
VertexDescriptor insertPoint(const cv::Mat &segmentation, const cv::Mat &orientations, cv::Point pt, Graph &graph);
void edges2graph(const cv::Mat &segmentation, const std::vector<Region> &regions, const cv::Mat &edges, Graph &graph);
cv::Point getNextPoint(cv::Point previous, cv::Point current);
bool areRegionsOnTheSameSide(const cv::Mat &path, cv::Point firstPathEdgel, cv::Point lastPathEdgel, cv::Point center_1, cv::Point center_2);

struct MLData
{
  cv::Mat samples;
  cv::Mat responses;

  bool isValid() const;
  void push_back(const MLData &mlData);
  int getDimensionality() const;
  int getSamplesCount() const;
  void write(const std::string name) const;

  void evaluate(const cv::Mat &predictedLabels, float &error, cv::Mat &confusionMatrix) const;

private:
  int computeClassesCount() const;
};

class GlassClassifier
{
  public:
    void train();
    void test(const SegmentedImage &testImage, const cv::Mat &groundTruthMask, cv::Mat &boundaryStrength) const;
  private:
    typedef cv::Vec4f Sample;

    static void segmentedImage2MLData(const SegmentedImage &image, const cv::Mat &groundTruthMask, bool withAllSymmetricSamples, MLData &mlData);
    static void segmentedImage2pairwiseSamples(const SegmentedImage &segmentedImage, cv::Mat &samples, const cv::Mat &scalingSlope = cv::Mat(), const cv::Mat &scalingIntercept = cv::Mat());
    static void segmentedImage2pairwiseResponses(const SegmentedImage &segmentedImage, const cv::Mat &groundTruthMask, cv::Mat &responses);
    void predict(const MLData &data, cv::Mat &confidences) const;

    void computeNormalizationParameters(const MLData &trainingData);

    //TODO: block junctions
    //TODO: extend to the cases when regions are not connected to each other
    static void estimateAffinities(const Graph &graph, size_t regionCount, cv::Size imageSize, int regionIndex, std::vector<float> &affinities);
    static void computeAllAffinities(const std::vector<Region> &regions, const Graph &graph, cv::Mat &affinities);
    static void computeBoundaryPresences(const std::vector<Region> &regions, const cv::Mat &edges, cv::Mat &boundaryPresences);
    void computeAllDiscrepancies(const SegmentedImage &testImage, const cv::Mat &groundTruthMask, cv::Mat &discrepancies) const;
    void computeBoundaryStrength(const SegmentedImage &testImage, const cv::Mat &edges, const cv::Mat &groundTruthMask, const Graph &graph, float affinityWeight, cv::Mat &boundaryStrength) const;


    static void regions2samples(const Region &region_1, const Region &region_2, cv::Mat &ecaSample, cv::Mat &dcaSample, cv::Mat &fullSample);

    CvSVM svm;
    cv::Mat scalingSlope, scalingIntercept;
    float normalizationSlope, normalizationIntercept;
};

enum TrainingLabels {THE_SAME = 0, GLASS_COVERED = 1, INVALID = 2};
void normalizeTrainingData(cv::Mat &trainingData, cv::Mat &scalingSlope, cv::Mat &scalingIntercept);
void getNormalizationParameters(const CvSVM *svm, const cv::Mat &trainingData, const std::vector<int> &trainingLabelsVec, float &normalizationSlope, float &normalizationIntercept);



void visualizeClassification(const std::vector<Region> &regions, const std::vector<float> &labels, cv::Mat &visualization);

#endif
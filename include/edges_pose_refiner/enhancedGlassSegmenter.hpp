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
void computeMedianColorDistance(const Region &region_1, const Region &region_2, float &distance);
void computeTextureDistortion(const Region &region_1, const Region &region_2, float &distance);
void computeOverlayConsistency(const Region &region_1, const Region &region_2, float &slope, float &intercept);
void computeContrastDistance(const Region &region_1, const Region &region_2, float &rmsDistance, float &michelsonDistance, float &robustMichelsonDistance);

//TODO: is it possible to use the index to access a vertex directly?
VertexDescriptor getRegionVertex(const Graph &graph, int regionIndex);
VertexDescriptor insertPoint(const cv::Mat &segmentation, const cv::Mat &orientations, cv::Point pt, Graph &graph);
void edges2graph(const cv::Mat &segmentation, const std::vector<Region> &regions, const cv::Mat &edges, Graph &graph);
cv::Point getNextPoint(cv::Point previous, cv::Point current);
bool areRegionsOnTheSameSide(const cv::Mat &path, cv::Point firstPathEdgel, cv::Point lastPathEdgel, cv::Point center_1, cv::Point center_2);

struct MLData
{
  cv::Mat samples;
  cv::Mat mask;
  cv::Mat responses;

  bool isValid() const;
  void push_back(const MLData &mlData);
  int getDimensionality() const;
  int getSamplesCount() const;
  void write(const std::string name) const;

  void removeMaskedOutSamples();

  void evaluate(const cv::Mat &predictedLabels, float &error, cv::Mat &confusionMatrix) const;

private:
  int computeClassesCount() const;
};

struct GlassClassifierParams
{
  int targetNegativeSamplesCount;

  GlassClassifierParams()
  {
    targetNegativeSamplesCount = 60000;
  }
};

class GlassClassifier
{
  public:
    GlassClassifier(const GlassClassifierParams &params = GlassClassifierParams());
    void train(const std::string &trainingFilesList, const std::string &groundTruthFilesList);
    void test(const SegmentedImage &testImage, const cv::Mat &groundTruthMask, cv::Mat &boundaryStrength) const;

    static void regions2samples(const Region &region_1, const Region &region_2, cv::Mat &fullSample);

    bool read(const std::string &filename);
    void write(const std::string &filename);
  private:
    typedef cv::Vec<float, 8> Sample;

    static void segmentedImage2MLData(const SegmentedImage &image, const cv::Mat &groundTruthMask, bool useOnlyAdjacentRegions, MLData &mlData);
    static void segmentedImage2pairwiseSamples(const SegmentedImage &segmentedImage, cv::Mat &samples, const cv::Mat &scalingSlope = cv::Mat(), const cv::Mat &scalingIntercept = cv::Mat());
    static void segmentedImage2pairwiseResponses(const SegmentedImage &segmentedImage, const cv::Mat &groundTruthMask, bool useOnlyAdjacentRegions, cv::Mat &responses);
    void predict(const MLData &data, cv::Mat &confidences) const;

    void computeNormalizationParameters(const MLData &trainingData);

    //TODO: block junctions
    //TODO: extend to the cases when regions are not connected to each other
    static void estimateAffinities(const Graph &graph, size_t regionCount, cv::Size imageSize, int regionIndex, std::vector<float> &affinities);
    static void computeAllAffinities(const std::vector<Region> &regions, const Graph &graph, cv::Mat &affinities);
    static void computeBoundaryPresences(const std::vector<Region> &regions, const cv::Mat &edges, cv::Mat &boundaryPresences);
    void computeAllDiscrepancies(const SegmentedImage &testImage, const cv::Mat &groundTruthMask, cv::Mat &discrepancies, std::vector<bool> &isRegionValid) const;
    void computeBoundaryStrength(const SegmentedImage &testImage, const cv::Mat &edges, const cv::Mat &groundTruthMask, const Graph &graph, float affinityWeight, cv::Mat &boundaryStrength) const;

    CvSVM svm;
    cv::Mat scalingSlope, scalingIntercept;
    float normalizationSlope, normalizationIntercept;

    GlassClassifierParams params;
};

enum TrainingLabels {THE_SAME = 0, GLASS_COVERED = 1, GROUND_TRUTH_INVALID = 2, COMPLETELY_INVALID = 3};
void normalizeTrainingData(cv::Mat &trainingData, cv::Mat &scalingSlope, cv::Mat &scalingIntercept);
void getNormalizationParameters(const CvSVM *svm, const cv::Mat &trainingData, const std::vector<int> &trainingLabelsVec, float &normalizationSlope, float &normalizationIntercept);



void visualizeClassification(const std::vector<Region> &regions, const std::vector<float> &labels, cv::Mat &visualization);

#endif

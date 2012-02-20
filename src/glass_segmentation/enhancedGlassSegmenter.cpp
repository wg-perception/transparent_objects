#include <opencv2/opencv.hpp>
#include <fstream>
#include <numeric>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

#include "edges_pose_refiner/utils.hpp"
#include "edges_pose_refiner/geodesicActiveContour.hpp"

#include "edges_pose_refiner/enhancedGlassSegmenter.hpp"
#include "edges_pose_refiner/segmentedImage.hpp"

using namespace cv;
using std::cout;
using std::endl;

enum RegionLabel {GLASS, BACKGROUND, NOT_VALID};

EdgeProperties::EdgeProperties()
{
  length = 0.0f;
  maximumAngle = 0.0f;
}


void computeColorSimilarity(const Region &region_1, const Region &region_2, float &distance)
{
  Mat hist_1 = region_1.getColorHistogram();
  Mat hist_2 = region_2.getColorHistogram();

  //TODO: experiment with different distances
  distance = norm(hist_1 - hist_2);
}

void computeTextureDistortion(const Region &region_1, const Region &region_2, float &distance)
{
  Mat hist_1 = region_1.getTextonHistogram();
  Mat hist_2 = region_2.getTextonHistogram();

  //TODO: experiment with different distances
  distance = norm(hist_1 - hist_2);
}

//TODO: what if I = I_B?
void computeOverlayConsistency(const Region &region_1, const Region &region_2, float &slope, float &intercept)
{
  //TODO: move up
  const float minAlpha =-0.001f;
  const float maxAlpha = 1.001f;

  Mat clusters_1 = region_1.getIntensityClusters();
  Mat clusters_2 = region_2.getIntensityClusters();

  Mat b = clusters_1.clone();
  const int dim = 2;
  Mat A = Mat(b.rows, dim, b.type());
  Mat col_0 = A.col(0);
  clusters_2.copyTo(col_0);
  A.col(1).setTo(1.0);

  Mat model;
  solve(A, b, model, DECOMP_SVD);
  CV_Assert(model.type() == CV_32FC1);
  CV_Assert(model.total() == dim);
  if (model.at<float>(0) < minAlpha || model.at<float>(0) > maxAlpha)
  {
    b = clusters_2.clone();
    clusters_1.copyTo(col_0);
    solve(A, b, model, DECOMP_SVD);
  }
  if (model.at<float>(0) < minAlpha || model.at<float>(0) > maxAlpha)
  {
    cout << A << endl;
    cout << b << endl;
    cout << model << endl;
    //TODO: fix the problem with uniform regions
    CV_Error(CV_StsError, "Cannot estimate overlay consistency");
  }

  slope = model.at<float>(0);
  intercept = model.at<float>(1);
}


void regions2samples(const Region &region_1, const Region &region_2, cv::Mat &ecaSample, cv::Mat &dcaSample, cv::Mat &fullSample)
{
  float colorDistance;
  computeColorSimilarity(region_1, region_2, colorDistance);
  float slope, intercept;
  computeOverlayConsistency(region_1, region_2, slope, intercept);
  float textureDistance;
  computeTextureDistortion(region_1, region_2, textureDistance);

  const int dim = 3;
  ecaSample = (Mat_<float>(1, dim) << intercept, colorDistance, slope);
  dcaSample = (Mat_<float>(1, dim) << textureDistance, colorDistance, slope);
  fullSample = (Mat_<float>(1, 4) << intercept, colorDistance, slope, textureDistance);
}

void normalizeTrainingData(cv::Mat &trainingData, cv::Mat &scalingSlope, cv::Mat &scalingIntercept)
{
  Mat maxData, minData;
  reduce(trainingData, maxData, 0, CV_REDUCE_MAX);
  reduce(trainingData, minData, 0, CV_REDUCE_MIN);
//  Mat scalingSlope, scalingIntercept;
  scalingSlope = 2.0 / (maxData - minData);
  scalingIntercept = -scalingSlope.mul(minData) - 1.0;
  CV_Assert(scalingSlope.size() == scalingIntercept.size());
  CV_Assert(scalingSlope.rows == 1 && scalingSlope.cols == trainingData.cols);
  for (int i = 0; i < trainingData.rows; ++i)
  {
    Mat row = trainingData.row(i);
    Mat scaledRow = row.mul(scalingSlope)  + scalingIntercept;
    scaledRow.copyTo(row);
  }

  //TODO: remove
  reduce(trainingData, maxData, 0, CV_REDUCE_MAX);
  reduce(trainingData, minData, 0, CV_REDUCE_MIN);
  cout << "max: " << maxData << endl;
  cout << "min: " << minData << endl;
}

void getNormalizationParameters(const CvSVM *svm, const cv::Mat &trainingData, const std::vector<int> &trainingLabelsVec, float &normalizationSlope, float &normalizationIntercept)
{
  int wrongClassificationCount = 0;
  float minSVMDistance = 0.0f;
  float maxSVMDistance = 0.0f;
  CV_Assert(trainingData.rows > 0);
  Mat zeroSample = trainingData.row(0);
  int zeroPredictedLabel = cvRound(svm->predict(zeroSample));
  float zeroDistance = svm->predict(zeroSample, true);
  int negativeLabel = zeroDistance < 0 ? zeroPredictedLabel : 1 - zeroPredictedLabel;
  cout << "Negative label: " << negativeLabel << endl;

  const int labelCount = 2;
  Mat confusionMatrix(labelCount, labelCount, CV_32SC1, Scalar(0));
  for (size_t i = 0; i < trainingData.rows; ++i)
  {
    Mat sample = trainingData.row(i);
    float distance = svm->predict(sample, true);
    int label = distance < 0 ? negativeLabel : 1 - negativeLabel;
    minSVMDistance = std::min(minSVMDistance, distance);
    maxSVMDistance = std::max(maxSVMDistance, distance);

    if (label != trainingLabelsVec[i])
    {
      ++wrongClassificationCount;
    }
    ++confusionMatrix.at<int>(trainingLabelsVec[i], label);
  }
  cout << "confusion matrix: " << endl;
  cout << confusionMatrix << endl;
  Mat normalizedConfusionMatrix;
  confusionMatrix.convertTo(normalizedConfusionMatrix, CV_32FC1);
  normalizedConfusionMatrix.at<float>(0, 0) /= confusionMatrix.at<int>(0, 0) + confusionMatrix.at<int>(0, 1);
  normalizedConfusionMatrix.at<float>(0, 1) /= confusionMatrix.at<int>(0, 0) + confusionMatrix.at<int>(0, 1);
  normalizedConfusionMatrix.at<float>(1, 0) /= confusionMatrix.at<int>(1, 0) + confusionMatrix.at<int>(1, 1);
  normalizedConfusionMatrix.at<float>(1, 1) /= confusionMatrix.at<int>(1, 0) + confusionMatrix.at<int>(1, 1);
  cout << "normalized confusion matrix: " << endl;
  cout << normalizedConfusionMatrix << endl;

  float spread = maxSVMDistance - minSVMDistance;
  const float eps = 1e-2;
  CV_Assert(spread > eps);
  normalizationSlope = 1.0 / spread;
  normalizationIntercept = -minSVMDistance * normalizationSlope;

  if (negativeLabel == GLASS_COVERED)
//  if (negativeLabel == THE_SAME)
  {
    normalizationSlope = -normalizationSlope;
    normalizationIntercept = -normalizationIntercept + 1;
  }
  float hyperplaneValue = normalizationIntercept;

  cout << "training error: " << static_cast<float>(wrongClassificationCount) / trainingData.rows << endl;


  wrongClassificationCount = 0;
  for (size_t i = 0; i < trainingData.rows; ++i)
  {
    Mat sample = trainingData.row(i);
    float distance = normalizationSlope * svm->predict(sample, true) + normalizationIntercept;
    int label = distance < hyperplaneValue ? THE_SAME : GLASS_COVERED;

    if (label != trainingLabelsVec[i])
    {
      ++wrongClassificationCount;
    }
  }

  cout << "training error: " << static_cast<float>(wrongClassificationCount) / trainingData.rows << endl;
}


bool MLData::isValid() const
{
  return (samples.type() == CV_32FC1 &&
          responses.type() == CV_32SC1 &&
          samples.rows == responses.rows);
}

int MLData::getDimensionality() const
{
  return samples.cols;
}

int MLData::getSamplesCount() const
{
  return samples.rows;
}

void MLData::push_back(const MLData &mlData)
{
  samples.push_back(mlData.samples);
  responses.push_back(mlData.responses);
}

void MLData::save(const std::string name) const
{
  CV_Assert(isValid());

/*
  {
    string wekaFilename = name + "_weka.csv";
    std::ofstream fout(wekaFilename.c_str());
    CV_Assert(false);
    fout << "first, second, third, label" << endl;
    for (int i = 0; i < trainingData.rows; ++i)
    {
      for (int j = 0; j < trainingData.cols; ++j)
      {
        fout << trainingData.at<float>(i, j) << ", ";
      }
      if (labels.at<int>(i) == THE_SAME)
      {
        fout << "theSame" << endl;
      }
      else
      {
        fout << "different" << endl;
      }
    }
    fout.close();
  }
*/

  {
    string svmFilename = name + "_svm.csv";
    std::ofstream fout(svmFilename.c_str());
    for (int i = 0; i < samples.rows; ++i)
    {
      fout << responses.at<int>(i) << " ";
      for (int j = 0; j < samples.cols; ++j)
      {
        fout << j << ":" << samples.at<float>(i, j) << " ";
      }
      fout << endl;
    }
    fout.close();
  }
}

void GlassClassifier::train()
{
  //TODO: move up
  const string trainingFilesList = "/media/2Tb/transparentBases/rgbGlassData/trainingImages.txt";
  const string groundTruthFilesList = "/media/2Tb/transparentBases/rgbGlassData/trainingImagesGroundTruth.txt";
  const float maxSampleDistance = 0.1f;
  float confidentLabelArea = 0.9f;
//  float confidentLabelArea = 0.6f;

  vector<string> trainingGroundTruhFiles;
  readLinesInFile(groundTruthFilesList, trainingGroundTruhFiles);

  vector<string> trainingFiles;
  readLinesInFile(trainingFilesList, trainingFiles);

  const size_t imageCount = trainingGroundTruhFiles.size();
  CV_Assert(trainingFiles.size() == imageCount);

  MLData trainingData;
  for (size_t imageIndex = 0; imageIndex < imageCount; ++imageIndex)
  {
    Mat trainingImage = imread(trainingFiles[imageIndex]);
    CV_Assert(!trainingImage.empty());

    Mat groundTruthMask = imread(trainingGroundTruhFiles[imageIndex], CV_LOAD_IMAGE_GRAYSCALE);
    CV_Assert(!groundTruthMask.empty());
    CV_Assert(trainingImage.size() == groundTruthMask.size());

    SegmentedImage segmentedImage(trainingImage);
    segmentedImage.showSegmentation("train segmentation");

    MLData currentMLData;
    segmentedImage2MLData(segmentedImage, groundTruthMask, false, currentMLData);
    CV_Assert(currentMLData.isValid());
    trainingData.push_back(currentMLData);
  }
  CV_Assert(trainingData.isValid());
//  Mat ecaTrainingData = fullTrainingData.colRange(Range(0, 3));
//  Mat dcaTrainingData = fullTrainingData.colRange(Range(1, 4));
//  CV_Assert(ecaTrainingData.size() == dcaTrainingData.size());
//  CV_Assert(ecaTrainingData.type() == CV_32FC1);
//  CV_Assert(dcaTrainingData.type() == CV_32FC1);

//  Mat ecaScalingSlope(1, ecaTrainingData.cols, CV_32FC1);
//  ecaScalingSlope = 1.0;
//  Mat ecaScalingIntercept = Mat::zeros(1, ecaTrainingData.cols, CV_32FC1);
//  Mat dcaScalingSlope(1, dcaTrainingData.cols, CV_32FC1);
//  Mat dcaScalingIntercept = Mat::zeros(1, dcaTrainingData.cols, CV_32FC1);
//  normalizeTrainingData(ecaTrainingData, ecaScalingSlope, ecaScalingIntercept);
//  normalizeTrainingData(dcaTrainingData, dcaScalingSlope, dcaScalingIntercept);
//  scalingSlope = ecaScalingSlope;
//  scalingIntercept = ecaScalingIntercept;

  Mat fullScalingSlope(1, trainingData.getDimensionality(), CV_32FC1);
  fullScalingSlope = 1.0;
  Mat fullScalingIntercept = Mat::zeros(1, trainingData.getDimensionality(), CV_32FC1);
  scalingSlope = fullScalingSlope;
  scalingIntercept = fullScalingIntercept;

//  CV_Assert(trainingLabels.rows == ecaTrainingData.rows);
//  CV_Assert(trainingLabels.cols == 1);
//  CV_Assert(trainingLabels.type() == CV_32SC1);

  CvSVMParams svmParams;
  //TODO: move up
  svmParams.svm_type = CvSVM::C_SVC;
  svmParams.C = 2;
//  svmParams.svm_type = CvSVM::NU_SVC;

  svmParams.nu = 0.1;
//  svmParams.nu = 0.01;
  svmParams.gamma = 8;
  svmParams.kernel_type = CvSVM::RBF;

//  cout << dcaTrainingData << endl;
//  saveMLData("ecaData.csv", ecaTrainingData, trainingLabels);
//  saveMLData("dcaData.csv", dcaTrainingData, trainingLabels);
  trainingData.save("fullTrainingData");

//  cout << ecaTrainingData << endl;
//  cout << "ecaTrainingData size: " << ecaTrainingData.rows << " x " << ecaTrainingData.cols << endl;
//  cout << "dcaTrainingData size: " << dcaTrainingData.rows << " x " << dcaTrainingData.cols << endl;
//  cout << "fullTrainingData size: " << fullTrainingData.rows << " x " << fullTrainingData.cols << endl;
  cout << "Glass covered: " << countNonZero(trainingData.responses == GLASS_COVERED) << endl;
  cout << "The same: " << countNonZero(trainingData.responses == THE_SAME) << endl;

  cout << "training...  " << std::flush;

//  bool isTrained = svm.train(dcaTrainingData, trainingLabels, Mat(), Mat(), svmParams);
//  bool isTrained = svm.train(ecaTrainingData, trainingLabels, Mat(), Mat(), svmParams);
  bool isTrained = svm.train(trainingData.samples, trainingData.responses, Mat(), Mat(), svmParams);
//  bool isTrained = svm.train_auto(ecaTrainingData, trainingLabels, Mat(), Mat(), svmParams);
//  bool isTrained = svm.train_auto(dcaTrainingData, trainingLabels, Mat(), Mat(), svmParams);
  cout << "done: " << isTrained << endl;
  CvSVMParams optimalParams = svm.get_params();
  cout << "C: " << optimalParams.C << endl;
  cout << "nu: " << optimalParams.nu << endl;
  cout << "gamma: " << optimalParams.gamma << endl;


  getNormalizationParameters(&svm, trainingData.samples, trainingData.responses, normalizationSlope, normalizationIntercept);
//  getNormalizationParameters(&svm, ecaTrainingData, trainingLabelsVec, normalizationSlope, normalizationIntercept);
//  getNormalizationParameters(&svm, dcaTrainingData, trainingLabelsVec, normalizationSlope, normalizationIntercept);
}

//TODO: is it possible to use the index to access a vertex directly?
VertexDescriptor getRegionVertex(const Graph &graph, int regionIndex)
{
  boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
  for (tie(vi, vi_end) = vertices(graph); vi != vi_end; ++vi)
  {
    if (graph[*vi].isRegion && graph[*vi].regionIndex == regionIndex)
    {
      return *vi;
    }
  }
  CV_Assert(false);
}

VertexDescriptor insertPoint(const cv::Mat &segmentation, const cv::Mat &orientations, cv::Point pt, Graph &graph)
{
  //TODO: move up
  const int maxDistanceToRegion = 3;
  const float regionEdgeLength = 1e6;


  boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
  for (tie(vi, vi_end) = vertices(graph); vi != vi_end; ++vi)
  {
    if (pt == graph[*vi].pt)
    {
      return *vi;
    }
  }

  VertexDescriptor v = boost::add_vertex(graph);
  graph[v].pt = pt;
  CV_Assert(orientations.type() == CV_32FC1);
  graph[v].orientation = orientations.at<float>(pt.y, pt.x);


  CV_Assert(segmentation.type() == CV_32SC1);
  for (int dy = -maxDistanceToRegion; dy <= maxDistanceToRegion; ++dy)
  {
    for (int dx = -maxDistanceToRegion; dx <= maxDistanceToRegion; ++dx)
    {
      Point shiftedPt = pt + Point(dx, dy);
      if (!isPointInside(segmentation, shiftedPt))
      {
        continue;
      }

      int regionIndex = segmentation.at<int>(shiftedPt.y, shiftedPt.x);
      VertexDescriptor regionVertex = getRegionVertex(graph, regionIndex);
      bool isNew;
      EdgeDescriptor addedEdge;
      tie(addedEdge, isNew) = boost::add_edge(v, regionVertex, graph);
      if (isNew)
      {
        graph[addedEdge].length = regionEdgeLength;
      }
    }
  }

  return v;
}

void edges2graph(const cv::Mat &segmentation, const vector<Region> &regions, const cv::Mat &edges, Graph &graph)
{
  //TODO: move up
  const int medianIndex = 3;

  for (size_t i = 0; i < regions.size(); ++i)
  {
    VertexDescriptor v = boost::add_vertex(graph);
    graph[v].isRegion = true;
    graph[v].regionIndex = i;
    graph[v].pt = regions[i].getCenter();
  }

  Mat edgesMap = edges.clone();
  Mat orientations;
  computeEdgeOrientations(edgesMap, orientations, medianIndex);
  CV_Assert(orientations.type() == CV_32FC1);
  //TODO: check with NaNs and remove magic numbers

  edgesMap = edges.clone();
  edgesMap.setTo(0, ~((orientations >= -10.0) & (orientations <= 10.0)));

  vector<vector<Point> > contours;
  findContours(edgesMap, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
  for (size_t contourIndex = 0; contourIndex < contours.size(); ++contourIndex)
  {
    VertexDescriptor previousVertex = insertPoint(segmentation, orientations, contours[contourIndex][0], graph);

    for (size_t edgelIndex = 1; edgelIndex < contours[contourIndex].size(); ++edgelIndex)
    {
      VertexDescriptor currentVertex = insertPoint(segmentation, orientations, contours[contourIndex][edgelIndex], graph);
      EdgeDescriptor currentEdge;
      bool isNewEdge;
      tie(currentEdge, isNewEdge) = boost::add_edge(previousVertex, currentVertex, graph);
      if (isNewEdge)
      {
        //TODO: use better estimation
        graph[currentEdge].length = norm(graph[previousVertex].pt - graph[currentVertex].pt);
        graph[currentEdge].maximumAngle = fabs(graph[previousVertex].orientation - graph[currentVertex].orientation);
      }
      previousVertex = currentVertex;
    }
  }
}

cv::Point getNextPoint(cv::Point previous, cv::Point current)
{
  Point next = current + (current - previous);
  return next;
}

//TODO: use orientations of endpoints
//TODO: process adjacent regions when the shortest path has the single edgel
bool areRegionsOnTheSameSide(const cv::Mat &path, Point firstPathEdgel, Point lastPathEdgel, Point center_1, Point center_2)
{
  //TODO: move up
  float minDistanceToEndpoints = 1.8f;
  Mat dilatedPath;
  dilate(path, dilatedPath, Mat());
  vector<vector<Point> > contours;
  findContours(dilatedPath, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  CV_Assert(contours.size() <= 1);
  if (contours.size() == 0)
  {
    return true;
  }

  for (vector<Point>::iterator it = contours[0].begin(); it != contours[0].end();)
  {
    if (norm(*it - firstPathEdgel) < minDistanceToEndpoints || norm(*it - lastPathEdgel) < minDistanceToEndpoints)
    {
      it = contours[0].erase(it);
    }
    else
    {
      ++it;
    }
  }

  Mat boundaries(path.size(), CV_8UC1, Scalar(0));
  drawContours(boundaries, contours, -1, Scalar(255));
  contours.clear();
  findContours(boundaries, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  CV_Assert(contours.size() <= 2);
  if (contours.size() != 2)
  {
    return true;
  }

  //TODO: remove code duplication
  Mat firstBoundary(path.size(), CV_8UC1, Scalar(0));
  drawContours(firstBoundary, contours, 0, Scalar(255));
  Mat firstDT;
  distanceTransform(~firstBoundary, firstDT, CV_DIST_L2, CV_DIST_MASK_PRECISE);

  Mat secondBoundary(path.size(), CV_8UC1, Scalar(0));
  drawContours(secondBoundary, contours, 1, Scalar(255));
  Mat secondDT;
  distanceTransform(~secondBoundary, secondDT, CV_DIST_L2, CV_DIST_MASK_PRECISE);

  bool isFirst_1 = firstDT.at<float>(center_1) < secondDT.at<float>(center_1);
  bool isFirst_2 = firstDT.at<float>(center_2) < secondDT.at<float>(center_2);

  return (!(isFirst_1 ^ isFirst_2));
}

//TODO: block junctions
//TODO: extend to the cases when regions are not connected to each other
void GlassClassifier::estimateAffinities(const Graph &graph, size_t regionCount, cv::Size imageSize, int regionIndex, std::vector<float> &affinities)
{
  //TODO: move up
  const float regionEdgeLength = 1e6;

  vector<double> distances(boost::num_vertices(graph));
  vector<VertexDescriptor> predecessors(boost::num_vertices(graph));
  boost::dijkstra_shortest_paths(graph, getRegionVertex(graph, regionIndex),
        boost::weight_map(boost::get(&EdgeProperties::length, graph))
        .distance_map(make_iterator_property_map(distances.begin(), get(boost::vertex_index, graph)))
        .predecessor_map(&predecessors[0]));


  //TODO: use dynamic programming
  affinities.resize(regionCount);
  int affinityIndex = -1;
  boost::graph_traits<Graph>::vertex_iterator vi, vi_end;
  for (tie(vi, vi_end) = vertices(graph); vi != vi_end; ++vi)
  {
    if (!graph[*vi].isRegion)
    {
      continue;
    }
    ++affinityIndex;

    if (graph[*vi].regionIndex == regionIndex)
    {
      affinities[affinityIndex] = 0.0f;
      continue;
    }

    VertexDescriptor currentVertex = *vi;
    VertexDescriptor predecessorVertex = predecessors[currentVertex];

    if (distances[affinityIndex] > 3 * regionEdgeLength || currentVertex == predecessorVertex)
    {
      affinities[affinityIndex] = CV_PI;
      continue;
    }

    Mat path(imageSize, CV_8UC1, Scalar(0));
    float maxAngle = 0.0f;
    Point firstEdgel = graph[predecessorVertex].pt;
    Point lastEdgel;
    Point predecessorLocation = graph[predecessorVertex].pt;
    bool pathIsInvalid = false;
    while (predecessorVertex != currentVertex)
    {
      if (currentVertex != *vi && graph[currentVertex].isRegion)
      {
        maxAngle = CV_PI;
        pathIsInvalid = true;
        break;
      }

      Point currentLocation = graph[currentVertex].pt;
      CV_Assert(isPointInside(path, currentLocation));
      CV_Assert(isPointInside(path, predecessorLocation));
      if (!graph[currentVertex].isRegion && !graph[predecessorVertex].isRegion)
      {
        line(path, currentLocation, predecessorLocation, Scalar(255));
      }

      EdgeDescriptor edge;
      bool doesExist;
      tie(edge, doesExist) = boost::edge(currentVertex, predecessorVertex, graph);
      CV_Assert(doesExist);
      maxAngle = std::max(maxAngle, graph[edge].maximumAngle);

      lastEdgel = graph[currentVertex].pt;
      currentVertex = predecessorVertex;
      predecessorVertex = predecessors[currentVertex];
      predecessorLocation = graph[predecessorVertex].pt;
    }
    if (pathIsInvalid)
    {
      affinities[affinityIndex] = CV_PI;
      continue;
    }
    if (areRegionsOnTheSameSide(path, firstEdgel, lastEdgel, graph[*vi].pt, graph[currentVertex].pt))
    {
      affinities[affinityIndex] = maxAngle;
    }
    else
    {
      affinities[affinityIndex] = CV_PI;
    }
  }

  for (size_t i = 0; i < affinities.size(); ++i)
  {
    affinities[i] = 1.0 - affinities[i] / CV_PI;
    const float eps = 1e-4;
    CV_Assert(affinities[i] > -eps && affinities[i] < CV_PI + eps);
  }
}

void GlassClassifier::computeAllAffinities(const std::vector<Region> &regions, const Graph &graph, cv::Mat &affinities)
{
  affinities.create(regions.size(), regions.size(), CV_32FC1);
  affinities.setTo(-1);
  //TODO: use the Floyd-Warshall algorithm
  for (size_t regionIndex = 0; regionIndex < regions.size(); ++regionIndex)
  {
    vector<float> currentAffinities;
    estimateAffinities(graph, regions.size(), regions[0].getMask().size(), regionIndex, currentAffinities);
    CV_Assert(currentAffinities.size());
    Mat affinitiesMat = Mat(currentAffinities);
    Mat affinitiesFloat;
    affinitiesMat.convertTo(affinitiesFloat, CV_32FC1);
    Mat row = affinities.row(regionIndex);
    affinitiesFloat.reshape(1, 1).copyTo(row);
  }
}

void GlassClassifier::computeBoundaryPresences(const std::vector<Region> &regions, const cv::Mat &edges, cv::Mat &boundaryPresences)
{
  //TODO: move up
  const int maxDistanceToEdgel = 3; //iterations of dilation
  const float minIntersectionRatio = 0.05f;

  boundaryPresences.create(regions.size(), regions.size(), CV_8UC1);
  boundaryPresences.setTo(Scalar(0));
  vector<Mat> dilatedMasks(regions.size());
  for (size_t i = 0; i < regions.size(); ++i)
  {
    dilate(regions[i].getMask(), dilatedMasks[i], Mat(), Point(-1, -1), maxDistanceToEdgel);
  }

  for (size_t i = 0; i < regions.size(); ++i)
  {
    for (size_t j = 0; j < regions.size(); ++j)
    {
      if (i == j)
      {
        continue;
      }

      Mat intersectionMask = dilatedMasks[i] & dilatedMasks[j];
      int intersectionArea = countNonZero(intersectionMask);
      if (intersectionArea == 0)
      {
        continue;
      }

      Mat intersectionEdges = intersectionMask & edges;
      int edgelCount = countNonZero(intersectionEdges);
      float intersectionRatio = static_cast<float>(edgelCount) / intersectionArea;

      if (intersectionRatio > minIntersectionRatio)
      {
        boundaryPresences.at<uchar>(i, j) = 255;
      }
    }
  }
}

void GlassClassifier::computeAllDiscrepancies(const SegmentedImage &testImage, const cv::Mat &groundTruthMask, cv::Mat &discrepancies) const
{
  Mat samples;
  segmentedImage2samples(testImage, samples);
  vector<Region> regions = testImage.getRegions();

  discrepancies.create(regions.size(), regions.size(), CV_32FC1);
  for (size_t i = 0; i < regions.size(); ++i)
  {
    vector<float> labels(regions.size());
    for (size_t j = 0; j < regions.size(); ++j)
    {
      if (j == i)
      {
        labels[j] = 0.0f;
        continue;
      }
      Mat fullSample = Mat(samples.at<Vec4f>(i, j)).reshape(1, 1);
      CV_Assert(fullSample.rows == 1);
      CV_Assert(fullSample.cols == 4);
      CV_Assert(fullSample.channels() == 1);

//      ecaSample = ecaSample.mul(scalingSlope) + scalingIntercept;
//      dcaSample = dcaSample.mul(scalingSlope) + scalingIntercept;
//      labels[j] = normalizationSlope * svm->predict(dcaSample, true) + normalizationIntercept;
//      labels[j] = normalizationSlope * svm->predict(ecaSample, true) + normalizationIntercept;

//      Mat fullSample = mlData.samples.row(sampleIndex).mul(scalingSlope) + scalingIntercept;
      fullSample = fullSample.mul(scalingSlope) + scalingIntercept;
      labels[j] = normalizationSlope * svm.predict(fullSample, true) + normalizationIntercept;
    }
    Mat row = discrepancies.row(i);
    Mat(labels).reshape(1, 1).copyTo(row);
  }
}

void GlassClassifier::computeBoundaryStrength(const SegmentedImage &testImage, const cv::Mat &edges, const cv::Mat &groundTruthMask, const Graph &graph, float affinityWeight, cv::Mat &boundaryStrength) const
{
  //TODO: move up
  const int neighborDistance = 1;


  CV_Assert(affinityWeight >= 0.0f && affinityWeight <= 1.0f);
  //TODO: use lazy evaluations
  Mat affinities = getFromCache("affinities");
  vector<Region> regions = testImage.getRegions();
  if (affinities.empty())
  {
    computeAllAffinities(regions, graph, affinities);
    saveToCache("affinities", affinities);
  }

  Mat boundaryPresences = getFromCache("boundaryPresences");
  if (boundaryPresences.empty())
  {
    computeBoundaryPresences(regions, edges, boundaryPresences);
    saveToCache("boundaryPresences", boundaryPresences);
  }
  affinities.setTo(0, boundaryPresences);
#if 1
  Mat discrepancies = getFromCache("discrepancies");
  if (discrepancies.empty())
  {
    computeAllDiscrepancies(testImage, groundTruthMask, discrepancies);
    saveToCache("discrepancies", discrepancies);
  }
  CV_Assert(discrepancies.type() == CV_32FC1);
#endif

  CV_Assert(affinities.type() == CV_32FC1);

  Mat segmentation = testImage.getSegmentation();
  Mat pixelDiscrepancies(segmentation.size(), CV_32FC1, Scalar(0));
  Mat pixelAffinities(segmentation.size(), CV_32FC1);
  for (int i = 0; i < segmentation.rows; ++i)
  {
    for (int j = 0; j < segmentation.cols; ++j)
    {
      Point srcPt = Point(j, i);
      int srcRegionIndex = segmentation.at<int>(srcPt);
      float maxDiscrepancy = 0.0f;
      float maxAffinity = 0.0f;
      for (int dy = -neighborDistance; dy <= neighborDistance; ++dy)
      {
        for (int dx = -neighborDistance; dx <= neighborDistance; ++dx)
        {
          Point diffPt(dx, dy);
          Point pt = srcPt + diffPt;
          if (!isPointInside(segmentation, pt))
          {
            continue;
          }

          int regionIndex = segmentation.at<int>(pt);
          if (regionIndex != srcRegionIndex)
          {
            //TODO: what about symmetry
#if 1
            maxDiscrepancy = std::max(maxDiscrepancy, discrepancies.at<float>(srcRegionIndex, regionIndex));
#endif
            maxAffinity = std::max(maxAffinity, affinities.at<float>(srcRegionIndex, regionIndex));
          }
        }
      }
#if 1
      pixelDiscrepancies.at<float>(srcPt) = maxDiscrepancy;
#endif
      pixelAffinities.at<float>(srcPt) = maxAffinity;
    }
  }

  boundaryStrength = pixelDiscrepancies - affinityWeight * pixelAffinities;
}

void GlassClassifier::segmentedImage2samples(const SegmentedImage &segmentedImage, cv::Mat &samples)
{
  vector<Region> regions = segmentedImage.getRegions();
  samples.create(regions.size(), regions.size(), CV_32FC4);
  for (int i = 0; i < regions.size(); ++i)
  {
    for (int j = 0; j < regions.size(); ++j)
    {
      Mat ecaSample, dcaSample, fullSample;
      regions2samples(regions[i], regions[j], ecaSample, dcaSample, fullSample);
      CV_Assert(fullSample.cols == 4);
      samples.at<Vec4f>(i, j) = fullSample;
    }
  }
}

void GlassClassifier::segmentedImage2responses(const SegmentedImage &segmentedImage, const cv::Mat &groundTruthMask, cv::Mat &responses)
{
  //TODO: move up
  const float confidentLabelArea = 0.9f;
//  float confidentLabelArea = 0.6f;

  vector<Region> regions = segmentedImage.getRegions();
  vector<RegionLabel> regionLabels(regions.size());
  for (size_t i = 0; i < regions.size(); ++i)
  {
    int regionArea = countNonZero(regions[i].getMask() != 0);
    int glassArea = countNonZero(regions[i].getMask() & groundTruthMask);
    float glassAreaRatio = static_cast<float>(glassArea) / regionArea;
    if (glassAreaRatio > confidentLabelArea)
    {
      regionLabels[i] = GLASS;
    }
    else
    {
      if (glassAreaRatio < 1.0 - confidentLabelArea)
      {
        regionLabels[i] = BACKGROUND;
      }
      else
      {
        regionLabels[i] = NOT_VALID;
      }
    }
  }

  responses.create(regions.size(), regions.size(), CV_32SC1);
  for (size_t i = 0; i < regions.size(); ++i)
  {
    Mat dilatedMask;
    dilate(regions[i].getMask(), dilatedMask, Mat());
    for (size_t j = 0; j < regions.size(); ++j)
    {
      int currentLabel;
      if (regionLabels[i] == NOT_VALID || regionLabels[j] == NOT_VALID || i == j)
      {
        currentLabel = INVALID;
      }
      else
      {
        if (regionLabels[i] ^ regionLabels[j])
        {
          currentLabel = GLASS_COVERED;
        }
        else
        {
          currentLabel = THE_SAME;
        }
      }
//        if (countNonZero(dilatedMask & regions[j].getMask()) == 0)
//        {
//          continue;
//        }

      responses.at<int>(i, j) = currentLabel;
    }
  }
}

void GlassClassifier::segmentedImage2MLData(const SegmentedImage &segmentedImage, const cv::Mat &groundTruthMask, bool withAllSymmetricSamples, MLData &mlData)
{
  //TODO: move up
  const float maxSampleDistance = 0.1f;

  Mat fullTrainingData;
  vector<int> trainingLabelsVec;
  Mat samples;
  segmentedImage2samples(segmentedImage, samples);
  Mat responses;
  segmentedImage2responses(segmentedImage, groundTruthMask, responses);

  vector<Region> regions = segmentedImage.getRegions();
  for (size_t i = 0; i < regions.size(); ++i)
  {
    Mat dilatedMask;
    dilate(regions[i].getMask(), dilatedMask, Mat());
    for (size_t j = i + 1; j < regions.size(); ++j)
    {
      int currentResponse = responses.at<int>(i, j);
      if (currentResponse == INVALID)
      {
        continue;
      }
      CV_Assert(responses.at<int>(j, i) != INVALID);

      Mat currentSample = Mat(samples.at<Vec4f>(i, j)).reshape(1, 1);
      CV_Assert(currentSample.rows == 1);
      CV_Assert(currentSample.cols == 4);
      CV_Assert(currentSample.channels() == 1);
      Mat symmetricSample = Mat(samples.at<Vec4f>(j, i)).reshape(1, 1);
      fullTrainingData.push_back(currentSample);
      trainingLabelsVec.push_back(currentResponse);

      if (norm(currentSample - symmetricSample) > maxSampleDistance)
      {
        //TODO: is it a correct way to process such cases?
        fullTrainingData.push_back(symmetricSample);
        trainingLabelsVec.push_back(currentResponse);
      }
    }
  }

  mlData.samples = fullTrainingData;
  mlData.responses = Mat(trainingLabelsVec).clone();
  CV_Assert(mlData.isValid());
}

void GlassClassifier::test(const SegmentedImage &testImage, const cv::Mat &groundTruthMask, cv::Mat &boundaryStrength) const
{
  //TODO: move up
  const float affinityWeight = 0.0f;
  const double cannyThreshold1 = 100.0;
  const double cannyThreshold2 = 50.0;

  Mat grayscaleImage;
  cvtColor(testImage.getOriginalImage(), grayscaleImage, CV_BGR2GRAY);
  Mat edges;
  Canny(grayscaleImage, edges, cannyThreshold1, cannyThreshold2);
  imshow("edges", edges.clone());
  Graph graph;
  edges2graph(testImage.getSegmentation(), testImage.getRegions(), edges, graph);

  computeBoundaryStrength(testImage, edges, groundTruthMask, graph, affinityWeight, boundaryStrength);
  //TODO: do you need this?
  boundaryStrength.setTo(0, boundaryStrength < 0);

  Mat strengthVisualization = boundaryStrength.clone();
  strengthVisualization -= normalizationIntercept;
  strengthVisualization.setTo(0, strengthVisualization < 0);
  imshow("boundary", strengthVisualization);
//  imshow("boundary", boundaryStrength);
  waitKey();
}

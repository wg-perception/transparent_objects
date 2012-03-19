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

//#define VISUALIZE_SEGMENTATION
//#define DEBUG_ML

enum RegionLabel {GLASS, GLASS_BACKGROUND, OTHER_BACKGROUND, REGION_GROUND_TRUTH_INVALID, REGION_COMPLETELY_INVALID};

EdgeProperties::EdgeProperties()
{
  length = 0.0f;
  maximumAngle = 0.0f;
}

float getMinToMaxRatio(float a, float b)
{
  const float eps = 1e-4;
  float ratio = a < b ? a / (b + eps) : b / (a + eps);
  return ratio;
}

void computeContrastDistance(const Region &region_1, const Region &region_2, float &rmsDistance, float &michelsonDistance, float &robustMichelsonDistance)
{
  float rms_1 = region_1.getRMSContrast();
  float rms_2 = region_2.getRMSContrast();
  rmsDistance = getMinToMaxRatio(rms_1, rms_2);

  float michelson_1 = region_1.getMichelsonContrast();
  float michelson_2 = region_2.getMichelsonContrast();
  michelsonDistance = getMinToMaxRatio(michelson_1, michelson_2);

  float robustMichelson_1 = region_1.getRobustMichelsonContrast();
  float robustMichelson_2 = region_2.getRobustMichelsonContrast();
  robustMichelsonDistance = getMinToMaxRatio(robustMichelson_1, robustMichelson_2);
}

void computeColorSimilarity(const Region &region_1, const Region &region_2, float &distance)
{
  Mat hist_1 = region_1.getColorHistogram();
  Mat hist_2 = region_2.getColorHistogram();

  //TODO: experiment with different distances
  distance = norm(hist_1 - hist_2);
}

void computeMedianColorDistance(const Region &region_1, const Region &region_2, float &distance)
{
  Vec3i color_1 = region_1.getMedianColor();
  Vec3i color_2 = region_2.getMedianColor();

  distance = norm(color_1 - color_2);
}

void computeTextureDistortion(const Region &region_1, const Region &region_2, float &distance)
{
  Mat hist_1 = region_1.getTextonHistogram();
  Mat hist_2 = region_2.getTextonHistogram();

  if (hist_2.size() != hist_1.size())
  {
    hist_2 = hist_2.t();
  }

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

void computeDepthDistance(const Region &region_1, const Region &region_2, float &depthDistance)
{
  float depthRatio_1 = region_1.getDepthRatio();
  float depthRatio_2 = region_2.getDepthRatio();

  depthDistance = fabs(depthRatio_1 - depthRatio_2);
}


void GlassClassifier::regions2samples(const Region &region_1, const Region &region_2, cv::Mat &fullSample)
{
  float colorDistance;
  computeColorSimilarity(region_1, region_2, colorDistance);
  float medianColorDistance;
  computeMedianColorDistance(region_1, region_2, medianColorDistance);
  float slope, intercept;
  computeOverlayConsistency(region_1, region_2, slope, intercept);
  float textureDistance;
  computeTextureDistortion(region_1, region_2, textureDistance);
  float depthDistance;
  computeDepthDistance(region_1, region_2, depthDistance);


  float rmsDistance, michelsonDistance, robustMichelsonDistance;
  computeContrastDistance(region_1, region_2, rmsDistance, michelsonDistance, robustMichelsonDistance);

//  fullSample = (Mat_<float>(1, Sample::channels) << slope, intercept, colorDistance, textureDistance, medianColorDistance);

  fullSample = (Mat_<float>(1, Sample::channels) << rmsDistance, michelsonDistance, robustMichelsonDistance, slope, intercept, colorDistance, textureDistance, medianColorDistance, depthDistance);
//  fullSample = (Mat_<float>(1, Sample::channels) << rmsDistance, michelsonDistance, robustMichelsonDistance, slope, intercept, colorDistance, textureDistance, medianColorDistance);
//  fullSample = (Mat_<float>(1, Sample::channels) << depthDistance);
//  fullSample = (Mat_<float>(1, Sample::channels) << rmsDistance, robustMichelsonDistance, colorDistance, textureDistance, medianColorDistance);
}

void normalizeTrainingData(cv::Mat &trainingData, cv::Mat &scalingSlope, cv::Mat &scalingIntercept)
{
  Mat maxData, minData;
  reduce(trainingData, maxData, 0, CV_REDUCE_MAX);
  reduce(trainingData, minData, 0, CV_REDUCE_MIN);
//  Mat scalingSlope, scalingIntercept;
  scalingSlope = 2.0 / (maxData - minData);
  scalingIntercept = -scalingSlope.mul(minData) - 1.0;
//  scalingSlope = 1.0 / (maxData - minData);
//  scalingIntercept = -scalingSlope.mul(minData);
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

void GlassClassifier::computeNormalizationParameters(const MLData &trainingData)
{
  CV_Assert(trainingData.getSamplesCount() > 0);

  Mat zeroSample = trainingData.samples.row(0);
  int zeroPredictedLabel = cvRound(svm.predict(zeroSample));
  float zeroDistance = svm.predict(zeroSample, true);
  int negativeLabel = zeroDistance < 0 ? zeroPredictedLabel : 1 - zeroPredictedLabel;
  cout << "Negative label: " << negativeLabel << endl;

  normalizationSlope = 1.0;
  normalizationIntercept = 0.0;
  Mat trainingConfidences;
  predict(trainingData, trainingConfidences);
  double minSVMDistance = 0.0f;
  double maxSVMDistance = 0.0f;
  minMaxLoc(trainingConfidences, &minSVMDistance, &maxSVMDistance);

  float spread = maxSVMDistance - minSVMDistance;
  const float eps = 1e-2;
  CV_Assert(spread > eps);
  normalizationSlope = 1.0 / spread;
  normalizationIntercept = -minSVMDistance * normalizationSlope;

  if (negativeLabel == GLASS_COVERED)
  {
    normalizationSlope = -normalizationSlope;
    normalizationIntercept = -normalizationIntercept + 1;
  }

  Mat predictedLabels = trainingConfidences >= 0;
  predictedLabels.setTo(GLASS_COVERED, predictedLabels == 255);
  if (negativeLabel == GLASS_COVERED)
  {
    predictedLabels = 1 - predictedLabels;
  }
  float trainingError;
  Mat confusionMatrix;
  trainingData.evaluate(predictedLabels, trainingError, confusionMatrix);
  cout << "samples: " << trainingData.getSamplesCount() << endl;
  cout << "training error: " << trainingError << endl;
  cout << "confusion matrix: " << endl << confusionMatrix << endl;

  //TODO: remove
  {
    Mat trainingConfidences;
    predict(trainingData, trainingConfidences);
    Mat trainingPredictions = (trainingConfidences >= normalizationIntercept);
    trainingPredictions.setTo(GLASS_COVERED, trainingPredictions == 255);
    int missclassifiedSamples = countNonZero(trainingPredictions != predictedLabels);
    //TODO: move up
    const int maxMissclassifiedSamplesCount = 3;
    CV_Assert(missclassifiedSamples < maxMissclassifiedSamplesCount);
  }
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
  int result = mask.empty() ? samples.rows : countNonZero(mask);
  return result;
}

void MLData::push_back(const MLData &mlData)
{
  samples.push_back(mlData.samples);
  responses.push_back(mlData.responses);
}

int MLData::computeClassesCount() const
{
  double minLabel, maxLabel;
  minMaxLoc(responses, &minLabel, &maxLabel);
  return (cvRound(maxLabel) - cvRound(minLabel) + 1);
}

void MLData::evaluate(const cv::Mat &predictedLabels, float &error, cv::Mat &confusionMatrix) const
{
  int classesCount = computeClassesCount();
  confusionMatrix.create(classesCount, classesCount, CV_32SC1);
  Mat currentMask = mask.empty() ? Mat(responses.size(), CV_8UC1, Scalar(255)) : mask;
  for (int i = 0; i < classesCount; ++i)
  {
    for (int j = 0; j < classesCount; ++j)
    {
      confusionMatrix.at<int>(i, j) = countNonZero((responses == i) & (predictedLabels == j) & currentMask);
    }
  }

  float accuracy = sum(confusionMatrix.diag())[0] / getSamplesCount();
  error = 1.0 - accuracy;
}

void MLData::write(const std::string name) const
{
  CV_Assert(isValid());

  {
    string wekaFilename = name + "_weka.csv";
    std::ofstream fout(wekaFilename.c_str());
    for (int i = 0; i < getDimensionality(); ++i)
    {
      fout << "<- " << i << " ->, ";
    }
    fout << "label\n";
    for (int i = 0; i < samples.rows; ++i)
    {
      for (int j = 0; j < samples.cols; ++j)
      {
        fout << samples.at<float>(i, j) << ", ";
      }

      fout << "[" << responses.at<int>(i) << "]\n";
    }
    fout.close();
  }

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
      fout << '\n';
    }
    fout.close();
  }
}

void MLData::removeMaskedOutSamples()
{
  CV_Assert(!mask.empty());
  int newSamplesCount = getSamplesCount();
  Mat newSamples(newSamplesCount, samples.cols, samples.type());
  Mat newResponses(newSamplesCount, responses.cols, responses.type());

  int newIndex = 0;
  for (int i = 0; i < mask.rows; ++i)
  {
    if (mask.at<uchar>(i) == 0)
    {
      continue;
    }

    Mat currentNewSample = newSamples.row(newIndex);
    samples.row(i).copyTo(currentNewSample);

    Mat currentNewResponse = newResponses.row(newIndex);
    responses.row(i).copyTo(currentNewResponse);
    ++newIndex;
  }
  samples = newSamples;
  responses = newResponses;
  mask = Mat();
}

GlassClassifier::GlassClassifier(const GlassClassifierParams &_params)
{
  params = _params;
}

void GlassClassifier::train(const std::string &trainingFilesList, const std::string &groundTruthFilesList, const std::string &depthMatFilesList, const cv::Mat &registrationMask)
{
  vector<string> trainingGroundTruhFiles;
  readLinesInFile(groundTruthFilesList, trainingGroundTruhFiles);

  vector<string> trainingFiles;
  readLinesInFile(trainingFilesList, trainingFiles);

  vector<string> depthMatFiles;
  if (!depthMatFilesList.empty())
  {
    readLinesInFile(depthMatFilesList, depthMatFiles);
  }

  const size_t imageCount = trainingGroundTruhFiles.size();
  CV_Assert(trainingFiles.size() == imageCount);

  MLData trainingData;
  for (size_t imageIndex = 0; imageIndex < imageCount; ++imageIndex)
  {
    if (trainingFiles[imageIndex][0] == '#' || trainingGroundTruhFiles[imageIndex][0] == '#')
    {
      continue;
    }

    cout << "training image #" << imageIndex << endl;

    Mat groundTruthMask = imread(trainingGroundTruhFiles[imageIndex], CV_LOAD_IMAGE_GRAYSCALE);
    CV_Assert(!groundTruthMask.empty());

    SegmentedImage segmentedImage;
    segmentedImage.read(trainingFiles[imageIndex]);
    if (!depthMatFiles.empty())
    {
      //TODO: use TODBaseImporter
      FileStorage depthFs(depthMatFiles[imageIndex], FileStorage::READ);
      Mat depthMat;
      depthFs["depth_image"] >> depthMat;
      depthFs.release();
      CV_Assert(!depthMat.empty());

      Mat invalidDepthMask = getInvalidDepthMask(depthMat, registrationMask);
      segmentedImage.setDepth(invalidDepthMask);
    }

#ifdef VISUALIZE_SEGMENTATION
    segmentedImage.showSegmentation("train segmentation");
    segmentedImage.showBoundaries("train boundaries");
#endif

    MLData currentMLData;
    //TODO: move up parameter
//    bool useOnlyAdjacentRegions = countNonZero(groundTruthMask == 255) == 0;
//    segmentedImage2MLData(segmentedImage, groundTruthMask, useOnlyAdjacentRegions, currentMLData);
    segmentedImage2MLData(segmentedImage, groundTruthMask, false, currentMLData);

    CV_Assert(currentMLData.isValid());
    trainingData.push_back(currentMLData);
  }
  CV_Assert(trainingData.isValid());
  int randomizer = trainingData.getSamplesCount() / params.targetNegativeSamplesCount;
  CV_Assert(trainingData.mask.empty());
  CV_Assert(trainingData.responses.type() == CV_32SC1);
  trainingData.mask = Mat(trainingData.responses.size(), CV_8UC1, Scalar(255));
  for (int i = 0; i < trainingData.responses.rows; ++i)
  {
    if (trainingData.responses.at<int>(i) == GLASS_COVERED)
    {
      continue;
    }

    if (rand() % randomizer != 0)
    {
      trainingData.mask.at<uchar>(i) = 0;
    }
  }
  trainingData.removeMaskedOutSamples();


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
  normalizeTrainingData(trainingData.samples, fullScalingSlope, fullScalingIntercept);
  scalingSlope = fullScalingSlope;
  scalingIntercept = fullScalingIntercept;
  cout << "scaling slope: " << scalingSlope << endl;
  cout << "scaling intercept: " << scalingIntercept << endl;

//  CV_Assert(trainingLabels.rows == ecaTrainingData.rows);
//  CV_Assert(trainingLabels.cols == 1);
//  CV_Assert(trainingLabels.type() == CV_32SC1);

  CvSVMParams svmParams;
  //TODO: move up
  svmParams.svm_type = CvSVM::C_SVC;
//  svmParams.C = 128.0;

  svmParams.C = 32.0;
//  svmParams.svm_type = CvSVM::NU_SVC;

//  svmParams.nu = 0.1;
//  svmParams.nu = 0.01;
//  svmParams.gamma = 8.0;
  svmParams.gamma = 0.5;
  svmParams.kernel_type = CvSVM::RBF;
  svmParams.term_crit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100000, 0.1);

//  cout << dcaTrainingData << endl;
//  saveMLData("ecaData.csv", ecaTrainingData, trainingLabels);
//  saveMLData("dcaData.csv", dcaTrainingData, trainingLabels);
//#ifdef DEBUG_ML
  trainingData.write("fullTrainingData");
//#endif

//  cout << ecaTrainingData << endl;
//  cout << "ecaTrainingData size: " << ecaTrainingData.rows << " x " << ecaTrainingData.cols << endl;
//  cout << "dcaTrainingData size: " << dcaTrainingData.rows << " x " << dcaTrainingData.cols << endl;
//  cout << "fullTrainingData size: " << fullTrainingData.rows << " x " << fullTrainingData.cols << endl;
  cout << "The same: " << countNonZero(trainingData.responses == THE_SAME) << endl;
  cout << "Glass covered: " << countNonZero(trainingData.responses == GLASS_COVERED) << endl;
  cout << "Completely invalid: " << countNonZero(trainingData.responses == COMPLETELY_INVALID) << endl;
  cout << "Ground truth invalid: " << countNonZero(trainingData.responses == GROUND_TRUTH_INVALID) << endl;

  cout << "training...  " << std::flush;

//  bool isTrained = svm.train(dcaTrainingData, trainingLabels, Mat(), Mat(), svmParams);
//  bool isTrained = svm.train(ecaTrainingData, trainingLabels, Mat(), Mat(), svmParams);
  bool isTrained = svm.train(trainingData.samples, trainingData.responses, Mat(), Mat(), svmParams);
//  bool isTrained = svm.train_auto(trainingData.samples, trainingData.responses, Mat(), Mat(), svmParams);
//  bool isTrained = svm.train_auto(ecaTrainingData, trainingLabels, Mat(), Mat(), svmParams);
//  bool isTrained = svm.train_auto(dcaTrainingData, trainingLabels, Mat(), Mat(), svmParams);
  cout << "done: " << isTrained << endl;
  CvSVMParams optimalParams = svm.get_params();
  cout << "C: " << optimalParams.C << endl;
  cout << "nu: " << optimalParams.nu << endl;
  cout << "gamma: " << optimalParams.gamma << endl;


  computeNormalizationParameters(trainingData);

//  getNormalizationParameters(&svm, ecaTrainingData, trainingLabelsVec, normalizationSlope, normalizationIntercept);
//  getNormalizationParameters(&svm, dcaTrainingData, trainingLabelsVec, normalizationSlope, normalizationIntercept);

  cout << "normalization slope: " << normalizationSlope << endl;
  cout << "normalization intercept: " << normalizationIntercept << endl;
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

void GlassClassifier::predict(const MLData &data, cv::Mat &confidences) const
{
  cout << "predicting... " << std::flush;
  confidences.create(data.samples.rows, 1, CV_32FC1);
  CV_Assert(data.mask.empty() || (data.mask.type() == CV_8UC1 && data.mask.rows == data.samples.rows));

#pragma omp parallel for schedule(dynamic, 100)
  for (int i = 0; i < data.samples.rows; ++i)
  {
    if (!data.mask.empty() && data.mask.at<uchar>(i) == 0)
    {
      continue;
    }
    Mat sample = data.samples.row(i);
    confidences.at<float>(i) = normalizationSlope * svm.predict(sample, true) + normalizationIntercept;
  }
  cout << "done" << std::endl;
}

void GlassClassifier::computeAllDiscrepancies(const SegmentedImage &testImage, const cv::Mat &groundTruthMask, cv::Mat &discrepancies, std::vector<bool> &isRegionValid) const
{
  Mat pairwiseSamples;
  segmentedImage2pairwiseSamples(testImage, pairwiseSamples, scalingSlope, scalingIntercept);
  Mat pairwiseResponses;
//  segmentedImage2pairwiseResponses(testImage, groundTruthMask, true, pairwiseResponses);
  segmentedImage2pairwiseResponses(testImage, groundTruthMask, false, pairwiseResponses);

  Mat samplesMask = (pairwiseResponses != COMPLETELY_INVALID);
  isRegionValid.resize(samplesMask.rows);
  for (int i = 0; i < samplesMask.rows; ++i)
  {
    Mat row = samplesMask.row(i);
    int invalidLabelsCount = countNonZero(row == 0);
    isRegionValid[i] = (invalidLabelsCount != samplesMask.cols);
  }

  MLData testData;
  testData.samples = pairwiseSamples.reshape(1, pairwiseSamples.total());
  testData.mask = samplesMask.reshape(1, samplesMask.total());
  testData.responses = pairwiseResponses.reshape(1, pairwiseResponses.total());

  Mat testConfidences;
  predict(testData, testConfidences);
  testConfidences.setTo(-std::numeric_limits<float>::max(), ~testData.mask);
  discrepancies = testConfidences.reshape(1, pairwiseResponses.rows);
  Mat diag = discrepancies.diag();
  diag.setTo(-std::numeric_limits<float>::max());

  Mat predictedLabels = testConfidences > normalizationIntercept;
  predictedLabels.setTo(GLASS_COVERED, predictedLabels == 255);
  float error;
  Mat confusionMatrix;
  testData.evaluate(predictedLabels, error, confusionMatrix);
  cout << "samples: " << testData.getSamplesCount() << endl;
  cout << "test error: " << error << endl;
  cout << "confusion matrix: " << confusionMatrix << endl;

#ifdef DEBUG_ML
  testData.write("testData");
#endif
/*
  {
    MLData testDataInTrainFormat;
    segmentedImage2MLData(testImage, groundTruthMask, false, testDataInTrainFormat);

    testDataInTrainFormat.write("testDataInTrainFormat");
    Mat confidences;
    predict(testDataInTrainFormat, confidences);
    Mat predictedLabels = confidences > normalizationIntercept;
    predictedLabels.setTo(GLASS_COVERED, predictedLabels == 255);
    float error;
    Mat confusionMatrix;
    testDataInTrainFormat.evaluate(predictedLabels, error, confusionMatrix);
    cout << "samples: " << testDataInTrainFormat.getSamplesCount() << endl;
    cout << "test error: " << error << endl;
    cout << "confusion matrix: " << confusionMatrix << endl;
  }
*/
}

void GlassClassifier::computeBoundaryStrength(const SegmentedImage &testImage, const cv::Mat &edges, const cv::Mat &groundTruthMask, const Graph &graph, float affinityWeight, cv::Mat &boundaryStrength) const
{
  //TODO: move up
  const int neighborDistance = 1;


  CV_Assert(affinityWeight >= 0.0f && affinityWeight <= 1.0f);

  vector<Region> regions = testImage.getRegions();

#ifdef USE_AFFINITY
  //TODO: use lazy evaluations
  Mat affinities = getFromCache("affinities");
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

  CV_Assert(affinities.type() == CV_32FC1);
#endif

#if 1
  Mat discrepancies = getFromCache("discrepancies");
  vector<bool> isRegionValid;
  if (discrepancies.empty())
  {
    computeAllDiscrepancies(testImage, groundTruthMask, discrepancies, isRegionValid);
    saveToCache("discrepancies", discrepancies);
  }
  CV_Assert(discrepancies.type() == CV_32FC1);
#endif


  Mat segmentation = testImage.getSegmentation();
  Mat pixelDiscrepancies(segmentation.size(), CV_32FC1, Scalar(0));
#ifdef USE_AFFINITY
  Mat pixelAffinities(segmentation.size(), CV_32FC1);
#endif
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
#ifdef USE_AFFINITY
            maxAffinity = std::max(maxAffinity, affinities.at<float>(srcRegionIndex, regionIndex));
#endif
          }
        }
      }
#if 1
      pixelDiscrepancies.at<float>(srcPt) = maxDiscrepancy;
#endif
#ifdef USE_AFFINITY
      pixelAffinities.at<float>(srcPt) = maxAffinity;
#endif
    }
  }

#ifdef USE_AFFINITY
  boundaryStrength = pixelDiscrepancies - affinityWeight * pixelAffinities;
#else
  boundaryStrength = pixelDiscrepancies;
#endif

  discrepancies.setTo(0.0, discrepancies < normalizationIntercept);

  Mat summedDiscrepancies;
//  reduce(discrepancies, summedDiscrepancies, 1, CV_REDUCE_AVG);
  reduce(discrepancies, summedDiscrepancies, 1, CV_REDUCE_SUM);
  Mat neighboredSummedDiscrepances = summedDiscrepancies.clone();

  CV_Assert(regions.size() == isRegionValid.size());
  for (size_t i = 0; i < isRegionValid.size(); ++i)
  {
    if (isRegionValid[i])
    {
      continue;
    }

    float maxDiscrepancy = -std::numeric_limits<float>::max();
    for (size_t j = 0; j < regions.size(); ++j)
    {
      if (!testImage.areRegionsAdjacent(i, j))
      {
        continue;
      }
      CV_Assert(summedDiscrepancies.type() == CV_32FC1);
      const float eps = 1e-4;
//      maxDiscrepancy = std::max(maxDiscrepancy, summedDiscrepancies.at<float>(j)) - eps;
      maxDiscrepancy = std::max(maxDiscrepancy, summedDiscrepancies.at<float>(j) - eps);
    }

//    summedDiscrepancies.at<float>(i) = maxDiscrepancy;
    neighboredSummedDiscrepances.at<float>(i) = maxDiscrepancy;
  }
  summedDiscrepancies = neighboredSummedDiscrepances.clone();


  Mat sortedIndices;
  sortIdx(summedDiscrepancies, sortedIndices, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);
  Mat bestRegionsImage(pixelDiscrepancies.size(), CV_8UC1, Scalar(0));
//  int bestRegionsCount = 40;

  //TODO: move up
  int bestRegionsCount = 100;
  Mat regionsStrength(pixelDiscrepancies.size(), CV_32FC1);
  for (int i = 0; i < sortedIndices.rows; ++i)
  {
    int currentIndex = sortedIndices.at<int>(i);
    regionsStrength.setTo(summedDiscrepancies.at<float>(currentIndex), regions[currentIndex].getMask());
#ifdef VISUALIZE_SEGMENTATION
//    cout << "depth: " << regions[currentIndex].getDepthRatio() << endl;
    if (i < bestRegionsCount)
    {
      bestRegionsImage.setTo(255 - i * 255.0 / bestRegionsCount, regions[currentIndex].getMask());
    }
//    imshow("bestRegions", bestRegionsImage);
//    waitKey();
//    cout << summedDiscrepancies.at<float>(currentIndex) << endl;
#endif
  }
#ifdef VISUALIZE_SEGMENTATION
  imshow("bestRegions", bestRegionsImage);
#endif

  boundaryStrength = regionsStrength;
}

void GlassClassifier::segmentedImage2pairwiseSamples(const SegmentedImage &segmentedImage, cv::Mat &samples, const cv::Mat &scalingSlope, const cv::Mat &scalingIntercept)
{
  vector<Region> regions = segmentedImage.getRegions();
  samples.create(regions.size(), regions.size(), Sample::type);
  Mat mlSamples;
  for (int i = 0; i < regions.size(); ++i)
  {
    for (int j = 0; j < regions.size(); ++j)
    {
      Mat fullSample;
      regions2samples(regions[i], regions[j], fullSample);
      if (!scalingSlope.empty() && !scalingIntercept.empty())
      {
        fullSample = fullSample.mul(scalingSlope) + scalingIntercept;
      }
      samples.at<Sample>(i, j) = fullSample;
    }
  }
}

bool isRegionLabeled(const Region &region, int regionArea, const cv::Mat &mask, float confidentLabelArea)
{
  int labelArea = countNonZero(region.getMask() & mask);
  float areaRatio = static_cast<float>(labelArea) / regionArea;
  return (areaRatio > confidentLabelArea);
}

void GlassClassifier::segmentedImage2pairwiseResponses(const SegmentedImage &segmentedImage, const cv::Mat &groundTruthMask, bool useOnlyAdjacentRegions, cv::Mat &responses)
{
  CV_Assert(!groundTruthMask.empty());
  //TODO: move up
  const float confidentLabelArea = 0.7f;
//  const int minErodedArea = 11;

  const int minErodedArea = 16;
//  const int maxDistanceBetweenRegions = 100;
  const int maxDistanceBetweenRegions = 50;
//  float confidentLabelArea = 0.6f;

  const uchar glassBackgroundLabel = 0;
  const uchar otherBackgroundLabel = 127;
  const uchar glassLabel = 255;

  vector<Region> regions = segmentedImage.getRegions();
  vector<RegionLabel> regionLabels(regions.size());
  int invalidRegionsCount = 0;
  Mat invalidRegionsImage(groundTruthMask.size(), CV_8UC1, Scalar(0));
  Mat glassBackgroundMask = groundTruthMask == glassBackgroundLabel;
  Mat otherBackgroundMask = groundTruthMask == otherBackgroundLabel;
  Mat glassMask = groundTruthMask == glassLabel;
  for (size_t i = 0; i < regions.size(); ++i)
  {
    regionLabels[i] = REGION_COMPLETELY_INVALID;
    if (countNonZero(regions[i].getErodedMask()) < minErodedArea)
    {
      ++invalidRegionsCount;
      invalidRegionsImage.setTo(255, regions[i].getMask());
      continue;
    }

    int regionArea = countNonZero(regions[i].getMask() != 0);

    if (isRegionLabeled(regions[i], regionArea, glassBackgroundMask, confidentLabelArea))
    {
      regionLabels[i] = GLASS_BACKGROUND;
    }
    if (isRegionLabeled(regions[i], regionArea, otherBackgroundMask, confidentLabelArea))
    {
      regionLabels[i] = OTHER_BACKGROUND;
    }
    if (isRegionLabeled(regions[i], regionArea, glassMask, confidentLabelArea))
    {
      regionLabels[i] = GLASS;
    }

    if (regionLabels[i] == REGION_COMPLETELY_INVALID)
    {
      regionLabels[i] = REGION_GROUND_TRUTH_INVALID;
      ++invalidRegionsCount;
      invalidRegionsImage.setTo(255, regions[i].getMask());
    }
  }
  cout << "invalid regions: " << invalidRegionsCount << endl;
#ifdef VISUALIZE_SEGMENTATION
  imshow("invalid regions", invalidRegionsImage);
  waitKey(100);
#endif

  responses.create(regions.size(), regions.size(), CV_32SC1);
  for (size_t i = 0; i < regions.size(); ++i)
  {
    for (size_t j = 0; j < regions.size(); ++j)
    {
      int currentLabel;
      if (regionLabels[i] == REGION_COMPLETELY_INVALID || regionLabels[j] == REGION_COMPLETELY_INVALID || i == j ||
          useOnlyAdjacentRegions && !segmentedImage.areRegionsAdjacent(i, j) ||
          norm(regions[i].getCenter() - regions[j].getCenter()) > maxDistanceBetweenRegions)
      {
        currentLabel = COMPLETELY_INVALID;
        responses.at<int>(i, j) = currentLabel;
        continue;
      }

      if (regionLabels[i] == REGION_GROUND_TRUTH_INVALID || regionLabels[j] == REGION_GROUND_TRUTH_INVALID || i == j ||
          regionLabels[i] == GLASS && regionLabels[j] == GLASS)
      {
        currentLabel = GROUND_TRUTH_INVALID;
        responses.at<int>(i, j) = currentLabel;
        continue;
      }

      if (regionLabels[i] == GLASS || regionLabels[j] == GLASS)
      {
        if (regionLabels[i] == GLASS_BACKGROUND || regionLabels[j] == GLASS_BACKGROUND)
        {
          currentLabel = GLASS_COVERED;
        }
      }
      if (regionLabels[i] == regionLabels[j])
      {
        currentLabel = THE_SAME;
      }

      responses.at<int>(i, j) = currentLabel;
    }
  }
}

void GlassClassifier::segmentedImage2MLData(const SegmentedImage &segmentedImage, const cv::Mat &groundTruthMask, bool useOnlyAdjacentRegions, MLData &mlData)
{
  //TODO: move up
  const float maxSampleDistance = 0.1f;

  Mat fullTrainingData;
  vector<int> trainingLabelsVec;
  Mat samples;
  segmentedImage2pairwiseSamples(segmentedImage, samples);
  Mat responses;
  segmentedImage2pairwiseResponses(segmentedImage, groundTruthMask, useOnlyAdjacentRegions, responses);

  vector<Region> regions = segmentedImage.getRegions();
  for (size_t i = 0; i < regions.size(); ++i)
  {
    for (size_t j = i + 1; j < regions.size(); ++j)
    {
      int currentResponse = responses.at<int>(i, j);
      if (currentResponse == COMPLETELY_INVALID || currentResponse == GROUND_TRUTH_INVALID)
      {
        continue;
      }
      CV_Assert(! (useOnlyAdjacentRegions && responses.at<int>(i, j) == GLASS_COVERED));

      CV_Assert(responses.at<int>(j, i) != COMPLETELY_INVALID || responses.at<int>(j, i) != GROUND_TRUTH_INVALID);

      Mat currentSample = Mat(samples.at<Sample>(i, j)).reshape(1, 1);
      CV_Assert(currentSample.rows == 1);
      CV_Assert(currentSample.cols == Sample::channels);
      CV_Assert(currentSample.channels() == 1);
      Mat symmetricSample = Mat(samples.at<Sample>(j, i)).reshape(1, 1);
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
#ifdef VISUALIZE_SEGMENTATION
  imshow("edges", edges.clone());
#endif
  Graph graph;

#ifdef USE_AFFINITY
  edges2graph(testImage.getSegmentation(), testImage.getRegions(), edges, graph);
#endif

  computeBoundaryStrength(testImage, edges, groundTruthMask, graph, affinityWeight, boundaryStrength);
  //TODO: do you need this?
  boundaryStrength.setTo(0, boundaryStrength < 0);

  Mat strengthVisualization = boundaryStrength.clone();
  strengthVisualization -= normalizationIntercept;
  strengthVisualization.setTo(0, strengthVisualization < 0);

#ifdef VISUALIZE_SEGMENTATION
  imshow("boundary", strengthVisualization);
  imshow("boundary binary", strengthVisualization > 0);
//  imshow("boundary", boundaryStrength);
  waitKey();
#endif
}

bool GlassClassifier::read(const std::string &filename)
{
  FileStorage fs(filename, FileStorage::READ);
  if (!fs.isOpened())
  {
    return false;
  }

  fs["scalingSlope"] >> scalingSlope;
  fs["scalingIntercept"] >> scalingIntercept;
  fs["normalizationSlope"] >> normalizationSlope;
  fs["normalizationIntercept"] >> normalizationIntercept;
  string svmFilename;
  fs["svmFilename"] >> svmFilename;
  svm.load(svmFilename.c_str());
  fs.release();

  return true;
}

void GlassClassifier::write(const std::string &filename)
{
  FileStorage fs(filename, FileStorage::WRITE);
  CV_Assert(fs.isOpened());

  string svmFilename = filename + ".svm.xml";
  svm.save(svmFilename.c_str());

  fs << "scalingSlope" << scalingSlope;
  fs << "scalingIntercept" << scalingIntercept;
  fs << "normalizationSlope" << normalizationSlope;
  fs << "normalizationIntercept" << normalizationIntercept;
  fs << "svmFilename" << svmFilename;

  fs.release();
}

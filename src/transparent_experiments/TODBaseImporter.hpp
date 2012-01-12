/*
 * TODBaseImporter.hpp
 *
 *  Created on: Aug 12, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef TODBASEIMPORTER_HPP_
#define TODBASEIMPORTER_HPP_

#include <opencv2/opencv.hpp>
#include <edges_pose_refiner/edgeModel.hpp>

class TODBaseImporter
{
public:
  TODBaseImporter();
  TODBaseImporter(const std::string &trainFolder, const std::string &testFolder);

  void readTrainObjectsNames(const std::string &trainConfigFilename, std::vector<std::string> &trainObjectsNames);
  void readCameraParams(const std::string &folder, PinholeCamera &camera, bool addFilename = true);
  void readMultiCameraParams(const std::string &camerasListFilename, std::vector<PinholeCamera> &allCameras, std::vector<bool> &camerasMask);

  void importEdgeModel(const std::string &modelsPath, const std::string &objectName, EdgeModel &edgeModel) const;
  void importTestIndices(std::vector<int> &testIndices) const;
  void importDepth(int testImageIdx, cv::Mat &depth) const;
  void importBGRImage(int testImageIdx, cv::Mat &bgrImage) const;
  void importGroundTruth(int testImageIdx, PoseRT &model2test) const;
  void importPointCloud(int testImageIdx, pcl::PointCloud<pcl::PointXYZ> &cloud) const;

  void exportTrainPointClouds(const std::string &outFolder) const;
private:
  void readTrainSamples();
  void readRegisteredClouds(const std::string &configFilename, std::vector<std::vector<cv::Point3f> > &registeredClouds) const;
  void matchRegisteredClouds(const std::vector<std::vector<cv::Point3f> > &registeredClouds, EdgeModel &edgeModel) const;
  void alignModel(EdgeModel &edgeModel) const;
  void computeStableEdgels(EdgeModel &edgeModel) const;
  void readRawEdgeModel(const std::string &filename, EdgeModel &edgeModel);
  void createEdgeModel(EdgeModel &edgeModel);


  std::vector<EdgeModelCreator::TrainSample> trainSamples;

  std::string trainFolder, testFolder;
  cv::Mat cameraMatrix, distCoeffs;
};

bool isNan(const cv::Point3f& p);

#endif /* TODBASEIMPORTER_HPP_ */

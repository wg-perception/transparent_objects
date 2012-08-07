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
//  TODBaseImporter(const std::string &trainFolder, const std::string &testFolder);
  TODBaseImporter(const std::string &testFolder);

  void readTrainObjectsNames(const std::string &trainConfigFilename, std::vector<std::string> &trainObjectsNames);
  void readCameraParams(const std::string &folder, PinholeCamera &camera, bool addFilename = true);
  void readMultiCameraParams(const std::string &camerasListFilename, std::vector<PinholeCamera> &allCameras, std::vector<bool> &camerasMask);

  static void importCamera(const std::string &filename, PinholeCamera &camera);
  void importEdgeModel(const std::string &modelsPath, const std::string &objectName, EdgeModel &edgeModel) const;
  void importTestIndices(std::vector<int> &testIndices) const;
  void importGroundTruth(int testImageIdx, PoseRT &model2test, bool shiftByOffset = true, PoseRT *offsetPtr = 0) const;
  void importAllGroundTruth(std::map<int, PoseRT> &allPoses) const;
  void importOcclusionObjects(const std::string &modelsPath,
                              std::vector<EdgeModel> &occlusionObjects, std::vector<PoseRT> &occlusionOffsets) const;

  static void importRegistrationMask(const std::string &filename, cv::Mat &registrationMask);

  void importDepth(int testImageIdx, cv::Mat &depth) const;
  static void importDepth(const std::string &filename, cv::Mat &depth);

  void importBGRImage(int testImageIdx, cv::Mat &bgrImage) const;
  static void importBGRImage(const std::string &filename, cv::Mat &depth);

  void importRawMask(int testImageIdx, cv::Mat &bgrImage) const;

  void importPointCloud(int testImageIdx, pcl::PointCloud<pcl::PointXYZ> &cloud) const;
  static void importPointCloud(const std::string &filename, pcl::PointCloud<pcl::PointXYZ> &cloud);
  static void importPointCloud(const std::string &filename, cv::Mat &cloud);

  void exportTrainPointClouds(const std::string &outFolder) const;
private:
  void readRegisteredClouds(const std::string &configFilename, std::vector<std::vector<cv::Point3f> > &registeredClouds) const;

  std::string trainFolder, testFolder;
  cv::Mat cameraMatrix, distCoeffs;
};

bool isNan(const cv::Point3f& p);

#endif /* TODBASEIMPORTER_HPP_ */

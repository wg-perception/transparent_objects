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
  TODBaseImporter(const std::string &baseFolder, const std::string &testFolder);

  void importAllData(const std::string *trainedModelsPath = 0, const std::vector<std::string> *trainObjectNames = 0,
                     PinholeCamera *kinectCamera = 0, cv::Mat *registrationMask = 0,
                     std::vector<EdgeModel> *edgeModels = 0, std::vector<int> *testIndices = 0,
                     std::vector<EdgeModel> *occlusionObjects = 0, std::vector<PoseRT> *occlusionOffsets = 0,
                     PoseRT *offset = 0) const;

//  void readTrainObjectsNames(const std::string &trainConfigFilename, std::vector<std::string> &trainObjectsNames);
  void readMultiCameraParams(const std::string &camerasListFilename, std::vector<PinholeCamera> &allCameras, std::vector<bool> &camerasMask);

  static void importCamera(const std::string &filename, PinholeCamera &camera);
  void importCamera(PinholeCamera &camera) const;

  void importEdgeModel(const std::string &modelsPath, const std::string &objectName, EdgeModel &edgeModel) const;
  void importTestIndices(std::vector<int> &testIndices) const;
  void importGroundTruth(int testImageIdx, PoseRT &model2test, bool shiftByOffset = true, PoseRT *offsetPtr = 0, bool isKeyFrame = false) const;
  void importOffset(PoseRT &offset) const;
  void importAllGroundTruth(std::map<int, PoseRT> &allPoses) const;
  void importOcclusionObjects(const std::string &modelsPath,
                              std::vector<EdgeModel> &occlusionObjects, std::vector<PoseRT> &occlusionOffsets) const;

  void importRegistrationMask(cv::Mat &registrationMask) const;
  static void importRegistrationMask(const std::string &filename, cv::Mat &registrationMask);

  void importDepth(int testImageIdx, cv::Mat &depth) const;
  static void importDepth(const std::string &filename, cv::Mat &depth);

  void importBGRImage(int testImageIdx, cv::Mat &bgrImage) const;
  static void importBGRImage(const std::string &filename, cv::Mat &bgrImage);

  void importRawMask(int testImageIdx, cv::Mat &rawMask) const;
  void importUserMask(int testImageIdx, cv::Mat &userMask) const;

  //void importPointCloud(int testImageIdx, std::vector<cv::Point3f> &cloud) const;
  static void importPointCloud(const std::string &filename, cv::Mat &cloud, cv::Mat &normals);

  void exportTrainPointClouds(const std::string &outFolder) const;
private:
  void readRegisteredClouds(const std::string &configFilename, std::vector<std::vector<cv::Point3f> > &registeredClouds) const;

  std::string baseFolder, testFolder;
  cv::Mat cameraMatrix, distCoeffs;
};

bool isNan(const cv::Point3f& p);

#endif /* TODBASEIMPORTER_HPP_ */

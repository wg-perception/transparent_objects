/*
 * TODBaseImporter.cpp
 *
 *  Created on: Aug 12, 2011
 *      Author: Ilya Lysenkov
 */

#include "edges_pose_refiner/TODBaseImporter.hpp"
#include <fstream>
#include <iomanip>
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"

#include <sys/types.h>
#include <dirent.h>

using namespace cv;
using std::cout;
using std::endl;


TODBaseImporter::TODBaseImporter()
{
}

//TODBaseImporter::TODBaseImporter(const std::string &_trainFolder, const std::string &_testFolder)
TODBaseImporter::TODBaseImporter(const std::string &_testFolder)
{
//  trainFolder = _trainFolder;
  testFolder = _testFolder;

//  PinholeCamera camera;
//  readCameraParams(trainFolder, camera);
//  cameraMatrix = camera.cameraMatrix;
//  distCoeffs = camera.distCoeffs;
}

void TODBaseImporter::readCameraParams(const string &folder, PinholeCamera &camera, bool addFilename)
{
  string cameraFilename = addFilename ? folder + "/camera.yml" : folder;
  camera.read(cameraFilename);
}

void TODBaseImporter::readMultiCameraParams(const string &camerasListFilename, std::vector<PinholeCamera> &allCameras, vector<bool> &camerasMask)
{
  vector<string> intrinsicsFilenames;
  readLinesInFile(camerasListFilename, intrinsicsFilenames);
  camerasMask.resize(intrinsicsFilenames.size());
  size_t activeCamerasCount = 0;
  for(size_t i=0; i<intrinsicsFilenames.size(); i++)
  {
    camerasMask[i] = (intrinsicsFilenames[i][0] != '#');
    if(camerasMask[i])
      activeCamerasCount++;
  }

  allCameras.resize(activeCamerasCount);
  for(size_t i=0, cameraIdx=0; i<intrinsicsFilenames.size(); i++)
  {
    if(camerasMask[i])
    {
      readCameraParams(intrinsicsFilenames[i], allCameras[cameraIdx], false);
      cameraIdx++;
    }
  }
}

void TODBaseImporter::readTrainObjectsNames(const string &trainConfigFilename, std::vector<string> &trainObjectsNames)
{
  trainObjectsNames.clear();
  string configFilename = trainFolder + "/" + trainConfigFilename;
  std::ifstream configFile(configFilename.c_str());
  CV_Assert(configFile.is_open());

  while(!configFile.eof())
  {
    string object;
    configFile >> object;
    if(object.empty())
    {
      break;
    }

    trainObjectsNames.push_back(object);
    cout << object << endl;
  }
  configFile.close();
}

bool isNan(const Point3f& p)
{
 return isnan(p.x) || isnan(p.y) || isnan(p.z);
}

void TODBaseImporter::readRegisteredClouds(const string &configFilename, vector<vector<cv::Point3f> > &registeredClouds) const
{
  std::ifstream configFile(configFilename.c_str());
  vector<string> filenames;
  while(!configFile.eof())
  {
    string name;
    configFile >> name;
    if(!name.empty())
    {
      filenames.push_back(name);
    }
  }

  registeredClouds.resize(filenames.size());
  for(size_t i=0; i<filenames.size(); i++)
  {
    readPointCloud(filenames[i], registeredClouds[i]);
  }
}

void TODBaseImporter::importOcclusionObjects(const std::string &modelsPath,
                                             std::vector<EdgeModel> &occlusionObjects, std::vector<PoseRT> &occlusionOffsets) const
{
  //TODO: move up
  const string occlusionPrefix = "occlusion_";
  const string occlusionPostFix = ".xml";
  //TOOD: port to other systems too
  DIR *directory = opendir(testFolder.c_str());
  CV_Assert(directory != 0);

  occlusionObjects.clear();
  for (dirent *entry = readdir(directory); entry != 0; entry = readdir(directory))
  {
    string filename = entry->d_name;
    if (filename.substr(0, occlusionPrefix.length()) != occlusionPrefix)
    {
      continue;
    }

    int objectNameLength = static_cast<int>(filename.length()) - static_cast<int>(occlusionPostFix.length()) - static_cast<int>(occlusionPrefix.length());
    string objectName = filename.substr(occlusionPrefix.length(), objectNameLength);

    EdgeModel edgeModel;
    importEdgeModel(modelsPath, objectName, edgeModel);
    occlusionObjects.push_back(edgeModel);

    PoseRT offset;
    offset.read(testFolder + "/" + filename);
    occlusionOffsets.push_back(offset);
  }
}

void TODBaseImporter::importEdgeModel(const std::string &modelsPath, const std::string &objectName, EdgeModel &edgeModel) const
{
  string modelFilename = modelsPath + "/" + objectName + ".xml";
  try
  {
    edgeModel.read(modelFilename);
  }
  catch(cv::Exception &e)
  {
    cout << "Cannot read edge model: " << modelFilename << endl;
    vector<Point3f> points, normals;
    readPointCloud(modelsPath + "/downPointClouds/" + objectName + ".ply", points, &normals);
    edgeModel = EdgeModel(points, normals, true, true);

/*  uncomment for obsolete models (sourCream)

    for (size_t i = 0; i < points.size(); ++i)
    {
      points[i].z *= -1;
    }
    edgeModel = EdgeModel(points, true, true);
*/
/*
    EdgeModel rotatedEdgeModel;
    Mat R, tvec;
    getRotationTranslation(edgeModel.Rt_obj2cam, R, tvec);
    Mat rvec = Mat::zeros(3, 1, CV_64FC1);
    tvec = Mat::zeros(3, 1, CV_64FC1);
    rvec.at<double>(0) = CV_PI;
    edgeModel.rotate_obj(PoseRT(rvec, 2 * tvec), rotatedEdgeModel);
    edgeModel = rotatedEdgeModel;
*/

    edgeModel.write(modelFilename);
  }
}

void TODBaseImporter::importTestIndices(vector<int> &testIndices) const
{
  testIndices.clear();

  string basePath = testFolder + "/";
  string imagesList = basePath + "testImages.txt";
  std::ifstream fin(imagesList.c_str());
  if (!fin.is_open())
  {
    CV_Error(CV_StsError, "Cannot open the file " + imagesList);
  }
  while(!fin.eof())
  {
    int idx = -1;
    fin >> idx;

    if(idx >= 0)
    {
      testIndices.push_back(idx);
    }
  }
}

void TODBaseImporter::importDepth(const std::string &filename, cv::Mat &depth)
{
  FileStorage fs(filename, FileStorage::READ);
  CV_Assert(fs.isOpened());
  fs["depth_image"] >> depth;
  fs.release();
  CV_Assert(!depth.empty());
}

void TODBaseImporter::importDepth(int testImageIdx, cv::Mat &depth) const
{
  std::stringstream depthFilename;
  depthFilename << testFolder << "/depth_image_" << std::setfill('0') << std::setw(5) << testImageIdx << ".xml.gz";
  importDepth(depthFilename.str(), depth);
}

void TODBaseImporter::importBGRImage(const std::string &filename, cv::Mat &bgrImage)
{
  bgrImage = imread(filename, CV_LOAD_IMAGE_UNCHANGED);
  if (bgrImage.empty())
  {
    CV_Error(CV_StsBadArg, "Cannot read the image " + filename);
  }
}

void TODBaseImporter::importBGRImage(int testImageIdx, cv::Mat &bgrImage) const
{
  std::stringstream imageFilename;
  imageFilename << testFolder << "/image_" << std::setfill('0') << std::setw(5) << testImageIdx << ".png";
  importBGRImage(imageFilename.str(), bgrImage);
}

void TODBaseImporter::importRawMask(int testImageIdx, cv::Mat &mask) const
{
  std::stringstream imageFilename;
  imageFilename << testFolder << "/image_" << std::setfill('0') << std::setw(5) << testImageIdx << ".png.raw_mask.png";
  importBGRImage(imageFilename.str(), mask);
  CV_Assert(mask.channels() == 1);
}

void TODBaseImporter::importGroundTruth(int testImageIdx, PoseRT &model2test, bool shiftByOffset, PoseRT *offsetPtr) const
{
  std::stringstream testPoseFilename;
  testPoseFilename << testFolder +"/image_" << std::setfill('0') << std::setw(5) << testImageIdx << ".png.pose.yaml";
  FileStorage testPoseFS;
  testPoseFS.open(testPoseFilename.str(), FileStorage::READ);
  CV_Assert(testPoseFS.isOpened());

  testPoseFS["pose"]["rvec"] >> model2test.rvec;
  testPoseFS["pose"]["tvec"] >> model2test.tvec;
  testPoseFS.release();

  if (shiftByOffset || offsetPtr != 0)
  {
    //TODO: move up
    const string offsetFilename = "offset.xml";
    PoseRT offset;
    offset.read(testFolder + "/" + offsetFilename);
    if (shiftByOffset)
    {
      model2test = model2test * offset;
    }
    if (offsetPtr != 0)
    {
      *offsetPtr = offset;
    }
  }
}

void TODBaseImporter::importAllGroundTruth(std::map<int, PoseRT> &allPoses) const
{
  allPoses.clear();
  vector<int> testIndices;
  importTestIndices(testIndices);
  for (size_t testIndex = 0; testIndex < testIndices.size(); ++testIndex)
  {
    int imageIndex = testIndices[testIndex];
    PoseRT pose;
    importGroundTruth(imageIndex, pose);
    allPoses[imageIndex] = pose;
  }
}

void TODBaseImporter::importPointCloud(const std::string &filename, pcl::PointCloud<pcl::PointXYZ> &cloud)
{
  pcl::io::loadPCDFile(filename, cloud);
}

void TODBaseImporter::importPointCloud(int testImageIdx, pcl::PointCloud<pcl::PointXYZ> &cloud) const
{
  std::stringstream pointCloudFilename;
  pointCloudFilename << testFolder << "/new_cloud_" << std::setfill('0') << std::setw(5) << testImageIdx << ".pcd";
  importPointCloud(pointCloudFilename.str(), cloud);
}

void TODBaseImporter::importPointCloud(const std::string &filename, cv::Mat &cloud)
{
  FileStorage fs(filename, FileStorage::READ);
  CV_Assert(fs.isOpened());
  fs["cloud"] >> cloud;
  fs.release();
  CV_Assert(!cloud.empty());
}

void TODBaseImporter::importRegistrationMask(const std::string &filename, cv::Mat &registrationMask)
{
  registrationMask = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
  CV_Assert(!registrationMask.empty());
}

void TODBaseImporter::importCamera(const std::string &filename, PinholeCamera &camera)
{
  camera.read(filename);
}

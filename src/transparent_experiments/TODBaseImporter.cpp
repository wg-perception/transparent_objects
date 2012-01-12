/*
 * TODBaseImporter.cpp
 *
 *  Created on: Aug 12, 2011
 *      Author: Ilya Lysenkov
 */

#include "TODBaseImporter.hpp"
#include <fstream>
#include <iomanip>
#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"

using namespace cv;
using std::cout;
using std::endl;


TODBaseImporter::TODBaseImporter()
{
}

TODBaseImporter::TODBaseImporter(const std::string &_trainFolder, const std::string &_testFolder)
{
  trainFolder = _trainFolder;
  testFolder = _testFolder;

  PinholeCamera camera;
  readCameraParams(trainFolder, camera);
  cameraMatrix = camera.cameraMatrix;
  distCoeffs = camera.distCoeffs;
  readTrainSamples();
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

void TODBaseImporter::readTrainSamples()
{
  string modelFilename = trainFolder + "/edgeModel.xml";
  string configFilename = trainFolder + "/trainImages.txt";

  std::ifstream configFile(configFilename.c_str());
  CV_Assert(configFile.is_open());
  vector<int> trainIndices;
  while(!configFile.eof())
  {
    int idx;
    configFile >> idx;
    if(std::find(trainIndices.begin(), trainIndices.end(), idx) == trainIndices.end())
      trainIndices.push_back(idx);
  }
  configFile.close();
  CV_Assert(trainIndices.size() > 0);


  string objFolder = trainFolder;

  {
    trainSamples.resize(trainIndices.size());
    cout << "Reading train samples..." << endl;
    for(size_t sampleIdx=0; sampleIdx < trainIndices.size(); sampleIdx++)
    {
      EdgeModelCreator::TrainSample &sample = trainSamples[sampleIdx];
      std::stringstream idx;
      idx << std::setfill('0') << std::setw(5) << trainIndices[sampleIdx];
      cout << objFolder + "/image_" + idx.str() + ".png"<< endl;
      sample.image = imread(objFolder + "/image_" + idx.str() + ".png", CV_LOAD_IMAGE_GRAYSCALE);
      sample.mask = imread(objFolder + "/image_" + idx.str() + ".png.raw_mask.png", CV_LOAD_IMAGE_GRAYSCALE);

      FileStorage fs(objFolder + "/image_" + idx.str() + ".png.pose.yaml", FileStorage::READ);
      CV_Assert(fs.isOpened());
      fs["pose"]["rvec"] >> sample.rvec;
      fs["pose"]["tvec"] >> sample.tvec;
      fs.release();


      string cloudFilename = objFolder + "/cloud_" + idx.str() + ".pcd";

      const Mat zeros3x1 = Mat::zeros(3, 1, CV_64FC1);

      pcl::PointCloud<pcl::PointXYZ> point_cloud;
      pcl::io::loadPCDFile(cloudFilename.c_str(), point_cloud);

      CV_Assert(sample.mask.type() == CV_8UC1);

//      cout << cameraMatrix << endl;
//      cout << distCoeffs << endl;

      for(size_t ptIdx=0; ptIdx < point_cloud.points.size(); ptIdx++)
      {
        pcl::PointXYZ ptXYZ = point_cloud.points[ptIdx];
        Point3f pt3f(ptXYZ.x, ptXYZ.y, ptXYZ.z);
//        cout << pt3f << endl;
        if(isNan(pt3f))
          continue;

        vector<Point3f> points;
        points.push_back(pt3f);
        vector<Point2f> projectedPoints;
        projectPoints(Mat(points), zeros3x1, zeros3x1, cameraMatrix, distCoeffs, projectedPoints);
        Point2f pt2f = projectedPoints[0];
        Point pt = pt2f;

        if(pt.x < 0 || pt.y < 0 || pt.x >= sample.mask.cols || pt.y >= sample.mask.rows)
          continue;

        if(sample.mask.at<uchar>(pt) != 0)
          sample.pointCloud.push_back(pt3f);
      }
      cout << "point_cloud size: " << sample.pointCloud.size() << endl;

/*
      vector<Point3f> rotatedCloud;
      project3dPoints(sample.pointCloud, sample.rvec, sample.tvec, rotatedCloud);
      sample.pointCloud = rotatedCloud;
*/

      Mat Rt;
      createProjectiveMatrix(sample.rvec, sample.tvec, Rt);
      getRvecTvec(Rt.inv(DECOMP_SVD), sample.rvec, sample.tvec);


    }
    cout << "Done." << endl;
  }

}


void TODBaseImporter::createEdgeModel(EdgeModel &edgeModel)
{
  cout << "Started create edge model" << endl;
  EdgeModelCreatorParams params;
  params.useOnlyEdges = false;
  EdgeModelCreator edgeModelCreator(cameraMatrix, distCoeffs, false, params);
  //edgeModelCreator.createEdgeModel(trainSamples, edgeModel);
  edgeModelCreator.createEdgeModel(trainSamples, edgeModel);
  cout << "done" << endl;
}

void TODBaseImporter::readRawEdgeModel(const string &filename, EdgeModel &edgeModel)
{
  edgeModel.clear();

  readPointCloud(filename, edgeModel.points);

//  {
//    Scalar meanVal = mean(Mat(edgeModel.points));
//    Point3d center = Point3d(meanVal[0], meanVal[1], meanVal[2]);
//    drawAxis2(center, Point3d(0, 0, 1), pointsPublisher, 40, Scalar(255, 0, 255));
//  }


  EdgeModelCreatorParams params;
  params.useOnlyEdges = false;
  EdgeModelCreator edgeModelCreator(cameraMatrix, distCoeffs, false, params);
  edgeModelCreator.alignModel(trainSamples, edgeModel);

//  {
//    Scalar meanVal = mean(Mat(edgeModel.points));
//    Point3d center = Point3d(meanVal[0], meanVal[1], meanVal[2]);
//    drawAxis2(center, Point3d(0, 0, 1), pointsPublisher, 40, Scalar(255, 0, 255));
//  }

//  const int pointsCount = 300;
//  float ratio = pointsCount / double(edgeModel.points.size());
//
//  edgeModelCreator.downsampleEdgeModel(edgeModel, ratio);
//
//  if(pointsPublisher != 0)
//  {
//    namedWindow("Ready to publish downsampled model");
//    waitKey();
//    publishPoints(edgeModel.points, *pointsPublisher);
//    namedWindow("Done");
//    waitKey();
//  }

}

void TODBaseImporter::exportTrainPointClouds(const string &outFolder) const
{
  EdgeModelCreatorParams params;
  params.useOnlyEdges = false;
  EdgeModelCreator edgeModelCreator(cameraMatrix, distCoeffs, 0, params);
  vector<vector<Point3f> > edgels;
  edgeModelCreator.getEdgePointClouds(trainSamples, edgels);

  for(size_t i=0; i<edgels.size(); i++)
  {
    std::stringstream filename;
    filename << outFolder << "/" << i << ".asc";
    writePointCloud(filename.str(), edgels[i]);
  }
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

void TODBaseImporter::matchRegisteredClouds(const std::vector<std::vector<cv::Point3f> > &registeredClouds, EdgeModel &edgeModel) const
{
  edgeModel.clear();
  EdgeModelCreatorParams params;
  params.useOnlyEdges = false;

  params.inliersRatio = 0.5f;
  params.neighborsRatio = 0.25f;
  //params.neighborsRatio = 0.25f;
  //params.neighborsRatio = 0.33f;

  //params.finalModelOutliersRatio = 0.3f;
  //params.finalModelOutliersRatio = 0.2f;
  params.finalModelOutliersRatio = 0.2f;

  EdgeModelCreator edgeModelCreator(cameraMatrix, distCoeffs, 0, params);
  cout << "Starting matching..." << endl;
  edgeModelCreator.matchPointClouds(registeredClouds, edgeModel.points);
  cout << "Done: " << edgeModel.points.size() << " edgels" << endl;

}

void TODBaseImporter::alignModel(EdgeModel &edgeModel) const
{
  EdgeModelCreatorParams params;
  params.useOnlyEdges = false;

  EdgeModelCreator edgeModelCreator(cameraMatrix, distCoeffs, false, params);
  edgeModelCreator.alignModel(trainSamples, edgeModel);
}

void TODBaseImporter::computeStableEdgels(EdgeModel &edgeModel) const
{
  EdgeModelCreator verboseEdgeModelCreator(cameraMatrix, distCoeffs, false);
  verboseEdgeModelCreator.computeStableEdgels(trainSamples, edgeModel);
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
    vector<Point3f> points;
    readPointCloud(modelsPath + "/downPointClouds/" + objectName + ".ply", points);
    edgeModel = EdgeModel(points, true, true);
    edgeModel.write(modelFilename);
  }
}

void TODBaseImporter::importTestIndices(vector<int> &testIndices) const
{
  testIndices.clear();

  string basePath = testFolder + "/";
  string imagesList = basePath + "testImages.txt";
  std::ifstream fin(imagesList.c_str());
  CV_Assert(fin.is_open());
  while(!fin.eof())
  {
    int idx = -1;
    fin >> idx;

    cout << idx << endl;

    if(idx >= 0)
      testIndices.push_back(idx);
  }
}

void TODBaseImporter::importDepth(int testImageIdx, cv::Mat &depth) const
{
  std::stringstream depthFilename;
  depthFilename << testFolder << "/depth_image_" << std::setfill('0') << std::setw(5) << testImageIdx << ".xml.gz";
  FileStorage fs(depthFilename.str(), FileStorage::READ);
  CV_Assert(fs.isOpened());
  fs["depth_image"] >> depth;
  fs.release();
}

void TODBaseImporter::importBGRImage(int testImageIdx, cv::Mat &bgrImage) const
{
  std::stringstream imageFilename;
  imageFilename << testFolder << "/image_" << std::setfill('0') << std::setw(5) << testImageIdx << ".png";
  bgrImage = imread(imageFilename.str());
  CV_Assert(!bgrImage.empty());
}

void TODBaseImporter::importGroundTruth(int testImageIdx, PoseRT &model2test) const
{
  std::stringstream testPoseFilename;
  testPoseFilename << testFolder +"/image_" << std::setfill('0') << std::setw(5) << testImageIdx << ".png.pose.yaml";
  FileStorage testPoseFS;
  testPoseFS.open(testPoseFilename.str(), FileStorage::READ);
  CV_Assert(testPoseFS.isOpened());

  testPoseFS["pose"]["rvec"] >> model2test.rvec;
  testPoseFS["pose"]["tvec"] >> model2test.tvec;
  testPoseFS.release();

  const string offsetFilename = "offset.xml";
  PoseRT offset;
  offset.read(testFolder + "/" + offsetFilename);
  model2test = model2test * offset;
}

void TODBaseImporter::importPointCloud(int testImageIdx, pcl::PointCloud<pcl::PointXYZ> &cloud) const
{
  std::stringstream pointCloudFilename;
  pointCloudFilename << testFolder << "/new_cloud_" << std::setfill('0') << std::setw(5) << testImageIdx << ".pcd";
  pcl::io::loadPCDFile(pointCloudFilename.str(), cloud);
}

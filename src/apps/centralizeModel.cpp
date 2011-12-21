/*
 * centralizeModel.cpp
 *
 *  Created on: Nov 14, 2011
 *      Author: Ilya Lysenkov
 */

#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/edgeModel.hpp"
using namespace cv;

using std::cout;
using std::endl;


int main(int argc, char *argv[])
{
  CV_Assert(argc == 3);
  string inModelFilename = argv[1];
  string outModelFilename = argv[2];

  EdgeModel outModel;
  const int extLength = 3;
  string extension = inModelFilename.substr(inModelFilename.length() - extLength, extLength);
  CV_Assert(extension == "ply" || extension == "xml");
  if (extension == "ply")
  {
    vector<Point3f> points;
    readPointCloud(inModelFilename, points);
    outModel = EdgeModel(points, true);
  }
  if (extension == "xml")
  {
    assert(false);
    //inModel.read(inModelFilename);
  }


  outModel.write(outModelFilename);

  std::cout << outModel.Rt_obj2cam << std::endl;
  cout << outModel.rotationAxis << endl;

  outModel.visualize();

  writePointCloud("centralizedModel.asc", outModel.points);

  return 0;
}

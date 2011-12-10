/*
 * stereoCalib2TOD.cpp
 *
 *  Created on: Sep 2, 2011
 *      Author: Ilya Lysenkov
 */

#include <opencv2/opencv.hpp>

using namespace cv;
using std::cout;
using std::endl;

void writeCameraFile(const string &filename, const Mat &cameraMatrix, const Mat &distCoeffs, const Mat &R, const Mat &t)
{
  FileStorage cameraFS(filename, FileStorage::WRITE);
  CV_Assert(cameraFS.isOpened());
  cameraFS << "camera" << "{";
  cameraFS << "K" << cameraMatrix;
  cameraFS << "D" << distCoeffs;
  cameraFS << "pose" << "{";
  Mat rvec;
  Rodrigues(R, rvec);
  cameraFS << "rvec" << rvec;
  cameraFS << "tvec" << t.reshape(1, 3);
  cameraFS << "estimated" << true;
  cameraFS << "}" << "}";
  cameraFS.release();
}

int main(int argc, char *argv[])
{
  if(argc != 5)
  {
    cout << argv[0] << " <intrinsics> <extrinsics> <cameraFile1> <cameraFile2>" << endl;
    return 0;
  }

  FileStorage intrinsics(argv[1], FileStorage::READ);
  CV_Assert(intrinsics.isOpened());
  Mat M1, D1, M2, D2;
  intrinsics["M1"] >> M1;
  intrinsics["D1"] >> D1;
  intrinsics["M2"] >> M2;
  intrinsics["D2"] >> D2;
  intrinsics.release();

  FileStorage extrinsics(argv[2], FileStorage::READ);
  CV_Assert(extrinsics.isOpened());
  Mat R, T;
  extrinsics["R"] >> R;
  extrinsics["T"] >> T;
  extrinsics.release();

  writeCameraFile(argv[3], M1, D1, Mat::eye(3, 3, CV_64FC1), Mat::zeros(3, 1, CV_64FC1));
  writeCameraFile(argv[4], M2, D2, R, T);

  return 0;
}

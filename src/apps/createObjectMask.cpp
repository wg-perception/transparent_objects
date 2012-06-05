#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/TODBaseImporter.hpp"

using namespace cv;
using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
  if (argc != 5)
  {
    cout << argv[0] << " <testFolder> <imageIndex> <model> <outMask>" << endl;
    return 0;
  }

  const int closingIterationsCount = 5;
  const float downFactor = 1.0f;
  bool visualize = false;

  string testFolder = argv[1];
  int imageIndex = atoi(argv[2]);
  string modelFilename = argv[3];
  string outFilename = argv[4];

  TODBaseImporter dataImporter(testFolder);

  PinholeCamera camera;
  dataImporter.readCameraParams(testFolder, camera, true);
  CV_Assert(camera.imageSize == Size(640, 480));
  Ptr<const PinholeCamera> cameraPtr = new PinholeCamera(camera);

  Mat image;
  dataImporter.importBGRImage(imageIndex, image);
  PoseRT model2test;
  dataImporter.importGroundTruth(imageIndex, model2test);

  EdgeModel edgeModel;
  edgeModel.read(modelFilename);

  Silhouette silhouette;
  edgeModel.getSilhouette(cameraPtr, model2test, silhouette, downFactor, closingIterationsCount);

  if (visualize)
  {
    silhouette.draw(image, 0);
    imshow("image", image);
    waitKey();
  }

  Mat edgelsMat;
  silhouette.getEdgels(edgelsMat);
  Mat edgelsMatInt;
  edgelsMat.convertTo(edgelsMatInt, CV_32SC2);
  vector<Point> edgels = edgelsMatInt;
  vector<vector<Point> > contours(1, edgels);

  Mat mask = Mat(image.size(), CV_8UC1, Scalar(0));
  drawContours(mask, contours, -1, Scalar(255), CV_FILLED);
  if (visualize)
  {
    imshow("mask", mask);
    waitKey();
  }
  imwrite(outFilename, mask);

  return 0;
}

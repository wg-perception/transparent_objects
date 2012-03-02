#include <opencv2/opencv.hpp>
#include <fstream>
#include "edges_pose_refiner/segmentedImage.hpp"

using namespace cv;

int main(int argc, char *argv[])
{
  CV_Assert(argc == 3);
  const string segmentedImageFilename = argv[1];
  const string segmentationFilename = argv[2];

  SegmentedImage segmentedImage;
  segmentedImage.read(segmentedImageFilename);

  Mat segmentation = segmentedImage.getSegmentation();
  std::ofstream fout(segmentationFilename.c_str());
  for (int i = 0; i < segmentation.rows; ++i)
  {
    for (int j = 0; j < segmentation.cols; ++j)
    {
      fout << segmentation.at<int>(i, j) << " ";
    }
    fout << "\n";
  }
  fout.close();

  return 0;
}

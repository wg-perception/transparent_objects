#include <opencv2/core/core.hpp>

void computeNormals(const cv::Mat &edges, cv::Mat &normals);
void computeDistanceTransform3D(const cv::Mat &edges,
                                std::vector<cv::Mat> &dtImages);

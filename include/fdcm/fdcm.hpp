#include <opencv2/core/core.hpp>

void computeNormals(const cv::Mat &edges, cv::Mat &normals, cv::Mat &orientationIndices);
void computeOrientationIndices(const std::vector<cv::Point2f> &points, const cv::Mat &dx, const cv::Mat &dy,
                               std::vector<int> &orientationIndices);
void computeDistanceTransform3D(const cv::Mat &edges,
                                std::vector<cv::Mat> &dtImages);

int theta2Index(float theta, int directionsCount);

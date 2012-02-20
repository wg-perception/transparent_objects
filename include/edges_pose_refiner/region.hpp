#ifndef REGION_HPP__
#define REGION_HPP__

#include <opencv2/core/core.hpp>

class Region
{
  public:
    Region(const cv::Mat &image, const cv::Mat &textonLabels, const cv::Mat &mask);

    cv::Point2f getCenter() const;
    const cv::Mat& getMask() const;
    const cv::Mat& getColorHistogram() const;
    const cv::Mat& getTextonHistogram() const;
    const cv::Mat& getIntensityClusters() const;
  private:
    void computeColorHistogram();
    void computeTextonHistogram();
    void clusterIntensities();
    void computeCenter();

    cv::Mat image, textonLabels, mask, erodedMask;
    cv::Mat grayscaleImage;

    cv::Mat hist;
    cv::Mat intensityClusterCenters;
    cv::Mat textonHistogram;

    cv::Point2f center;
};

#endif

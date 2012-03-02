#ifndef REGION_HPP__
#define REGION_HPP__

#include <opencv2/core/core.hpp>

class Region
{
  public:
    Region();
    Region(const cv::Mat &image, const cv::Mat &textonLabels, const cv::Mat &mask);

    cv::Point2f getCenter() const;
    cv::Vec3b getMedianColor() const;
    const cv::Mat& getMask() const;
    const cv::Mat& getErodedMask() const;
    const cv::Mat& getColorHistogram() const;
    const cv::Mat& getTextonHistogram() const;
    const cv::Mat& getIntensityClusters() const;
    bool isEmpty() const;

    void write(cv::FileStorage &fs) const;
    void read(const cv::Mat &image, const cv::Mat &mask, const cv::FileNode &fn);
  private:
    void computeColorHistogram();
    void computeTextonHistogram();
    void clusterIntensities();
    void computeCenter();
    void computeMedianColor();
    static void computeErodedMask(const cv::Mat &mask, const cv::Mat &erodedMask);

    cv::Mat image, textonLabels, mask, erodedMask;
    cv::Mat grayscaleImage;

    cv::Mat colorHistogram;
    cv::Mat intensityClusterCenters;
    cv::Mat textonHistogram;

    cv::Point2f center;
    cv::Vec3b medianColor;
    int erodedArea;
};

#endif

#ifndef REGION_HPP__
#define REGION_HPP__

#include <opencv2/core/core.hpp>

struct RegionParams
{
  int hbins;
  int sbins;
  int textonCount;
  int clusterCount;
  int erosionCount;
  float outliersRatio;

  RegionParams()
  {
    hbins = 20;
    sbins = 20;
    textonCount = 36;
    clusterCount = 10;
    erosionCount = 2;
    outliersRatio = 0.1f;
  }
};

class Region
{
  public:
    Region(const RegionParams &params = RegionParams());
    Region(const cv::Mat &image, const cv::Mat &textonLabels, const cv::Mat &mask,
           const RegionParams &params = RegionParams());

    cv::Point2f getCenter() const;
    cv::Vec3b getMedianColor() const;
    const cv::Mat& getMask() const;
    const cv::Mat& getErodedMask() const;
    const cv::Mat& getColorHistogram() const;
    const cv::Mat& getTextonHistogram() const;
    const cv::Mat& getIntensityClusters() const;
    float getRMSContrast() const;
    float getMichelsonContrast() const;
    float getRobustMichelsonContrast() const;
    bool isEmpty() const;

    void write(cv::FileStorage &fs) const;
    void read(const cv::Mat &image, const cv::Mat &mask, const cv::FileNode &fn);
  private:
    void computeIntensities();
    void computeColorHistogram();
    void computeTextonHistogram();
    void clusterIntensities();
    void computeCenter();
    void computeMedianColor();
    void computeRMSContrast();
    void computeMichelsonContrast();
    void computeRobustMichelsonContrast();
    void computeErodedMask(const cv::Mat &mask, cv::Mat &erodedMask);

    cv::Mat image, textonLabels, mask, erodedMask;
    cv::Mat grayscaleImage;

    cv::Mat colorHistogram;
    cv::Mat intensityClusterCenters;
    cv::Mat textonHistogram;

    cv::Point2f center;
    cv::Vec3b medianColor;
    int erodedArea;

    float rmsContrast, michelsonContrast, robustMichelsonContrast;
    std::vector<int> intensities;

    RegionParams params;
};

#endif

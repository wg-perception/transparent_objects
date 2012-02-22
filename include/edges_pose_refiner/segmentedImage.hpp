#ifndef SEGMENTED_IMAGE_HPP__
#define SEGMENTED_IMAGE_HPP__

#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/region.hpp"

class SegmentedImage
{
  public:
    SegmentedImage();
    SegmentedImage(const cv::Mat &image);
    const std::vector<Region>& getRegions() const;
    const cv::Mat& getSegmentation() const;
    const cv::Mat& getOriginalImage() const;
    void showSegmentation(const std::string &title) const;
    void write(const std::string &filename) const;
    void read(const std::string &filename);
  private:
    static void oversegmentImage(const cv::Mat &image, cv::Mat &segmentation);
    static void mergeThinRegions(cv::Mat &segmentation, std::vector<int> &labels);
    static void segmentation2regions(const cv::Mat &image, cv::Mat &segmentation, const std::vector<cv::Mat> &filterBank, std::vector<Region> &regions);

    cv::Mat image, segmentation;
    std::vector<Region> regions;
    static std::vector<cv::Mat> filterBank;
};


void loadFilterBank(const std::string &filename, std::vector<cv::Mat> &filterBank);
void convolveImage(const cv::Mat &image, const std::vector<cv::Mat> &filterBank, cv::Mat &responses);

#endif

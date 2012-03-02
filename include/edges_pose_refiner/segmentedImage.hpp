#ifndef SEGMENTED_IMAGE_HPP__
#define SEGMENTED_IMAGE_HPP__

#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/region.hpp"

class SegmentedImage
{
  public:
    SegmentedImage();
    SegmentedImage(const cv::Mat &image, const std::string &segmentationFilename = "seg.txt");

    const std::vector<Region>& getRegions() const;
    const Region& getRegion(int regionIndex) const;

    const cv::Mat& getSegmentation() const;
    const cv::Mat& getOriginalImage() const;
    bool areRegionsAdjacent(int i, int j) const;
    void showSegmentation(const std::string &title) const;
    void showBoundaries(const std::string &title, const cv::Scalar &color = cv::Scalar(255, 0, 255)) const;
    void showTextonLabelsMap(const std::string &title) const;
    void write(const std::string &filename) const;
    void read(const std::string &filename);
  private:
    static void computeTextonLabels(const cv::Mat &image, cv::Mat &textonLabels);
    static void oversegmentImage(const cv::Mat &image, const std::string &segmentationFilename, cv::Mat &segmentation);
    static void mergeThinRegions(cv::Mat &segmentation, std::vector<int> &labels);
    static void segmentation2regions(const cv::Mat &image, cv::Mat &segmentation, cv::Mat &textonLabels, const std::vector<cv::Mat> &filterBank, std::vector<Region> &regions);

    cv::Mat image, segmentation;
    cv::Mat textonLabelsMap;
    cv::Mat regionAdjacencyMat;
    std::vector<Region> regions;
    static std::vector<cv::Mat> filterBank;
};


void loadFilterBank(const std::string &filename, std::vector<cv::Mat> &filterBank);
void convolveImage(const cv::Mat &image, const std::vector<cv::Mat> &filterBank, cv::Mat &responses);

#endif

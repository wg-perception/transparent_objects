/*
 * This code is from OpenCV library (implementation by Maria Dimashova, Itseez)
 *
 */

#include <opencv2/core/core.hpp>

namespace transpod
{
  /*
   GMM - Gaussian Mixture Model
  */
  class GMM
  {
  public:
      static const int componentsCount = 5;

      GMM( cv::Mat& _model );
      double operator()( const cv::Vec3d color ) const;
      double operator()( int ci, const cv::Vec3d color ) const;
      int whichComponent( const cv::Vec3d color ) const;

      void initLearning();
      void addSample( int ci, const cv::Vec3d color );
      void endLearning();

  private:
      void calcInverseCovAndDeterm( int ci );
      cv::Mat model;
      double* coefs;
      double* mean;
      double* cov;

      double inverseCovs[componentsCount][3][3];
      double covDeterms[componentsCount];

      double sums[componentsCount][3];
      double prods[componentsCount][3][3];
      int sampleCounts[componentsCount];
      int totalSampleCount;
  };

  void addSamples(const cv::Mat &img, const cv::Mat &mask, std::vector<cv::Vec3f> &bgdSamples, std::vector<cv::Vec3f> &fgdSamples);
  void initGMMs(std::vector<cv::Vec3f> &bgdSamples, std::vector<cv::Vec3f> &fgdSamples, GMM &bgdGMM, GMM &fgdGMM);
}

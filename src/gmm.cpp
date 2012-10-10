/*
 * This code is from OpenCV library (implementation by Maria Dimashova, Itseez)
 *
 */

#include <opencv2/opencv.hpp>
#include <edges_pose_refiner/gmm.hpp>

using namespace cv;

namespace transpod
{
  GMM::GMM( Mat& _model )
  {
      const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
      if( _model.empty() )
      {
          _model.create( 1, modelSize*componentsCount, CV_64FC1 );
          _model.setTo(Scalar(0));
      }
      else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
          CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

      model = _model;

      coefs = model.ptr<double>(0);
      mean = coefs + componentsCount;
      cov = mean + 3*componentsCount;

      for( int ci = 0; ci < componentsCount; ci++ )
          if( coefs[ci] > 0 )
               calcInverseCovAndDeterm( ci );
  }

  double GMM::operator()( const Vec3d color ) const
  {
      double res = 0;
      for( int ci = 0; ci < componentsCount; ci++ )
          res += coefs[ci] * (*this)(ci, color );
      return res;
  }

  double GMM::operator()( int ci, const Vec3d color ) const
  {
      double res = 0;
      if( coefs[ci] > 0 )
      {
          CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
          Vec3d diff = color;
          double* m = mean + 3*ci;
          diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
          double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
                     + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
                     + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
          res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
      }
      return res;
  }

  int GMM::whichComponent( const Vec3d color ) const
  {
      int k = 0;
      double max = 0;

      for( int ci = 0; ci < componentsCount; ci++ )
      {
          double p = (*this)( ci, color );
          if( p > max )
          {
              k = ci;
              max = p;
          }
      }
      return k;
  }

  void GMM::initLearning()
  {
      for( int ci = 0; ci < componentsCount; ci++)
      {
          sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
          prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
          prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
          prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
          sampleCounts[ci] = 0;
      }
      totalSampleCount = 0;
  }

  void GMM::addSample( int ci, const Vec3d color )
  {
      sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
      prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
      prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
      prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
      sampleCounts[ci]++;
      totalSampleCount++;
  }

  void GMM::endLearning()
  {
      const double variance = 0.01;
      for( int ci = 0; ci < componentsCount; ci++ )
      {
          int n = sampleCounts[ci];
          if( n == 0 )
              coefs[ci] = 0;
          else
          {
              coefs[ci] = (double)n/totalSampleCount;

              double* m = mean + 3*ci;
              m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;

              double* c = cov + 9*ci;
              c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
              c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
              c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];

              double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
              if( dtrm <= std::numeric_limits<double>::epsilon() )
              {
                  // Adds the white noise to avoid singular covariance matrix.
                  c[0] += variance;
                  c[4] += variance;
                  c[8] += variance;
              }

              calcInverseCovAndDeterm(ci);
          }
      }
  }

  void GMM::calcInverseCovAndDeterm( int ci )
  {
      if( coefs[ci] > 0 )
      {
          double *c = cov + 9*ci;
          double dtrm =
                covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);

          CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
          inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
          inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
          inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
          inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
          inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
          inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
          inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
          inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
          inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
      }
  }

  void addSamples(const Mat &img, const Mat &mask, std::vector<cv::Vec3f> &bgdSamples, std::vector<cv::Vec3f> &fgdSamples)
  {
      Point p;
      for( p.y = 0; p.y < img.rows; p.y++ )
      {
          for( p.x = 0; p.x < img.cols; p.x++ )
          {
              //Use only confident samples
              if( mask.at<uchar>(p) == GC_BGD )
                  bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
              if( mask.at<uchar>(p) == GC_FGD )
                  fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
          }
      }
  }

  //void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
  void initGMMs( vector<Vec3f> &bgdSamples, vector<Vec3f> &fgdSamples, GMM& bgdGMM, GMM& fgdGMM )
  {
      const int kMeansItCount = 10;
      const int kMeansType = KMEANS_PP_CENTERS;

      Mat bgdLabels, fgdLabels;
  /*
      vector<Vec3f> bgdSamples, fgdSamples;
      Point p;
      for( p.y = 0; p.y < img.rows; p.y++ )
      {
          for( p.x = 0; p.x < img.cols; p.x++ )
          {
              if( mask.at<uchar>(p) == GC_BGD )
                  bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
              if( mask.at<uchar>(p) == GC_FGD )
                  fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
          }
      }
  */
      CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
      Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
      kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
              TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
      Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
      kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
              TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

      bgdGMM.initLearning();
      for( int i = 0; i < (int)bgdSamples.size(); i++ )
          bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
      bgdGMM.endLearning();

      fgdGMM.initLearning();
      for( int i = 0; i < (int)fgdSamples.size(); i++ )
          fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
      fgdGMM.endLearning();
  }
}

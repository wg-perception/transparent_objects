/*
 * silhouette.hpp
 *
 *  Created on: Oct 17, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef SILHOUETTE_HPP_
#define SILHOUETTE_HPP_

#include <opencv2/core/core.hpp>
#include "edges_pose_refiner/poseRT.hpp"
#include "edges_pose_refiner/pinholeCamera.hpp"

//TODO: test it on other platforms before merging with the master branch
#include <tr1/unordered_map>

class EdgeModel;

typedef std::pair<int, int> GHKey;
const int GH_KEY_DIMENSION = 2;
typedef cv::Vec3i GHValue;
//TODO: experiment with different structures: unordered map, distance transform, FLANN
//typedef std::multimap<GHKey, GHValue> GHTable;
typedef std::tr1::unordered_multimap<GHKey, GHValue> GHTable;

namespace std
{
  namespace tr1
  {
    template <>
    struct hash<GHKey>
    {
      public:
        size_t operator()(const GHKey &key) const
        {
          //TODO: move up
          const int maxCoordinate = 100000;
          int keyNumber = key.first * maxCoordinate + key.second;
          std::tr1::hash<int> hasher;
          return hasher(keyNumber);
        }
      };
  }
}

//TODO: use robust statistics
class Silhouette
{
public:
  Silhouette();
  void init(const cv::Mat &edgels, const PoseRT &initialPose_cam);
  void getEdgels(cv::Mat &edgels) const;
  void getDownsampledEdgels(cv::Mat &edgels) const;
  void getInitialPose(PoseRT &pose_cam) const;

  int size() const;
  int getDownsampledSize() const;
  void clear();

  void generateGeometricHash(int silhouetteIndex, GHTable &hashTable, cv::Mat &canonicScale, float granularity, int hashBasisStep, float minDistanceBetweenPoints);

  void affine2poseRT(const EdgeModel &edgeModel, const PinholeCamera &camera, const cv::Mat &affineTransformation, bool useClosedFormPnP, PoseRT &pose_cam) const;

  void match(const cv::Mat &testEdgels, cv::Mat &silhouette2test, int icpIterationsCount, float min2dScaleChange) const;

  void camera2object(const cv::Mat &similarityTransformation_cam, cv::Mat &similarityTransformation_obj) const;

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;

  void visualizeSimilarityTransformation(const cv::Mat &similarityTransformation, cv::Mat &image, cv::Scalar color = cv::Scalar::all(255)) const;
  void draw(cv::Mat &image, cv::Scalar color = cv::Scalar::all(255), int thickness = 0) const;
private:
  void generateHashForBasis(int firstIndex, int secondIndex, cv::Mat &transformedEdgels);
  static void getNormalizationTransform(const cv::Mat &points, cv::Mat &normalizationTransform);
  static void findSimilarityTransformation(const cv::Mat &src, const cv::Mat &dst, cv::Mat &transformationMatrix, int iterationsCount, float min2dScaleChange);

  static void showNormalizedPoints(const cv::Mat &points, const std::string &title = "normalized points");

  cv::Mat edgels, downsampledEdgels;
  cv::Point2f silhouetteCenter;
  cv::Mat silhouette2normalized;

  //TODO: use smart pointer
//  const EdgeModel *edgeModel;
//  cv::Ptr<const PinholeCamera> camera;

  PoseRT initialPose_cam;
};

void findSimilarityTransformation(const cv::Point2f &pt1, const cv::Point2f &pt2, cv::Mat &similarityTransformation);

cv::Mat affine2homography(const cv::Mat &transformationMatrix);
cv::Mat homography2affine(const cv::Mat &homography);
void composeAffineTransformations(const cv::Mat &firstTransformation, const cv::Mat &secondTransformation, cv::Mat &composedTransformation);

#endif /* SILHOUETTE_HPP_ */

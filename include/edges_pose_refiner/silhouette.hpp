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

class EdgeModel;

class Silhouette
{
public:
  Silhouette();
  void init(const cv::Mat &edgels, const PoseRT &initialPose_cam);
  void getEdgels(cv::Mat &edgels) const;
  void getInitialPose(PoseRT &pose_cam) const;

  void clear();

  void affine2poseRT(const EdgeModel &edgeModel, const PinholeCamera &camera, const cv::Mat &affineTransformation, PoseRT &pose_cam) const;

  void match(const cv::Mat &testEdgels, cv::Mat &silhouette2test) const;
  void draw(cv::Mat &image) const;

  void read(const cv::FileNode &fn);
  void write(cv::FileStorage &fs) const;
private:
  static void getNormalizationTransform(const cv::Mat &points, cv::Mat &normalizationTransform);
  static void findSimilarityTransformation(const cv::Mat &src, const cv::Mat &dst, cv::Mat &transformationMatrix);

  static void showNormalizedPoints(const cv::Mat &points, const std::string &title = "normalized points");

  cv::Mat edgels;
  cv::Mat silhouette2normalized;

  //TODO: use smart pointer
//  const EdgeModel *edgeModel;
//  cv::Ptr<const PinholeCamera> camera;

  PoseRT initialPose_cam;
};

#endif /* SILHOUETTE_HPP_ */

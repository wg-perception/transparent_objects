/*
 * pose.hpp
 *
 *  Created on: Oct 10, 2011
 *      Author: Ilya Lysenkov
 */

#ifndef POSE_HPP_
#define POSE_HPP_

#include <opencv2/core/core.hpp>

struct PoseRT
{
public:
  cv::Mat rvec;
  cv::Mat tvec;

  PoseRT();
  PoseRT(const cv::Mat &projectiveMatrix);
  PoseRT(const cv::Mat &rotation, const cv::Mat &translation);
  PoseRT(const PoseRT &pose);
  PoseRT& operator=(const PoseRT &pose);


  cv::Mat getRvec() const;
  cv::Mat getTvec() const;
  cv::Mat getRotationMatrix() const;
  cv::Mat getProjectiveMatrix() const;
  cv::Mat getQuaternion() const;

  void setRotation(const cv::Mat &rotation);
  void setProjectiveMatrix(const cv::Mat &rt);

  PoseRT operator*(const PoseRT &pose) const;

  static PoseRT generateRandomPose(double rotationAngleInRadians, double translation);
  static void computeMeanPose(const std::vector<PoseRT> &poses, PoseRT &meanPose);
  static void computeDistance(const PoseRT &pose1, const PoseRT &pose2, double &rotationDistance, double &translationDistance, const cv::Mat &Rt_obj2cam = cv::Mat());
  static void computeObjectDistance(const PoseRT &pose1, const PoseRT &pose2, double &rotationDistance, double &translationDistance);

  PoseRT obj2cam(const cv::Mat &Rt_obj2cam);
  PoseRT inv() const;


  void write(const std::string &filename) const;
  void write(cv::FileStorage &fs) const;
  void read(const std::string &filename);
  void read(const cv::FileNode &node);
  friend std::ostream& operator<<(std::ostream& output, const PoseRT& pose);
private:
  int dim;
};

/*
void write(cv::FileStorage &fs, const std::string&, const PoseRT &pose)
{
  pose.write(fs);
}

void read(const cv::FileNode &node, PoseRT &pose, const PoseRT &defaultValue = PoseRT())
{
  pose.read(node);
}
*/

#endif /* POSE_HPP_ */

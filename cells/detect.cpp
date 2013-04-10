#include <ecto/ecto.hpp>
#include <boost/foreach.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <object_recognition_core/common/pose_result.h>
#include <object_recognition_core/db/ModelReader.h>

#include <edges_pose_refiner/poseEstimator.hpp>
#include <edges_pose_refiner/glassSegmentator.hpp>
#include <edges_pose_refiner/utils.hpp>
#include <edges_pose_refiner/pclProcessing.hpp>
#include <edges_pose_refiner/detector.hpp>
#include "db_transparent_objects.hpp"

#ifdef ADD_COLLISION_OBJECTS
#include <ros/ros.h>
#include <arm_navigation_msgs/CollisionObject.h>
#endif

//#define TRANSPARENT_DEBUG

using ecto::tendrils;
using ecto::spore;
using object_recognition_core::db::ObjectId;
using object_recognition_core::common::PoseResult;
using object_recognition_core::db::ObjectDbPtr;

namespace transparent_objects
{
  struct TransparentObjectsDetector: public object_recognition_core::db::bases::ModelReaderBase
  {
    void
    parameter_callback(const object_recognition_core::db::Documents & db_documents)
    {
      BOOST_FOREACH(const object_recognition_core::db::Document & document, db_documents)
          {
            transpod::PoseEstimator currentPoseEstimator;
            // Load the detector for that class
            document.get_attachment<transpod::PoseEstimator>("detector", currentPoseEstimator);

            std::string object_id = document.get_field<ObjectId>("object_id");
            detector_->addTrainObject(object_id, currentPoseEstimator);
            printf("Loaded %s\n", object_id.c_str());
          }
    }

    static void
    declare_params(tendrils& params)
    {
      object_recognition_core::db::bases::declare_params_impl(params, "TransparentObjects");
      params.declare(&TransparentObjectsDetector::registrationMaskFilename_, "registrationMaskFilename", "The filename of the registration mask.");
      params.declare(&TransparentObjectsDetector::visualize_, "visualize", "Visualize results", false);
      params.declare(&TransparentObjectsDetector::object_db_, "object_db", "The DB parameters").required(true);
//      params.declare(&LinemodDetector::threshold_, "threshold", "Matching threshold, as a percentage", 90.0f);
    }

    static void
    declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
    {
      inputs.declare(&TransparentObjectsDetector::K_, "K", "Intrinsics of the test camera.");
      inputs.declare(&TransparentObjectsDetector::color_, "image", "An rgb full frame image.");
      inputs.declare(&TransparentObjectsDetector::depth_, "depth", "The 16bit depth image.");
      inputs.declare(&TransparentObjectsDetector::cloud_, "points3d", "The scene cloud.");

      outputs.declare(&TransparentObjectsDetector::pose_results_, "pose_results", "The results of object recognition");
    }

    void
    configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
    {
      configure_impl();
      detector_ = new transpod::Detector;
    }

    int
    process(const tendrils& inputs, const tendrils& outputs)
    {
#ifdef TRANSPARENT_DEBUG
      cv::FileStorage fs("input.xml", cv::FileStorage::READ);
      if (fs.isOpened())
      {
        fs["K"] >> *K_;
        fs["image"] >> *color_;
        fs["depth"] >> *depth_;
        fs["points3d"] >> *cloud_;
      }
      else
      {
        cv::imwrite("color.png", *color_);
        cv::imwrite("depth.png", *depth_);
        cv::FileStorage outFs("input.xml", cv::FileStorage::WRITE);
        outFs << "K" << *K_;
        outFs << "image" << *color_;
        outFs << "depth" << *depth_;
        outFs << "points3d" << *cloud_;
        outFs.release();
      }
      fs.release();
#endif

      std::vector<PoseRT> poses;
      PinholeCamera camera(*K_, cv::Mat(), PoseRT(), color_->size());
      detector_->initialize(camera);
      std::vector<float> posesQualities;
      std::vector<std::string> detectedObjects;

      cv::Mat registrationMask = cv::imread(*registrationMaskFilename_, CV_LOAD_IMAGE_GRAYSCALE);
      transpod::Detector::DebugInfo debugInfo;
      try
      {
        detector_->detect(*color_, *depth_, registrationMask, poses, posesQualities, detectedObjects, &debugInfo);
      }
      catch(const cv::Exception &)
      {
      }

      if (*visualize_)
      {
        imshow("glass mask", debugInfo.glassMask);
        cv::Mat visualization = color_->clone();
        detector_->visualize(poses, detectedObjects, visualization);
        imshow("all detected objects", visualization);
        cv::waitKey(300);
#ifdef USE_3D_VISUALIZATION
        CV_Assert(false);
        //detector_->visualize(poses, detectedObjects, pclCloud);
#endif
      }

      pose_results_->clear();

      if (!posesQualities.empty())
      {
        std::vector<float>::iterator bestDetection = std::min_element(posesQualities.begin(), posesQualities.end());
        int bestDetectionIndex = std::distance(posesQualities.begin(), bestDetection);

        PoseResult pose_result;
        pose_result.set_R(poses[bestDetectionIndex].getRotationMatrix());
        pose_result.set_T(poses[bestDetectionIndex].getTvec());
        pose_result.set_object_id(*object_db_, detectedObjects[bestDetectionIndex]);
        pose_results_->push_back(pose_result);

#ifdef ADD_COLLISION_OBJECTS
      ros::NodeHandle nh;

      std::vector<cv::Vec3f> collisionBoxesDimensions;
      std::vector<PoseRT> collisionBoxesPoses;
      transpod::reconstructCollisionMap(camera, debugInfo.tablePlane, debugInfo.glassMask,
                              detector_->getModel(detectedObjects[bestDetectionIndex]), poses[bestDetectionIndex],
                              collisionBoxesDimensions, collisionBoxesPoses);

      //TODO: remove cubes if there are no cubes at all
      int n_cubes = collisionBoxesPoses.size();
      std::cout << "n cubes: " << n_cubes <<std::endl;
      ros::Publisher pub = nh.advertise<arm_navigation_msgs::CollisionObject>("/collision_object", 2, true);
      ros::Duration(1).sleep();
      // Remove old cubes from the map

      // TODO n_cubes should be based on geometry
      nan_cubes_.id = "nan_cube";
      nan_cubes_.operation.operation = arm_navigation_msgs::CollisionObjectOperation::ADD;
      nan_cubes_.header.frame_id = "/head_mount_kinect_rgb_optical_frame";
      nan_cubes_.header.stamp = ros::Time::now();

      nan_cubes_.shapes.clear();
      nan_cubes_.poses.clear();
      for(size_t i=0; i < n_cubes; ++i)
      {
        arm_navigation_msgs::Shape object;
        object.type = arm_navigation_msgs::Shape::BOX;
        object.dimensions.resize(3);
        // Radius + length
        object.dimensions[0] = collisionBoxesDimensions[i][0];
        object.dimensions[1] = collisionBoxesDimensions[i][1];
        object.dimensions[2] = collisionBoxesDimensions[i][2];

        geometry_msgs::Pose pose;
        // TODO should be close to the glass
        pose.position.x = collisionBoxesPoses[i].getTvec().at<double>(0);
        pose.position.y = collisionBoxesPoses[i].getTvec().at<double>(1);
        pose.position.z = collisionBoxesPoses[i].getTvec().at<double>(2);

        float angle = norm(collisionBoxesPoses[i].getRvec());
        pose.orientation.x = collisionBoxesPoses[i].getRvec().at<double>(0) * sin(angle / 2.0) / angle;
        pose.orientation.y = collisionBoxesPoses[i].getRvec().at<double>(1) * sin(angle / 2.0) / angle;
        pose.orientation.z = collisionBoxesPoses[i].getRvec().at<double>(2) * sin(angle / 2.0) / angle;
        pose.orientation.w = cos(angle / 2.0);

        nan_cubes_.shapes.push_back(object);
        nan_cubes_.poses.push_back(pose);
      }
      pub.publish(nan_cubes_);
      ros::Duration(1).sleep();
#endif
        if (*visualize_)
        {
          cv::Mat visualization = color_->clone();
          detector_->visualize(std::vector<PoseRT>(1, poses[bestDetectionIndex]),
                               std::vector<std::string>(1, detectedObjects[bestDetectionIndex]), visualization);
          imshow("the best object", visualization);
          cv::waitKey(300);
        }
      }

//      {
//        poses.clear();
//        poses.push_back(PoseRT((*rvecs_)[0], (*tvecs_)[0]));
//
//        cv::Mat visualization = color_->clone();
//        detector_->visualize(poses, *object_ids_, visualization);
//        imshow("detection", visualization);
//        cv::waitKey(300);
//        detector_->visualize(poses, *object_ids_, pclCloud);
//      }

      return ecto::OK;
    }


    // Parameters
    spore<std::string> registrationMaskFilename_;
    spore<bool> visualize_;

    // Inputs
    spore<cv::Mat> K_, color_, depth_, cloud_;

    /** The object recognition results */
    ecto::spore<std::vector<PoseResult> > pose_results_;
    /** The DB parameters */
    ecto::spore<ObjectDbPtr> object_db_;

    cv::Ptr<transpod::Detector> detector_;

#ifdef ADD_COLLISION_OBJECTS
    /** Keep track of the cubes to delete */
    arm_navigation_msgs::CollisionObject nan_cubes_;
#endif
  };
}

ECTO_CELL(transparent_objects_cells, transparent_objects::TransparentObjectsDetector, "Detector",
          "Detection of transparent objects.");

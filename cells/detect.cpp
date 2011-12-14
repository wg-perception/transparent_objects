#include <ecto/ecto.hpp>
#include <boost/foreach.hpp>

#include <object_recognition/db/ModelReader.h>

#include <edges_pose_refiner/poseEstimator.hpp>
#include <edges_pose_refiner/glassDetector.hpp>
#include <edges_pose_refiner/utils.hpp>
#include <edges_pose_refiner/pclProcessing.hpp>
#include <edges_pose_refiner/transparentDetector.hpp>
#include "db_transparent_objects.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#define TRANSPARENT_DEBUG
#define VISUALIZE_DETECTION

using ecto::tendrils;
using ecto::spore;
using object_recognition::db::ObjectId;

namespace transparent_objects
{
  struct TransparentObjectsDetector: public object_recognition::db::bases::ModelReaderImpl
  {
    void
    ParameterCallback(const object_recognition::db::Documents & db_documents)
    {
      std::cout << "detector: ParameterCallback" << std::endl;
      BOOST_FOREACH(const object_recognition::db::Document & document, db_documents)
          {
            PoseEstimator currentPoseEstimator;
            // Load the detector for that class
            document.get_attachment<PoseEstimator>("detector", currentPoseEstimator);

            std::string object_id = document.get_value<ObjectId>("object_id");
            detector_->addObject(object_id, currentPoseEstimator);
            printf("Loaded %s\n", object_id.c_str());
          }
    }

    static void
    declare_params(tendrils& params)
    {
      std::cout << "detector: declare_params" << std::endl;

//      params.declare(&LinemodDetector::threshold_, "threshold", "Matching threshold, as a percentage", 90.0f);
    }

    static void
    declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
    {
      inputs.declare(&TransparentObjectsDetector::K_, "K", "Intrinsics of the test camera.");
      inputs.declare(&TransparentObjectsDetector::color_, "image", "An rgb full frame image.");
      inputs.declare(&TransparentObjectsDetector::depth_, "depth", "The 16bit depth image.");
      inputs.declare(&TransparentObjectsDetector::cloud_, "points3d", "The scene cloud.");

      outputs.declare(&TransparentObjectsDetector::rvecs_, "Rs", "Rotations of detected objects");
      outputs.declare(&TransparentObjectsDetector::tvecs_, "Ts", "Translations of detected objects");
      outputs.declare(&TransparentObjectsDetector::object_ids_, "object_ids", "The ids of the objects");
    }

    void
    configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
    {
      std::cout << "detector: configure" << std::endl;
      detector_ = new TransparentDetector;
      std::cout << "detector: leaving configure" << std::endl;
    }

    int
    process(const tendrils& inputs, const tendrils& outputs)
    {
      std::cout << "detector: process" << std::endl;
      cv::Mat tmpDepth;
      cv::resize(*depth_, tmpDepth, cv::Size(640, 480));
      *depth_ = tmpDepth;

      cv::Mat tmpColor;
      cv::resize(*color_, tmpColor, cv::Size(640, 480));
      *color_ = tmpColor;


#ifdef TRANSPARENT_DEBUG
      cv::FileStorage fs("input.xml", cv::FileStorage::READ);
      fs["K"] >> *K_;
      fs["image"] >> *color_;
      fs["depth"] >> *depth_;
      fs["points3d"] >> *cloud_;
      fs.release();
      
      cv::imwrite("color.png", *color_);
      cv::imwrite("depth.png", *depth_);
      cv::imwrite("glass.png", glassMask);
      cv::FileStorage fs("input.xml", cv::FileStorage::WRITE);
      fs << "K" << *K_;
      fs << "image" << *color_;
      fs << "depth" << *depth_;
      fs << "points3d" << *cloud_;
      fs.release();

#endif

      assert(cloud_->channels() == 3);
      std::vector<cv::Point3f> cvCloud = cloud_->reshape(3, cloud_->total());
      pcl::PointCloud<pcl::PointXYZ> pclCloud;
      cv2pcl(cvCloud, pclCloud);

      std::vector<PoseRT> poses;
      PinholeCamera camera(*K_);
      detector_->initialize(camera);
      std::vector<float> posesQualities;
      std::vector<std::string> detectedObjects;
      detector_->detect(*color_, *depth_, pclCloud, poses, posesQualities, detectedObjects);

#ifdef VISUALIZE_DETECTION
      cv::Mat visualization = color_->clone();
      detector_->visualize(poses, detectedObjects, visualization);
      imshow("detection", visualization);
      cv::waitKey(300);
      detector_->visualize(poses, detectedObjects, pclCloud);
#endif

      std::vector<float>::iterator bestDetection = std::min_element(posesQualities.begin(), posesQualities.end());
      int bestDetectionIndex = std::distance(posesQualities.begin(), bestDetection);

      rvecs_->push_back(poses[bestDetectionIndex].getRotationMatrix());
      tvecs_->push_back(poses[bestDetectionIndex].getTvec());
      object_ids_->push_back(detectedObjects[bestDetectionIndex]);


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
//    spore<float> threshold_;
    // Inputs
    spore<cv::Mat> K_, color_, depth_, cloud_;

    // Outputs
    spore<std::vector<ObjectId> > object_ids_;
    spore<std::vector<cv::Mat> > rvecs_, tvecs_;

    cv::Ptr<TransparentDetector> detector_;
  };
}

ECTO_CELL(transparent_objects_cells, object_recognition::db::bases::ModelReaderBase<transparent_objects::TransparentObjectsDetector>, "Detector",
          "Detection of transparent objects.");

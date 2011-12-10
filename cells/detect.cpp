#include <ecto/ecto.hpp>
#include <boost/foreach.hpp>

#include <object_recognition/db/ModelReader.h>

#include <edges_pose_refiner/poseEstimator.hpp>
#include <edges_pose_refiner/glassDetector.hpp>
#include <edges_pose_refiner/utils.hpp>
#include "db_transparent_objects.hpp"


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
            // Load the detector for that class
            document.get_attachment<PoseEstimator>("poseEstimator", *poseEstimator_);

            std::string object_id = document.get_value<ObjectId>("object_id");
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
//      detector_ = linemod::getDefaultLINEMOD();
    }

    int
    process(const tendrils& inputs, const tendrils& outputs)
    {
      std::cout << "detector: process" << std::endl;

      int numberOfComponents;
      cv::Mat glassMask;
      findGlassMask(*color_, *depth_, numberOfComponents, glassMask);
//      assert(numberOfComponents == 1);

      std::vector<PoseRT> poses;
      std::vector<float> posesQualities;

      assert(cloud_->channels() == 3);
      std::vector<cv::Point3f> cvCloud = cloud_->reshape(3, cloud_->total());
      pcl::PointCloud<pcl::PointXYZ> pclCloud;
      cv2pcl(cvCloud, pclCloud);
      poseEstimator_->estimatePose(*color_, glassMask, pclCloud, poses, posesQualities);
      assert(!poses.empty());
      //TODO: add detection
      rvecs_->push_back(poses[0].getRvec());
      tvecs_->push_back(poses[0].getTvec());
      object_ids_->push_back("object_id");
      return ecto::OK;
    }


    // Parameters
//    spore<float> threshold_;
    // Inputs
    spore<cv::Mat> K_, color_, depth_, cloud_;

    // Outputs
    spore<std::vector<ObjectId> > object_ids_;
    spore<std::vector<cv::Mat> > rvecs_, tvecs_;

    cv::Ptr<PoseEstimator> poseEstimator_;
  };

} // namespace ecto_linemod

ECTO_CELL(transparent_objects_cells, object_recognition::db::bases::ModelReaderBase<transparent_objects::TransparentObjectsDetector>, "Detector",
          "Detection of transparent objects.");

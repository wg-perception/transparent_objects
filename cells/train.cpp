#include <ecto/ecto.hpp>

#include <boost/foreach.hpp>
#include <object_recognition_core/common/json.hpp>
#include <object_recognition_core/db/document.h>
#include <fstream>

#include "edges_pose_refiner/pinholeCamera.hpp"
#include "edges_pose_refiner/poseEstimator.hpp"
#include "edges_pose_refiner/utils.hpp"


using ecto::tendrils;
using ecto::spore;

namespace transparent_objects
{
  struct Trainer
  {
    static void
    declare_params(tendrils& params)
    {
      params.declare(&Trainer::json_submethod_, "json_submethod", "The submethod to use, as a JSON string.").required(
          true);
      params.declare(&Trainer::json_K_, "json_K", "Intrinsics of the test camera.").required(true);
      params.declare(&Trainer::json_D_, "json_D", "Distortion coefficients of the test camera.");
      params.declare(&Trainer::imageWidth_, "imageWidth", "Width of the test image", 640);
      params.declare(&Trainer::imageHeight_, "imageHeight", "Height of the test image", 480);
    }

    static void
    declare_io(const tendrils& params, tendrils& inputs, tendrils& outputs)
    {
      inputs.declare(&Trainer::document_, "document", "document with the object model.").required(true);

      outputs.declare(&Trainer::poseEstimator_, "detector", "The pose estimator.");
    }

    void
    configure(const tendrils& params, const tendrils& inputs, const tendrils& outputs)
    {
      std::cout  << __PRETTY_FUNCTION__ << *json_submethod_  <<std::endl;
      or_json::mValue submethod = object_recognition_core::to_json(*json_submethod_);
//      if (submethod.get_str() == "default")
      {
        {
          std::vector<float> K_value;
          for (size_t i = 0; i < object_recognition_core::to_json(*json_K_).get_array().size(); ++i)
          {
            K_value.push_back(object_recognition_core::to_json(*json_K_).get_array()[i].get_real());
          }

//          BOOST_FOREACH(const or_json::mValue & value, object_recognition_core::to_json(*json_K_).get_array())
//                K_value.push_back(value.get_real());

          K_ = cv::Mat(K_value).clone();
          K_ = (K_).reshape(1, 3);
        }
        {
          std::vector<float> D_value;
//          BOOST_FOREACH(const or_json::mValue & value, object_recognition_core::to_json(*json_D_).get_array())
//            D_value.push_back(value.get_real());
          for (size_t i = 0; i < object_recognition_core::to_json(*json_D_).get_array().size(); ++i)
          {
            D_value.push_back(object_recognition_core::to_json(*json_D_).get_array()[i].get_real());
          }

          if (D_value.empty())
          {
            const int distortionCoefficientsCount = 5;
            D_value.resize(distortionCoefficientsCount, 0.0f);
          }

          D_ = cv::Mat(D_value).clone();
        }

        PinholeCamera camera(K_, D_, PoseRT(), cv::Size(*imageWidth_, *imageHeight_));
        *poseEstimator_ = new transpod::PoseEstimator(camera);
      }
    }

    int
    process(const tendrils& inputs, const tendrils& outputs)
    {
      std::cout << "training..." << std::endl;
      // Get the binary file
      char buffer [L_tmpnam];
      char *p = std::tmpnam (buffer);
      assert(p != 0);
      std::string file_name = std::string(buffer) + ".ply";
      std::stringstream ss;
      document_->get_attachment_stream("cloud.ply", ss); 

      std::ofstream writer(file_name.c_str());
      writer << ss.rdbuf();

      std::vector<cv::Point3f> points;
      std::vector<cv::Point3f> normals;
      std::vector<cv::Point3i> colors;
      //TODO: use the ply reader from PCL when a new version will be available in ROS
      readPointCloud(file_name, points, colors, normals);


      //EdgeModel edgeModel(points, std::vector<cv::Point3f>(), false, false);
      //TODO: estimate normals from a full point cloud, not a downsampled one
      EdgeModel edgeModel(points, false, false);
      assert(!poseEstimator_->empty());
      (*poseEstimator_)->setModel(edgeModel);
      std::cout << "done." << std::endl;
      return ecto::OK;
    }

    cv::Mat K_, D_;
    spore<std::string> json_K_, json_D_;
    spore<object_recognition_core::db::Document> document_;
    spore<cv::Ptr<transpod::PoseEstimator> > poseEstimator_;
    spore<std::string> json_submethod_;
    spore<int> imageWidth_, imageHeight_;
  };
}

ECTO_CELL(transparent_objects_cells, transparent_objects::Trainer, "Trainer", "Train the transparent objects detection and pose estimation algorithm.");

#include <ecto/ecto.hpp>

#include <boost/foreach.hpp>
#include <object_recognition/common/json.hpp>
#include <object_recognition/db/db.h>
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
      params.declare(&Trainer::json_K_, "json_K", "Intrinsics of the camera.").required(true);
      params.declare(&Trainer::json_D_, "json_D", "Distortion coefficients of the camera.");
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
      or_json::mValue submethod = object_recognition::to_json(*json_submethod_);
      if (submethod.get_str() == "default")
      {
        {
          std::vector<float> K_value;
          BOOST_FOREACH(const or_json::mValue & value, object_recognition::to_json(*json_K_).get_array())
            K_value.push_back(value.get_real());
          *K_ = cv::Mat_<float>(K_value);
          *K_ = (*K_).reshape(1, 3);
        }
        {
          std::vector<float> D_value;
          BOOST_FOREACH(const or_json::mValue & value, object_recognition::to_json(*json_D_).get_array())
            D_value.push_back(value.get_real());
          *D_ = cv::Mat_<float>(D_value);
        }

        PinholeCamera camera(*K_, *D_);
        *poseEstimator_ = new PoseEstimator(camera);
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

      std::vector<cv::Point3f> points, normals;
      std::vector<cv::Point3i> colors;
      readPointCloud(file_name, points, colors, normals);

      EdgeModel edgeModel(points, false);
      (*poseEstimator_)->addObject(edgeModel);
      std::cout << "done." << std::endl;
      return ecto::OK;
    }

    spore<cv::Mat> K_, D_;
    spore<std::string> json_K_, json_D_;
    spore<object_recognition::db::Document> document_;
    spore<cv::Ptr<PoseEstimator> > poseEstimator_;
    spore<std::string> json_submethod_;
  };
} 

ECTO_CELL(transparent_objects_cells, transparent_objects::Trainer, "Trainer", "Train the transparent objects detection and pose estimation algorithm.");

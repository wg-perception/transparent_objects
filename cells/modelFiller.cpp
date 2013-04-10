#include <ecto/ecto.hpp>
#include <opencv2/core/core.hpp>

#include <object_recognition_core/common/types.h>
#include <object_recognition_core/db/document.h>

#include "db_transparent_objects.hpp"
#include <edges_pose_refiner/poseEstimator.hpp>

using object_recognition_core::db::ObjectId;
using object_recognition_core::db::Document;

namespace transparent_objects
{
  struct ModelFiller
  {
  public:
    static void
    declare_io(const ecto::tendrils& params, ecto::tendrils& inputs, ecto::tendrils& outputs)
    {
      typedef ModelFiller C;
      inputs.declare(&C::detector_, "detector", "The transparent objects detector.").required(true);

      outputs.declare(&C::db_document_, "db_document", "The filled document.");
    }

    int
    process(const ecto::tendrils& inputs, const ecto::tendrils& outputs)
    {
      db_document_->set_attachment<transpod::PoseEstimator>("detector", **detector_);
      return ecto::OK;
    }

  private:
    ecto::spore<cv::Ptr<transpod::PoseEstimator> > detector_;
    ecto::spore<Document> db_document_;
  };
}

ECTO_CELL(transparent_objects_cells, transparent_objects::ModelFiller, "ModelFiller",
          "Populates a db document with a PoseEstimator for persisting a later date.")

#include <ecto/ecto.hpp>
#include <boost/foreach.hpp>

#include <object_recognition/db/ModelReader.h>

#include <edges_pose_refiner/poseEstimator.hpp>
#include <edges_pose_refiner/glassDetector.hpp>
#include <edges_pose_refiner/utils.hpp>
#include <edges_pose_refiner/pclProcessing.hpp>
#include "db_transparent_objects.hpp"

#include <opencv2/highgui/highgui.hpp>

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
            // Load the detector for that class
            document.get_attachment<PoseEstimator>("detector", *poseEstimator_);

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
      poseEstimator_ = new PoseEstimator(PinholeCamera());
      std::cout << "detector: leaving configure" << std::endl;
    }

    int
    process(const tendrils& inputs, const tendrils& outputs)
    {
      std::cout << "detector: process" << std::endl;

#ifdef TRANSPARENT_DEBUG
      cv::FileStorage fs("input.xml", cv::FileStorage::READ);
      fs["K"] >> *K_;
      fs["image"] >> *color_;
      fs["depth"] >> *depth_;
      fs["points3d"] >> *cloud_;
      fs.release();
#endif

      assert(cloud_->channels() == 3);
      std::vector<cv::Point3f> cvCloud = cloud_->reshape(3, cloud_->total());
      pcl::PointCloud<pcl::PointXYZ> pclCloud;
      cv2pcl(cvCloud, pclCloud);

      cv::Vec4f tablePlane;
      pcl::PointCloud<pcl::PointXYZ> tableHull;
      int kSearch = 10;
      float distanceThreshold = 0.02f;
      std::cout << "WARNING: hard-coded parameters" << std::endl;
      //TODO: fix
      bool isEstimated = computeTableOrientation(kSearch, distanceThreshold, pclCloud, tablePlane, &tableHull);
      if (!isEstimated)
      {
        std::cerr << "Cannot find a table plane" << std::endl;
        return ecto::OK;
      }
      else
      {
        std::cout << "table plane is estimated" << std::endl;
      }

      int numberOfComponents;
      cv::Mat glassMask;
      GlassSegmentator glassSegmentator;
      PinholeCamera camera(*K_);
      glassSegmentator.segment(*color_, *depth_, numberOfComponents, glassMask, &camera, &tablePlane, &tableHull);

#ifdef VISUALIZE_DETECTION
      cv::Mat segmentation = drawSegmentation(*color_, glassMask);
      imshow("glassMask", glassMask);
      imshow("segmentation", segmentation);
      cv::waitKey(100);
#endif
      
#ifdef TRANSPARENT_DEBUG
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


//      assert(numberOfComponents == 1);

      std::vector<PoseRT> poses;
      std::vector<float> posesQualities;


#ifdef TRANSPARENT_DEBUG
/*
      cv::Mat D = cv::Mat::zeros(5, 1, CV_32FC1);
      PinholeCamera camera(*K_, D);
      poseEstimator_ = new PoseEstimator(camera);

      std::vector<cv::Point3f> points, normals;
      std::vector<cv::Point3i> colors;
      readPointCloud("cloud.ply", points, colors, normals);

      EdgeModel edgeModel(points, false);
      poseEstimator_->addObject(edgeModel);
*/      
#endif
      assert(!poseEstimator_.empty());
      std::cout << "starting to estimate pose..." << std::endl;
      poseEstimator_->estimatePose(*color_, glassMask, pclCloud, poses, posesQualities, &tablePlane);
      std::cout << "done." << std::endl;
      if (poses.empty())
      {
        std::cerr << "Cannot estimate a pose" << std::endl;
        return ecto::OK;
      }

      //TODO: add detection
      rvecs_->push_back(poses[0].getRvec());
      tvecs_->push_back(poses[0].getTvec());
      object_ids_->push_back("object_id");

#ifdef VISUALIZE_DETECTION
      poseEstimator_->visualize(*color_, poses[0]);
      cv::waitKey(100);
      poseEstimator_->visualize(pclCloud, poses[0]);
#endif

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

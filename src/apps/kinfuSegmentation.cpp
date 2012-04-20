/*
 * kinfuSegmentation.cpp
 *
 *  Created on: Nov 18, 2011
 *      Author: Ilya Lysenkov
 */

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <string>
#include <opencv2/core/core.hpp>

#include "edges_pose_refiner/utils.hpp"

#include <pcl/visualization/cloud_viewer.h>


#include "pcl/ModelCoefficients.h"

#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"

#include "pcl/sample_consensus/method_types.h"
#include "pcl/sample_consensus/model_types.h"

#include "pcl/filters/passthrough.h"
#include "pcl/filters/project_inliers.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>

#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/segmentation/extract_polygonal_prism_data.h"
#include <pcl/segmentation/extract_clusters.h>

#include "pcl/surface/convex_hull.h"

#include "pcl/registration/registration.h"

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

#include <pcl/common/transform.h>

#include "edges_pose_refiner/pclProcessing.hpp"


using std::string;
using std::vector;
using std::cout;
using std::endl;

void read(const string &filename, pcl::PointCloud<pcl::PointXYZ> &cloud)
{
  vector<cv::Point3f> cvSceneCloud;
  readPointCloud(filename, cvSceneCloud);
  cv2pcl(cvSceneCloud, cloud);
}

int main(int argc, char *argv[])
{
  const float downLeafSize = 0.001f;
  const int kSearch = 10;
  const float distanceThreshold = 0.01f;
  const double minZ = 0.007;
  const double maxZ = 0.5;


  if (argc != 2)
  {
    cout << argv[0] << " <scene_cloud.ply>" << endl;
    return 0;
  }

  string sceneFilename = argv[1];

  pcl::PointCloud<pcl::PointXYZ> sceneCloud;
  read(sceneFilename, sceneCloud);
  cout << "all points: " << sceneCloud.points.size() << endl;

//  downsample(downLeafSize, sceneCloud);
//  cout << "down points: " << sceneCloud.points.size() << endl;

  pcl::PointCloud<pcl::Normal> sceneNormals;
  estimateNormals(kSearch, sceneCloud, sceneNormals);
  cout << "normals: " << sceneNormals.points.size() << endl;

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  segmentTable(distanceThreshold, sceneCloud, sceneNormals, inliers, coefficients);
  cout << "inliers: " << inliers->indices.size () << endl;

  pcl::PointCloud<pcl::PointXYZ> projectedInliers;
  projectInliersOnTable(sceneCloud, inliers, coefficients, projectedInliers);

  pcl::PointCloud<pcl::PointXYZ> tableHull;
  reconstructConvexHull(projectedInliers, tableHull);

  rotateTable(coefficients, sceneCloud, projectedInliers, tableHull);

  pcl::PointIndices::Ptr objectsIndices(new pcl::PointIndices);
  segmentObjects(minZ, maxZ, sceneCloud, tableHull, objectsIndices);

  pcl::PointCloud<pcl::PointXYZ> objectsInScene;
  extractPointCloud(sceneCloud, objectsIndices, objectsInScene);
  cout << "objects points: " << objectsInScene.points.size() << endl;

  pcl::PointCloud<pcl::PointXYZ> tablePlane;
  extractPointCloud(sceneCloud, inliers, tablePlane);

  pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
  viewer.showCloud(objectsInScene.makeShared(), "objectsInScene");
  //viewer.showCloud (tablePlane, "plane");
  while (!viewer.wasStopped ())
  {
  }

  vector<cv::Point3f> objects;
  pcl2cv(objectsInScene, objects);
  writePointCloud("objects.asc", objects);

  return 0;
}

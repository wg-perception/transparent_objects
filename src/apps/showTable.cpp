/*
 * showTable.cpp
 *
 *  Created on: 12/15/2012
 *      Author: ilysenkov
 */

#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/tableSegmentation.hpp"
#include "edges_pose_refiner/detector.hpp"

#include <opencv2/rgbd/rgbd.hpp>
#include <iomanip>
#include <omp.h>

using namespace cv;
using std::cout;
using std::endl;

//TODO: merge this program with runDetectionOnDataset

int main(int argc, char *argv[])
{
    CV_Assert(argc == 3);
    const string baseFolder = argv[1];
    const string objectName = argv[2];
    const string testFolder = baseFolder + "/" + objectName + "/";

    TODBaseImporter dataImporter(baseFolder, testFolder);
    vector<int> testIndices;
    PinholeCamera camera;
    Mat registrationMask;
    dataImporter.importAllData(0, 0, &camera, &registrationMask, 0, &testIndices);


    omp_set_num_threads(7);
#pragma omp parallel for
    for(size_t _testIdx = 0; _testIdx < testIndices.size(); ++_testIdx)
    {
        std::cout << _testIdx << std::endl;
        int testImageIndex = testIndices[_testIdx];

        std::stringstream filename;
        filename << "table_" << std::setw(5) << std::setfill('0') << testImageIndex << ".png";

        Mat bgrImage, depth;
        dataImporter.importBGRImage(testImageIndex, bgrImage);
        dataImporter.importDepth(testImageIndex, depth);

        {
            Vec4f tablePlane;
            vector<Point> tableHull;
            computeTableOrientationByRGBD(depth, camera, tablePlane, &tableHull);
            Mat rgbdVisualization = bgrImage.clone();
            polylines(rgbdVisualization, tableHull, true, Scalar(255, 0, 255));
            imwrite(filename.str() + ".rgbd.png", rgbdVisualization);
        }
/*
        {
            Mat points3d;
            depthTo3d(depth, camera.cameraMatrix, points3d);
            vector<Point3f> cvCloud = points3d.reshape(3, points3d.total());

            transpod::PCLPlaneSegmentationParams params;
            Vec4f tablePlane;
            vector<Point2f> tableHull;
            computeTableOrientationByPCL(params.downLeafSize, params.kSearch, params.distanceThreshold,
                                         cvCloud, tablePlane, &camera, &tableHull);
            Mat tableHull_int;
            Mat(tableHull).convertTo(tableHull_int, CV_32SC2);

            Mat pclVisualization = bgrImage.clone();
            polylines(pclVisualization, tableHull_int, true, Scalar(255, 0, 255));
            imwrite(filename.str() + ".pcl.png", pclVisualization);
        }
*/
    }
}

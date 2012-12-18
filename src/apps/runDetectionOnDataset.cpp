/*
 * runDetectionOnDataset.cpp
 *
 *  Created on: 12/16/2012
 *      Author: ilysenkov
 */

#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/detector.hpp"
#include "edges_pose_refiner/TODBaseImporter.hpp"

#include "edges_pose_refiner/pclProcessing.hpp"

#include <iomanip>
//#include <omp.h>

//#define RUN_ONLY_SEGMENTATION

using namespace cv;
using namespace transpod;
using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
    std::system("date");

    if (argc != 6)
    {
        cout << argv[0] << " <modelsPath> <baseFoldler> <testObjectName> <startIndex> <endIndex>" << endl;
        return -1;
    }

    const string trainedModelsPath = argv[1];
    const string baseFolder = argv[2];
    const string testObjectName = argv[3];
    const int startIndex = atoi(argv[4]);
    const int endIndex = atoi(argv[5]);

    const string testFolder = baseFolder + "/" + testObjectName + "/";
    const string visualizationPath = "/shared/hotels/visualized_results/";
    vector<string> objectNames;
    objectNames.push_back(testObjectName);

    cout << "writing to " << visualizationPath << endl;

    TODBaseImporter dataImporter(baseFolder, testFolder);

    PinholeCamera kinectCamera;
    vector<EdgeModel> edgeModels;
    vector<int> testIndices;
    Mat registrationMask;
    dataImporter.importAllData(&trainedModelsPath, &objectNames, &kinectCamera, &registrationMask, &edgeModels, &testIndices);

    DetectorParams params;
    params.planeSegmentationMethod = RGBD;
    params.glassSegmentationParams.grabCutErosionsIterations = 3;
    params.glassSegmentationParams.grabCutDilationsIterations = 3;

#ifndef RUN_ONLY_SEGMENTATION
    Detector detector(kinectCamera, params);
    for (size_t i = 0; i < edgeModels.size(); ++i)
    {
        detector.addTrainObject(objectNames[i], edgeModels[i]);
    }
#endif

    for(size_t _testIdx = startIndex; _testIdx < endIndex; ++_testIdx)
    {
        int testImageIndex = testIndices[_testIdx];
        cout << "Test: " << _testIdx << " " << testImageIndex << endl;
        Mat bgrImage, depth;
        dataImporter.importBGRImage(testImageIndex, bgrImage);
        dataImporter.importDepth(testImageIndex, depth);

        TickMeter recognitionTime;
        recognitionTime.start();

#ifdef RUN_ONLY_SEGMENTATION
        cv::Vec4f tablePlane;
        std::vector<cv::Point2f> tableHull;
        std::vector<Point> tableHullInt;
        bool isEstimated = computeTableOrientationByRGBD(depth, kinectCamera, tablePlane, &tableHullInt, params.pclPlaneSegmentationParams.verticalDirection);
        CV_Assert(isEstimated);
        for (size_t i = 0; i < tableHullInt.size(); ++i)
        {
            tableHull.push_back(tableHullInt[i]);
        }

        GlassSegmentator glassSegmentator(params.glassSegmentationParams);
        int numberOfComponents;
        Mat glassMask;
        glassSegmentator.segment(bgrImage, depth, registrationMask, numberOfComponents, glassMask, &tableHull);
#else
        vector<PoseRT> poses_cam;
        vector<float> posesQualities;
        vector<string> detectedObjectsNames;
        Detector::DebugInfo debugInfo;
        try
        {
            detector.detect(bgrImage, depth, registrationMask, Mat(), poses_cam, posesQualities, detectedObjectsNames, &debugInfo);
        }
        catch(const cv::Exception &)
        {
        }
        recognitionTime.stop();
        cout << "Recognition time: " << recognitionTime.getTimeSec() << "s" << endl;
        if (!posesQualities.empty())
        {
            cout << "quality: " << posesQualities[0] << endl;
        }

        Mat glassMask = debugInfo.glassMask;
#endif
        std::stringstream str;
        str << std::setw(5) << std::setfill('0') << testImageIndex;
        Mat segmentation = drawSegmentation(bgrImage, glassMask);
        imwrite(visualizationPath + "/" + objectNames[0] + "_" + str.str() + "_mask.png", segmentation);

#ifndef RUN_ONLY_SEGMENTATION
        Mat poseImage = bgrImage.clone();
        detector.visualize(poses_cam, detectedObjectsNames, poseImage);
        imwrite(visualizationPath + "/" + objectNames[0] + "_" + str.str() + "_pose.png", poseImage);
#endif

        const float depthNormalizationFactor = 100;
        imwrite(visualizationPath + "/" + objectNames[0] + "_" + str.str() + "_depth.png", depth * depthNormalizationFactor);
    }

    std::system("date");
}


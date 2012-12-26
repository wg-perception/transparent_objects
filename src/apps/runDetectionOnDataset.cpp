/*
 * runDetectionOnDataset.cpp
 *
 *  Created on: 12/16/2012
 *      Author: ilysenkov
 */

#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/detector.hpp"
#include "edges_pose_refiner/TODBaseImporter.hpp"

#include "edges_pose_refiner/tableSegmentation.hpp"

#include <iomanip>
#include <fstream>
#include <omp.h>
#include <sys/stat.h>

//#define RUN_ONLY_SEGMENTATION

using namespace cv;
using namespace transpod;
using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        cout << "Use the following format to run the program:" << endl << endl;
        cout << '\t' << argv[0] << " [-j numberOfThreads] <modelsFolder> <testFolder> <resultsFolder> objectName..."<< endl << endl;
        cout << "numberOfThreads\t\tNumber of OpenMP threads to use" << endl;
        cout << "modelsFolder\t\tFolder where trained models are stored" << endl;
        cout << "testFolder\t\tFolder with test data" << endl;
        cout << "resultsFolder\t\tFolder where results should be saved" << endl;
        cout << "objectName\t\tOne or several names of objects (separated by spaces) to detect" << endl;

        return -1;
    }
    std::system("date");

    bool isNumberOfThreadsPassed = strcmp(argv[1], "-j") == 0;
    const int numberOfThreads = isNumberOfThreadsPassed ? atoi(argv[2]) : 1;
    CV_Assert(numberOfThreads > 0);
    const int optionsShift = isNumberOfThreadsPassed ? 3 : 1;

    const string trainedModelsPath = argv[0 + optionsShift];
    const string testFolder        = argv[1 + optionsShift];
    const string visualizationPath = argv[2 + optionsShift];
    vector<string> objectNames;
    for (int i = 3 + optionsShift; i < argc; ++i)
    {
        objectNames.push_back(argv[i]);
    }
    const string baseFolder = testFolder + "/../";
    const string qualitiesFilename = visualizationPath + "/qualities.txt";
    omp_set_num_threads(numberOfThreads);

    cout << "Results will be stored at " << visualizationPath << endl;
    struct stat st = {0};
    if (stat(visualizationPath.c_str(), &st) == -1)
    {
        string mkdirCommand = "mkdir -p " + visualizationPath;
        int returnCode = std::system(mkdirCommand.c_str());
        if (returnCode != 0)
        {
            CV_Error(CV_StsBadArg, "Cannot create the results folder: " + visualizationPath);
        }
    }

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
        cout << "Training the detector for " << objectNames[i] << "...  " << std::flush;
        detector.addTrainObject(objectNames[i], edgeModels[i]);
        cout << "done." << endl;
    }
#endif

    vector<float> allQualities(testIndices.size(), std::numeric_limits<float>::max());
    cout << "Started detection" << endl;
    const int numberOfChunksPerThread = 20;
    int chunkSize = std::max(1, static_cast<int>(testIndices.size()) / (numberOfThreads * numberOfChunksPerThread));
#pragma omp parallel for schedule(dynamic, chunkSize)
    for(size_t _testIdx = 0; _testIdx < testIndices.size(); ++_testIdx)
    {
        int testImageIndex = testIndices[_testIdx];
        Mat bgrImage, depth;
        dataImporter.importBGRImage(testImageIndex, bgrImage);
        dataImporter.importDepth(testImageIndex, depth);
        vector<Point3f> cloud;

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
            detector.detect(bgrImage, depth, registrationMask, cloud, poses_cam, posesQualities, detectedObjectsNames, &debugInfo);
        }
        catch(const cv::Exception &)
        {
        }
        recognitionTime.stop();

        std::stringstream statusMessage;
        const int imageIndexWidth = 5;
        const int precision = 3;
        const int timeWidth = precision + 4;
        statusMessage << "Proccessed image " << std::setw(imageIndexWidth) << testImageIndex <<
                         " in " << std::fixed << std::setprecision(precision) << std::setw(timeWidth) << recognitionTime.getTimeSec() << " seconds" << endl;
        statusMessage << "Object errors:";
        //TODO: for several objects print +inf if one of them was not detected
        for (size_t i = 0; i < posesQualities.size(); ++i)
        {
            statusMessage << " " << posesQualities[i];
        }
        cout << statusMessage.str() << endl << endl;

        if (!posesQualities.empty() && objectNames.size() == 1)
        {
            allQualities[_testIdx] = posesQualities[0];
        }

        Mat glassMask = debugInfo.glassMask;
        if (glassMask.empty())
        {
            glassMask = Mat(bgrImage.size(), CV_8UC1, Scalar(0));
        }
#endif
        std::stringstream imageIndex;
        const int indexWidth = 5;
        imageIndex << std::setw(indexWidth) << std::setfill('0') << testImageIndex;
        Mat segmentation = drawSegmentation(bgrImage, glassMask);
        imwrite(visualizationPath + "/image_" + imageIndex.str() + "_segmentation.png", segmentation);

#ifndef RUN_ONLY_SEGMENTATION
        Mat detectionImage = bgrImage.clone();
        detector.visualize(poses_cam, posesQualities, detectedObjectsNames, detectionImage, &debugInfo);
        string detectionFilename = visualizationPath + "/image_" + imageIndex.str() + "_detection.png";
        bool isSuccess = imwrite(detectionFilename, detectionImage);
        if (!isSuccess)
        {
            CV_Error(CV_StsBadArg, "Cannot write to " + detectionFilename);
        }
#endif

        const float depthNormalizationFactor = 100;
        imwrite(visualizationPath + "/image_" + imageIndex.str() + "_depth.png", depth * depthNormalizationFactor);
    }

    if (objectNames.size() == 1)
    {
        std::ofstream fout(qualitiesFilename.c_str());
        if (!fout.is_open())
        {
            CV_Error(CV_StsBadArg, "Cannot write to " + qualitiesFilename);
        }
        for(size_t _testIdx = 0; _testIdx < testIndices.size(); ++_testIdx)
        {
            int testImageIndex = testIndices[_testIdx];
            fout << testImageIndex << " " << allQualities[_testIdx] << '\n';
        }
        fout.close();
    }

    std::system("date");
    return 0;
}

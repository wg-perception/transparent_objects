/*
 * reconstructObject.cpp
 *
 *  Created on: 1/14/2013
 *      Author: ilysenkov
 */

#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "ilpProblem.hpp"

//#define INITIAL_RUN

using namespace cv;
using std::cout;
using std::endl;

void getGroundTruthData(const string &trainedModelsPath, const string &baseFolder, const string &testObjectName,
                        PinholeCamera &camera, std::vector<PoseRT> &poses, std::vector<Mat> &masks,
                        std::vector<cv::Point3f> &groundTruthModel);

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cout << argv[0] << " <trainedModelsPath> <baseFolder> <objectName>" << endl;
        return 1;
    }

    const string trainedModelsPath = argv[1];
    const string baseFolder = argv[2];
    const string testObjectName = argv[3];

    PinholeCamera camera;
    vector<PoseRT> allPoses;
    vector<Mat> allMasks;
    vector<Point3f> groundTruthModel;
    getGroundTruthData(trainedModelsPath, baseFolder, testObjectName, camera, allPoses, allMasks, groundTruthModel);

//TODO: remove
#if 0
{
    Mat volumePoints_Vector(1, volumePoints.total(), CV_32FC3, volumePoints.data);
    for (size_t imageIndex = 0; imageIndex < allMasks.size(); ++imageIndex)
    {
        const Mat &mask = allMasks[imageIndex];
        Mat visualization;
        cvtColor(mask, visualization, CV_GRAY2BGR);
        CV_Assert(mask.type() == CV_8UC1);
        const PoseRT &pose = allPoses[imageIndex];

        vector<Point2f> projectedVolume;
        camera.projectPoints(volumePoints_Vector, pose, projectedVolume);

        drawPoints(projectedVolume, visualization, Scalar(0, 255, 0));
        imshow("viz", visualization);
        waitKey();
    }
}
    return 0;
#endif

#ifdef INITIAL_RUN
    ILPProblem ilpProblem(VolumeParams(), camera, allPoses, allMasks);
    ilpProblem.write("ilpProblem.txt");
    ilpProblem.writeLP("ilpProblem.lp");
#else
    ILPProblem ilpProblem(VolumeParams(), camera);
//    ilpProblem.read("ilpProblem.txt", "solution.csv");
    ilpProblem.read("ilpProblem.txt", "solution.txt");
    ilpProblem.setGroundTruthModel(groundTruthModel);

    vector<Point3f> model;
    ilpProblem.getModel(model);
    writePointCloud("model.asc", model);

//    ilpProblem.visualize(allPoses, allMasks);
    ilpProblem.visualizeVolumeVariables();
#endif

    return 0;
}

void getGroundTruthData(const string &trainedModelsPath, const string &baseFolder, const string &testObjectName,
                        PinholeCamera &camera, std::vector<PoseRT> &poses, std::vector<Mat> &masks,
                        std::vector<cv::Point3f> &groundTruthModel)
{
    const string testFolder = baseFolder + "/" + testObjectName + "/";

    vector<string> objectNames;
    objectNames.push_back(testObjectName);

    TODBaseImporter baseImporter(baseFolder, testFolder);
    vector<EdgeModel> allEdgeModels;
    vector<int> testIndices;
    baseImporter.importAllData(&trainedModelsPath, &objectNames, &camera, 0, &allEdgeModels, &testIndices);
    poses.clear();
    masks.clear();
    groundTruthModel.clear();
    for (size_t _testIdx = 0; _testIdx < testIndices.size(); ++ _testIdx)
    {
        int imageIndex = testIndices[_testIdx];
        PoseRT fiducialPose, offset;
        baseImporter.importGroundTruth(imageIndex, fiducialPose, false, &offset);

        if (groundTruthModel.empty())
        {
            project3dPoints(allEdgeModels[0].points, offset, groundTruthModel);
        }
//        PoseRT model2test = fiducialPose * offset;

        //TODO: wrap it as a single function
        vector<Point2f> projectedPoints;
//        camera.projectPoints(groundTruthModel, model2test, projectedPoints);
        camera.projectPoints(groundTruthModel, fiducialPose, projectedPoints);
        //TODO: move up
        const Size imageSize(640, 480);
        const float downFactor = 1.0f;
        const int closingIterationsCount = 5;
        bool cropMask = false;
        Point tl;
        Mat objectMask;
        EdgeModel::computePointsMask(projectedPoints, imageSize, downFactor, closingIterationsCount, objectMask, tl, cropMask);

        poses.push_back(fiducialPose);
        masks.push_back(objectMask);
    }

}

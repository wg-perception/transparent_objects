/*
 * reconstructObject.cpp
 *
 *  Created on: 1/14/2013
 *      Author: ilysenkov
 */

#include <opencv2/opencv.hpp>
#include <fstream>
#include "edges_pose_refiner/TODBaseImporter.hpp"

using namespace cv;
using std::cout;
using std::endl;

void getGroundTruthData(const string &trainedModelsPath, const string &baseFolder, const string &testObjectName,
                        PinholeCamera &camera, std::vector<PoseRT> &poses, std::vector<Mat> &masks);


struct VolumeParams
{
  cv::Vec3f minBound, maxBound, step;

  VolumeParams()
  {
    minBound = cv::Vec3f(0.1f, -0.1f, -0.3f);
    maxBound = cv::Vec3f(0.5f,  0.2f,  0.0f);
//    step = cv::Vec3f::all(0.01f);
    step = cv::Vec3f::all(0.05f);
  }
};

void initializeVolume(Mat &volumePoints, const VolumeParams &params = VolumeParams())
{
  Vec3f dimensions = (params.maxBound - params.minBound);
  Mat(dimensions, false) /= Mat(params.step);
  cout << "Volume dims: " << Mat(dimensions) << endl;

  int dims[] = {dimensions[2], dimensions[0], dimensions[1]};
  const int ndims= 3;
  volumePoints.create(ndims, dims, CV_32FC3);

  for (int z = 0; z < dims[0]; ++z)
  {
    for (int x = 0; x < dims[1]; ++x)
    {
      for (int y = 0; y < dims[2]; ++y)
      {
        Vec3f index(x, y, z);
        Vec3f pt = params.minBound + index.mul(params.step);
        volumePoints.at<Vec3f>(z, x, y) = pt;
      }
    }
  }
}

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


    vector<float> groundTruthLabels;
    std::ifstream fin("solution.csv");
    CV_Assert(fin.is_open());

    while(!fin.eof())
    {
        float label;
        fin >> label;
        groundTruthLabels.push_back(label);
    }

    PinholeCamera camera;
    vector<PoseRT> allPoses;
    vector<Mat> allMasks;
    getGroundTruthData(trainedModelsPath, baseFolder, testObjectName, camera, allPoses, allMasks);

    Mat volumePoints;
    initializeVolume(volumePoints);
    Mat volumePoints_Vector(1, volumePoints.total(), CV_32FC3, volumePoints.data);

    Mat A_volumeVariables, b(0, 1, CV_32SC1);
    vector<Mat> allA_pixelVariables;
    cout << "Number of images: " << allMasks.size() << endl;

    int globalPixelVariableIndex = 0;
    for (size_t imageIndex = 0; imageIndex < allMasks.size(); ++imageIndex)
    {
        const Mat &mask = allMasks[imageIndex];
        Mat visualization;
        cvtColor(mask, visualization, CV_GRAY2BGR);
        CV_Assert(mask.type() == CV_8UC1);
        const PoseRT &pose = allPoses[imageIndex];
        Mat A_pixelVariables;

        vector<Point2f> projectedVolume;
        camera.projectPoints(volumePoints_Vector, pose, projectedVolume);

        /*
        CV_Assert(!mask.empty());
        imshow("mask", mask);
        Mat image = mask.clone();
        drawPoints(projectedVolume, image);
        imshow("points", image);
        waitKey(2);
        */

        vector<vector<vector<size_t> > > projectedPointsIndices(mask.rows);
        for (int i = 0; i < mask.rows; ++i)
        {
            projectedPointsIndices[i].resize(mask.cols);
        }
        int pixelVariablesCount = 0;
        for (size_t i = 0; i < projectedVolume.size(); ++i)
        {
            Point pt = projectedVolume[i];
            if (!isPointInside(mask, pt))
            {
                continue;
            }

            if (projectedPointsIndices[pt.y][pt.x].empty())
            {
                ++pixelVariablesCount;
            }

            projectedPointsIndices[pt.y][pt.x].push_back(i);
        }
        cout << "pixelVariables: " << pixelVariablesCount << endl;

        int pixelVariableIndex = 0;
        for (int i = 0; i < mask.rows; ++i)
        {
            for (int j = 0; j < mask.cols; ++j)
            {
                if (projectedPointsIndices[i][j].empty())
                {
                    continue;
                }

                Mat row_pixelVariables(1, pixelVariablesCount, CV_32SC1, Scalar(0));
                row_pixelVariables.at<int>(pixelVariableIndex) = 1;
                if (mask.at<uchar>(i, j) == 0)
                {
                    for (size_t k = 0; k < projectedPointsIndices[i][j].size(); ++k)
                    {
                        Mat row_volumeVariables(1, volumePoints.total(), CV_32SC1, Scalar(0));
                        row_volumeVariables.at<int>(projectedPointsIndices[i][j][k]) = 1;

                        A_pixelVariables.push_back(row_pixelVariables);
                        A_volumeVariables.push_back(row_volumeVariables);
                        b.push_back(1);
                    }
                }
                else
                {
                    Mat row_volumeVariables(1, volumePoints.total(), CV_32SC1, Scalar(0));
                    for (size_t k = 0; k < projectedPointsIndices[i][j].size(); ++k)
                    {
                        row_volumeVariables.at<int>(projectedPointsIndices[i][j][k]) = -1;
                    }
                    A_pixelVariables.push_back(row_pixelVariables);
                    A_volumeVariables.push_back(row_volumeVariables);
                    b.push_back(0);
                }


                float label = groundTruthLabels[globalPixelVariableIndex];
                Scalar color = label > 0.5 ? Scalar(0, 255, 0) : Scalar(255, 0, 255);
                circle(visualization, Point(j, i), 2, color, -1);
//                visualization.at<Vec3b>(i, j) = label > 0.5 ? Vec3b(0, 255, 0) : Vec3b(255, 0, 255);
                ++pixelVariableIndex;
                ++globalPixelVariableIndex;
            }
        }

        imshow("viz", visualization);
        waitKey();

        allA_pixelVariables.push_back(A_pixelVariables);
    }

    vector<Point3f> model;
    for (int i = globalPixelVariableIndex; i < globalPixelVariableIndex + volumePoints.total(); ++i)
    {
        float label = groundTruthLabels[i];
        if (label > 0.00001f)
        {
            model.push_back(volumePoints_Vector.at<Vec3f>(i - globalPixelVariableIndex));
        }
    }
    writePointCloud("model.asc", model);

    int pixelVariablesCount = 0;
    int constraintsCount = 0;
    for (size_t i = 0; i < allA_pixelVariables.size(); ++i)
    {
        pixelVariablesCount += allA_pixelVariables[i].cols;
        constraintsCount += allA_pixelVariables[i].rows;
    }

    Mat A_pixelVariables(constraintsCount, pixelVariablesCount, CV_32SC1, Scalar(0));
    int currentRowIndex = 0;
    int currentColIndex = 0;
    for (size_t i = 0; i < allA_pixelVariables.size(); ++i)
    {
        Mat roi = A_pixelVariables(Range(currentRowIndex, currentRowIndex + allA_pixelVariables[i].rows),
                                   Range(currentColIndex, currentColIndex + allA_pixelVariables[i].cols));
        allA_pixelVariables[i].copyTo(roi);
        currentRowIndex += allA_pixelVariables[i].rows;
        currentColIndex += allA_pixelVariables[i].cols;
    }

    Mat A;
    cout << A_pixelVariables.rows << " x " << A_pixelVariables.cols << endl;
    cout << A_volumeVariables.rows << " x " << A_volumeVariables.cols << endl;
    hconcat(A_pixelVariables, A_volumeVariables, A);
//    cout << "A: " << endl << A << endl << endl;
//    cout << "b:" << endl << b << endl;

    cout << "Dimensionality:"<< endl;

    cout << constraintsCount << " x (" << pixelVariablesCount << " + " << volumePoints.total() << ")" << endl;

    {
        std::ofstream fout("A.csv");
        CV_Assert(fout.is_open());
        fout << format(A, "csv");
        fout.close();
    }
    {
        std::ofstream fout("b.csv");
        CV_Assert(fout.is_open());
        fout << format(b, "csv");
        fout.close();
    }

/*
    FileStorage fs("lpMatrices.xml.gz", FileStorage::WRITE);
    fs << "A" << A;
    fs << "b" << b;
    fs.release();
*/

    return 0;
}

void getGroundTruthData(const string &trainedModelsPath, const string &baseFolder, const string &testObjectName,
                        PinholeCamera &camera, std::vector<PoseRT> &poses, std::vector<Mat> &masks)
{
    const string testFolder = baseFolder + "/" + testObjectName + "/";

    vector<string> objectNames;
    objectNames.push_back(testObjectName);

    TODBaseImporter baseImporter(baseFolder, testFolder);
    vector<EdgeModel> allEdgeModels;
    vector<int> testIndices;
    baseImporter.importAllData(&trainedModelsPath, &objectNames, &camera, 0, &allEdgeModels, &testIndices);
    EdgeModel &edgeModel = allEdgeModels[0];
    poses.clear();
    masks.clear();
    for (size_t _testIdx = 0; _testIdx < testIndices.size(); ++ _testIdx)
    {
        int imageIndex = testIndices[_testIdx];
        PoseRT fiducialPose, offset;
        baseImporter.importGroundTruth(imageIndex, fiducialPose, false, &offset);
        PoseRT model2test = fiducialPose * offset;

        //TODO: wrap it as a single function
        vector<Point2f> projectedPoints;
        camera.projectPoints(edgeModel.points, model2test, projectedPoints);
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

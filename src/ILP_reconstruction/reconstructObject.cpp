/*
 * reconstructObject.cpp
 *
 *  Created on: 1/14/2013
 *      Author: ilysenkov
 */

#include <opencv2/opencv.hpp>
#include <fstream>
#include "edges_pose_refiner/TODBaseImporter.hpp"

#define INITIAL_RUN

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
//    minBound = cv::Vec3f(0.1f, -0.1f, -0.3f);
    minBound = cv::Vec3f(0.19f, -0.00f, -0.15f);
    maxBound = cv::Vec3f(0.30f,  0.14f,  0.0f);
    step = cv::Vec3f::all(0.0025f);
//    step = cv::Vec3f::all(0.005f);


//    step = cv::Vec3f::all(0.01f);
//    step = cv::Vec3f::all(0.01f);
//    step = cv::Vec3f::all(0.03f);
//    step = cv::Vec3f::all(0.05f);
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

struct VolumeVariable
{
    int ilpIndex;
    int volumeIndex;
    float label;
};

struct PixelVariable
{
    int ilpIndex;
    int imageIndex;
    int x, y;
    float label;
};

struct Constraint
{
    std::map<int, float> coefficients;
    int b;
};


void createProblemInstance(const cv::Mat &volumePoints,
                           const PinholeCamera &camera,
                           const std::vector<PoseRT> &allPoses,
                           const std::vector<cv::Mat> &allMasks,
                           std::vector<VolumeVariable> &volumeVariables,
                           std::vector<PixelVariable> &pixelVariables,
                           std::vector<Constraint> &constraints)
{
    volumeVariables.clear();
    pixelVariables.clear();
    constraints.clear();

    Mat volumePoints_Vector(1, volumePoints.total(), CV_32FC3, volumePoints.data);

    for (size_t i = 0; i < volumePoints.total(); ++i)
    {
        VolumeVariable variable;
        variable.volumeIndex = i;
        variable.ilpIndex = i;
        volumeVariables.push_back(variable);
    }

    int pixelVariableILPIndex = volumePoints.total();
    for (size_t imageIndex = 0; imageIndex < allMasks.size(); ++imageIndex)
    {
        const Mat &mask = allMasks[imageIndex];
        CV_Assert(mask.type() == CV_8UC1);
        const PoseRT &pose = allPoses[imageIndex];

        vector<Point2f> projectedVolume;
        camera.projectPoints(volumePoints_Vector, pose, projectedVolume);

        vector<vector<vector<size_t> > > projectedPointsIndices(mask.rows);
        for (int i = 0; i < mask.rows; ++i)
        {
            projectedPointsIndices[i].resize(mask.cols);
        }

        for (size_t i = 0; i < projectedVolume.size(); ++i)
        {
            Point pt = projectedVolume[i];
            if (!isPointInside(mask, pt))
            {
                continue;
            }

            projectedPointsIndices[pt.y][pt.x].push_back(i);
        }

        for (int i = 0; i < mask.rows; ++i)
        {
            for (int j = 0; j < mask.cols; ++j)
            {
                if (projectedPointsIndices[i][j].empty())
                {
                    continue;
                }


                if (mask.at<uchar>(i, j) == 0)
                {
                    for (size_t k = 0; k < projectedPointsIndices[i][j].size(); ++k)
                    {
                        Constraint constraint;
                        constraint.coefficients[pixelVariableILPIndex] = 1;
                        constraint.coefficients[projectedPointsIndices[i][j][k]] = 1;
                        constraint.b = 1;
                        constraints.push_back(constraint);
                    }
                }
                else
                {
                    Constraint constraint;
                    constraint.coefficients[pixelVariableILPIndex] = 1;
                    for (size_t k = 0; k < projectedPointsIndices[i][j].size(); ++k)
                    {
                        constraint.coefficients[projectedPointsIndices[i][j][k]] = -1;
                    }
                    constraint.b = 0;
                    constraints.push_back(constraint);
                }

                PixelVariable variable;
                variable.x = j;
                variable.y = i;
                variable.imageIndex = imageIndex;
                variable.ilpIndex = pixelVariableILPIndex;
                pixelVariables.push_back(variable);

                ++pixelVariableILPIndex;
            }
        }
    }
}

//TODO: implement read/write methods for these classes and call them in these functions
void writeProblemInstance(const std::string &filename,
                          const std::vector<VolumeVariable> &volumeVariables,
                          const std::vector<PixelVariable> &pixelVariables,
                          const std::vector<Constraint> &constraints)
{
    std::ofstream fout(filename.c_str());
    CV_Assert(fout.is_open());

    fout << "Volume variables: " << volumeVariables.size() << "\n";
    for (size_t i = 0; i < volumeVariables.size(); ++i)
    {
        fout << volumeVariables[i].ilpIndex << " " << volumeVariables[i].volumeIndex << "\n";
    }

    fout << "Pixel variables: " << pixelVariables.size() << "\n";
    for (size_t i = 0; i < pixelVariables.size(); ++i)
    {
        const PixelVariable &pv = pixelVariables[i];
        fout << pv.ilpIndex << " " << pv.imageIndex << " " << pv.x << " " << pv.y << "\n";
    }

    fout << "Constraints: " << constraints.size() << "\n";
    for (size_t i = 0; i < constraints.size(); ++i)
    {
        fout << constraints[i].b;
        for (std::map<int, float>::const_iterator it = constraints[i].coefficients.begin(); it != constraints[i].coefficients.end(); ++it)
        {
            fout << " " << it->first << " " << it->second;
        }
        fout << "\n";
    }
}

enum ReadingMode {READ_PIXEL_VARIABLES, READ_VOLUME_VARIABLES, READ_CONSTRAINTS};
void readProblemInstance(const std::string &problemInstanceFilename, const std::string &solutionFilename,
                         std::vector<VolumeVariable> &volumeVariables,
                         std::vector<PixelVariable> &pixelVariables,
                         std::vector<Constraint> &constraints)
{
    const std::string volumeVariablesTag = "Volume variables: ";
    const std::string pixelVariablesTag = "Pixel variables: ";
    const std::string constraintsTag = "Constraints: ";

    volumeVariables.clear();
    pixelVariables.clear();
    constraints.clear();

    {
        std::ifstream fin(problemInstanceFilename.c_str());
        CV_Assert(fin.is_open());


        int volumeVariablesCount, pixelVariablesCount, constraintsCount;
        std::string line;
        bool isReadingConstraints = false;
        ReadingMode mode = READ_VOLUME_VARIABLES;
        int iteration = 0;
        while (std::getline(fin, line))
        {
            std::istringstream input(line);
            if (mode == READ_VOLUME_VARIABLES)
            {
                //TODO: eliminate code duplication
                if (line.find(volumeVariablesTag) != string::npos)
                {
                    int suffixLength = line.length() - static_cast<int>(volumeVariablesTag.length());
                    volumeVariablesCount = atoi(line.substr(volumeVariablesTag.length(), suffixLength).c_str());
                }
                else
                {
                    if (line.find(pixelVariablesTag) != string::npos)
                    {
                        int suffixLength = line.length() - static_cast<int>(pixelVariablesTag.length());
                        pixelVariablesCount = atoi(line.substr(pixelVariablesTag.length(), suffixLength).c_str());
                        mode = READ_PIXEL_VARIABLES;
                        continue;
                    }

                    VolumeVariable var;
                    input >> var.ilpIndex >> var.volumeIndex;
                    volumeVariables.push_back(var);
                }
            }

            if (mode == READ_PIXEL_VARIABLES)
            {
                if (line.find(constraintsTag) != string::npos)
                {
                    int suffixLength = line.length() - static_cast<int>(constraintsTag.length());
                    constraintsCount = atoi(line.substr(constraintsTag.length(), suffixLength).c_str());
                    mode = READ_CONSTRAINTS;
                    continue;
                }

                PixelVariable var;
                input >> var.ilpIndex >> var.imageIndex >> var.x >> var.y;
                pixelVariables.push_back(var);
            }

            if (mode == READ_CONSTRAINTS)
            {
                Constraint constraint;
                input >> constraint.b;
                int index;
                float coefficient;
                while (input >> index >> coefficient)
                {
                    constraint.coefficients[index] = coefficient;
                }
                constraints.push_back(constraint);
            }
        }
        fin.close();

        cout << "counts: " << volumeVariablesCount << " " << pixelVariablesCount << " " << constraintsCount << endl;
        cout << "read counts: " << volumeVariables.size() << " " << pixelVariables.size() << " " << constraints.size() << endl;
    }

    {
        std::ifstream fin(solutionFilename.c_str());
        CV_Assert(fin.is_open());
        for (size_t i = 0; i < volumeVariables.size(); ++i)
        {
            fin >> volumeVariables[i].label;
        }

        for (size_t i = 0; i < pixelVariables.size(); ++i)
        {
            fin >> pixelVariables[i].label;
        }
        fin.close();
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

#ifdef WITH_GROUND_TRUTH
    vector<float> groundTruthLabels;
    std::ifstream fin("solution.csv");
    CV_Assert(fin.is_open());

    while(!fin.eof())
    {
        float label;
        fin >> label;
        groundTruthLabels.push_back(label);
    }

    //TODO: fix
    const int variablesCount = 3597;
    const int allPixelVariablesCount = 3309;
    const int allVolumeVariablesCount = variablesCount - allPixelVariablesCount;
    groundTruthLabels.resize(variablesCount);
    vector<float> volumeLabels;
    vector<Point3f> model;
    for (int i = allPixelVariablesCount; i < variablesCount; ++i)
    {
        cout << groundTruthLabels[i] << endl;
        volumeLabels.push_back(groundTruthLabels[i]);


        /*
        float label = groundTruthLabels[i];
        if (label > 0.00001f)
        {
            model.push_back(volumePoints_Vector.at<Vec3f>(i - globalPixelVariableIndex));
        }
        */
    }
//    writePointCloud("model.asc", model);
#endif


    PinholeCamera camera;
    vector<PoseRT> allPoses;
    vector<Mat> allMasks;
    getGroundTruthData(trainedModelsPath, baseFolder, testObjectName, camera, allPoses, allMasks);

    Mat volumePoints;
    initializeVolume(volumePoints);


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


    std::vector<VolumeVariable> volumeVariables;
    std::vector<PixelVariable> pixelVariables;
    std::vector<Constraint> constraints;

#ifdef INITIAL_RUN
    createProblemInstance(volumePoints, camera, allPoses, allMasks,
                          volumeVariables, pixelVariables, constraints);

    writeProblemInstance("ilpProblem.txt", volumeVariables, pixelVariables, constraints);
#else
    readProblemInstance("ilpProblem.txt", "solution.csv", volumeVariables, pixelVariables, constraints);

    Mat volumePoints_Vector(1, volumePoints.total(), CV_32FC3, volumePoints.data);
    int currentPixelVariableIndex = 0;

    vector<Point3f> model;
    for (size_t i = 0; i < volumePoints.total(); ++i)
    {
        //TODO: move up
        if (volumeVariables[i].label > 0.01f && volumeVariables[i].label < 0.99f)
        {
            cout << "not binary: " <<  volumeVariables[i].label << endl;
        }

        if (volumeVariables[i].label > 0.001f)
        {
            model.push_back(volumePoints_Vector.at<Point3f>(i));
        }
    }
    writePointCloud("model.asc", model);

    for (size_t imageIndex = 0; imageIndex < allMasks.size(); ++imageIndex)
    {
        const Mat &mask = allMasks[imageIndex];
        Mat visualization;
        cvtColor(mask, visualization, CV_GRAY2BGR);
        CV_Assert(mask.type() == CV_8UC1);
        const PoseRT &pose = allPoses[imageIndex];


        vector<Point2f> projectedVolume;
        camera.projectPoints(volumePoints_Vector, pose, projectedVolume);
        for (size_t i = 0; i < projectedVolume.size(); ++i)
        {
            if (volumeVariables[i].label > 0.0001f)
            {
                //TODO: move up
                Scalar color = volumeVariables[i].label > 0.0001f ? Scalar(0, 255, 0) : Scalar(255, 0, 255);
                circle(visualization, projectedVolume[i], 0, color, -1);
            }
        }

        imshow("volume", visualization);

        Mat pixelVariablesVisualization;
        cvtColor(mask, pixelVariablesVisualization, CV_GRAY2BGR);
        while (currentPixelVariableIndex < pixelVariables.size() && pixelVariables[currentPixelVariableIndex].imageIndex == imageIndex)
        {
            Scalar color = pixelVariables[currentPixelVariableIndex].label > 0.0001f ? Scalar(0, 255, 0) : Scalar(255, 0, 255);
            Point pt(pixelVariables[currentPixelVariableIndex].x, pixelVariables[currentPixelVariableIndex].y);
            circle(pixelVariablesVisualization, pt, 0, color, -1);
            ++currentPixelVariableIndex;
        }

        imshow("pixels", pixelVariablesVisualization);

        waitKey();
    }
#endif

#if 0
    Mat A_volumeVariables, b(0, 1, CV_32SC1);
    vector<Mat> allA_pixelVariables;
    cout << "Number of images: " << allMasks.size() << endl;
    int globalPixelVariableIndex = 0;
    for (size_t imageIndex = 0; imageIndex < allMasks.size(); ++imageIndex)
    {
        const Mat &mask = allMasks[imageIndex];
#ifdef WITH_GROUND_TRUTH
        Mat visualization, volumeVisualization;
        cvtColor(mask, visualization, CV_GRAY2BGR);
        cvtColor(mask, volumeVisualization, CV_GRAY2BGR);
#endif
        CV_Assert(mask.type() == CV_8UC1);
        const PoseRT &pose = allPoses[imageIndex];
        Mat A_pixelVariables;

        vector<Point2f> projectedVolume;
        camera.projectPoints(volumePoints_Vector, pose, projectedVolume);

#ifdef WITH_GROUND_TRUTH
        for (size_t i = 0; i < volumeLabels.size(); ++i)
        {
            //TODO: move up
            if (volumeLabels[i] > 0.001f)
            {
                circle(volumeVisualization, projectedVolume[i], 2, Scalar(255, 0, 255), -1);
            }
        }

        imshow("volumeViz", volumeVisualization);
#endif

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

#ifdef WITH_GROUND_TRUTH
                float label = groundTruthLabels[globalPixelVariableIndex];
                Scalar color = label > 0.5 ? Scalar(0, 255, 0) : Scalar(255, 0, 255);
                circle(visualization, Point(j, i), 2, color, -1);
//                visualization.at<Vec3b>(i, j) = label > 0.5 ? Vec3b(0, 255, 0) : Vec3b(255, 0, 255);
                ++globalPixelVariableIndex;
#endif
                ++pixelVariableIndex;
            }
        }

#ifdef WITH_GROUND_TRUTH
        imshow("viz", visualization);
//        waitKey();
#endif

        allA_pixelVariables.push_back(A_pixelVariables);
    }

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
        fout << format(b.t(), "csv");
        fout.close();
    }

/*
    FileStorage fs("lpMatrices.xml.gz", FileStorage::WRITE);
    fs << "A" << A;
    fs << "b" << b;
    fs.release();
*/
#endif

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

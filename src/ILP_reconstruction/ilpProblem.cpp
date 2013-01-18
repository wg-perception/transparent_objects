/*
 * ilpProblem.cpp
 *
 *  Created on: 1/18/2013
 *      Author: ilysenkov
 */

#include <opencv2/opencv.hpp>
#include <fstream>

#include "ilpProblem.hpp"
#include "edges_pose_refiner/utils.hpp"

using namespace cv;
using std::cout;
using std::endl;

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

ILPProblem::ILPProblem(const VolumeParams &_volumeParams, const PinholeCamera &_camera)
{
    volumeParams = _volumeParams;
    camera = _camera;
}

ILPProblem::ILPProblem(const VolumeParams &_volumeParams, const PinholeCamera &_camera,
                       const std::vector<PoseRT> &allPoses, const std::vector<cv::Mat> &allMasks)
{
    volumeParams = _volumeParams;
    camera = _camera;
    isSolved = false;

    initializeVolume(volumePoints, volumeParams);
    volumePoints_Vector = Mat(1, volumePoints.total(), CV_32FC3, volumePoints.data);

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

void ILPProblem::write(const std::string &filename) const
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
void ILPProblem::read(const std::string &problemInstanceFilename, const std::string &solutionFilename)
{
    isSolved = true;
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

void ILPProblem::getModel(std::vector<cv::Point3f> &model) const
{
    model.clear();
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
}

void ILPProblem::visualize(const std::vector<PoseRT> &allPoses, const std::vector<cv::Mat> &allMasks) const
{
    int currentPixelVariableIndex = 0;
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
}

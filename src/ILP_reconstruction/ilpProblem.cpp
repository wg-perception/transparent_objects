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

  int dims[] = {cvRound(dimensions[2]), cvRound(dimensions[0]), cvRound(dimensions[1])};
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

    initializeVolume(volumePoints, volumeParams);
    volumePoints_Vector = Mat(1, volumePoints.total(), CV_32FC3, volumePoints.data);
}

#if 1
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
        cout << "image: " << imageIndex << endl;

        const Mat &mask = allMasks[imageIndex];
        CV_Assert(mask.type() == CV_8UC1);
        const PoseRT &pose = allPoses[imageIndex];

        vector<vector<Point2f> > convexHulls;
        for (size_t i = 0; i < volumePoints.total(); ++i)
        {
            Point3f pt = volumePoints_Vector.at<Point3f>(i);

            vector<Point3f> corners;

            float dx = volumeParams.step[0] / 2;
            float dy = volumeParams.step[1] / 2;
            float dz = volumeParams.step[2] / 2;
            //TODO: use a loop
            corners.push_back(pt + Point3f(-dx, -dy, -dz));
            corners.push_back(pt + Point3f(-dx, -dy, +dz));
            corners.push_back(pt + Point3f(-dx, +dy, -dz));
            corners.push_back(pt + Point3f(-dx, +dy, +dz));
            corners.push_back(pt + Point3f(+dx, -dy, -dz));
            corners.push_back(pt + Point3f(+dx, -dy, +dz));
            corners.push_back(pt + Point3f(+dx, +dy, -dz));
            corners.push_back(pt + Point3f(+dx, +dy, +dz));

            vector<Point2f> projectedCorners;
            camera.projectPoints(corners, pose, projectedCorners);

            vector<Point2f> hull;
            convexHull(projectedCorners, hull);
            convexHulls.push_back(hull);
        }

        cout << "convex hulls are computed" << endl;

        //TODO: move up
        const int pixelStep = 4;
        for (int i = 0; i < mask.rows; i += pixelStep)
        {
            for (int j = 0; j < mask.cols; j += pixelStep)
            {
                vector<int> intersectedVoxelsIndices;
                for (size_t k = 0; k < convexHulls.size(); ++k)
                {
                    if (pointPolygonTest(convexHulls[k], Point(j, i), false) >= 0)
                    {
                        intersectedVoxelsIndices.push_back(k);
                    }
                }

                if (intersectedVoxelsIndices.empty())
                {
                    continue;
                }


                if (mask.at<uchar>(i, j) == 0)
                {
                    for (int &voxelIndex : intersectedVoxelsIndices)
                    {
                        Constraint constraint;
                        constraint.coefficients[pixelVariableILPIndex] = 1;
                        constraint.coefficients[voxelIndex] = 1;
                        constraint.b = 1;
                        constraints.push_back(constraint);
                    }
                }
                else
                {
                    Constraint constraint;
                    constraint.coefficients[pixelVariableILPIndex] = 1;
                    for (int &voxelIndex : intersectedVoxelsIndices)
                    {
                        constraint.coefficients[voxelIndex] = -1;
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

#else

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
#endif

void ILPProblem::write(const std::string &filename) const
{
    std::ofstream fout(filename.c_str());
    CV_Assert(fout.is_open());

    fout << "Volume params:\n";
    fout << volumeParams.minBound[0] << " " << volumeParams.minBound[1] << " " << volumeParams.minBound[2] << "\n";
    fout << volumeParams.maxBound[0] << " " << volumeParams.maxBound[1] << " " << volumeParams.maxBound[2] << "\n";
    fout << volumeParams.step[0] << " " << volumeParams.step[1] << " " << volumeParams.step[2] << "\n";

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

    fout.close();
}

void ILPProblem::writeLP(const std::string &filename) const
{
    std::ofstream fout(filename.c_str());
    CV_Assert(fout.is_open());
    fout << "Maximize\n";
    fout << " obj: ";
    for (size_t i = 0; i < pixelVariables.size(); ++i)
    {
        fout << (i > 0 ? " + " : "") << "x" << pixelVariables[i].ilpIndex;
    }

    fout << "\nSubject To\n";
    for (size_t i = 0; i < constraints.size(); ++i)
    {
        fout << " c" << i << ":";
        for (const auto &coeff : constraints[i].coefficients)
        {
            fout << " " << (coeff.second > 0 ? " + " : "") << coeff.second << " x" << coeff.first;
        }
        fout << " <= " << constraints[i].b << "\n";
    }

    fout << "Bounds\n";
    for (const auto &variable : volumeVariables)
    {
        fout << " 0 <= x" << variable.ilpIndex << " <= 1\n";
    }
    for (const auto &variable : pixelVariables)
    {
        fout << " 0 <= x" << variable.ilpIndex << " <= 1\n";
    }

    fout << "Binary\n";
    for (const auto &variable : volumeVariables)
    {
        fout << " x" << variable.ilpIndex << "\n";
    }
    for (const auto &variable : pixelVariables)
    {
        fout << " x" << variable.ilpIndex << "\n";
    }

    fout.close();
}

void ILPProblem::writeMPS(const std::string &filename) const
{
    CV_Assert(false);

    std::ofstream fout(filename.c_str());
    CV_Assert(fout.is_open());
    fout << "NAME\tTransparent_objects_reconstruction_" << volumeParams.step[0] << "\n";
    fout << "ROWS\n";
    fout << " N  COST\n";

    for (size_t i = 0; i < constraints.size(); ++i)
    {
        fout << " L  R" << i << "\n";
    }

    fout << "COLUMNS\n";

    fout.close();
}

enum ReadingMode {READ_VOLUME_PARAMS, READ_PIXEL_VARIABLES, READ_VOLUME_VARIABLES, READ_CONSTRAINTS};
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
        ReadingMode mode = READ_VOLUME_PARAMS;
        while (std::getline(fin, line))
        {
            std::istringstream input(line);
            if (mode == READ_VOLUME_PARAMS)
            {
                if (line.find(volumeVariablesTag) != string::npos)
                {
                    int suffixLength = line.length() - static_cast<int>(volumeVariablesTag.length());
                    volumeVariablesCount = atoi(line.substr(volumeVariablesTag.length(), suffixLength).c_str());
                    mode = READ_VOLUME_VARIABLES;
                    continue;
                }
            }

            if (mode == READ_VOLUME_VARIABLES)
            {
                //TODO: eliminate code duplication
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
    CV_Assert(isSolved);

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

void ILPProblem::visualizeVolumeVariables() const
{
    CV_Assert(isSolved);

    Mat labels(volumePoints.dims, volumePoints.size.p, CV_32FC1);
    Mat labels_Vector(1, labels.total(), CV_32FC3, labels.data);
    CV_Assert(labels.total() == volumeVariables.size());
    for (size_t i = 0; i < volumeVariables.size(); ++i)
    {
        labels_Vector.at<float>(i) = volumeVariables[i].label;
    }

    Mat visualization = labels;
    if (!groundTruthModel.empty())
    {
        supperimposeGroundTruth(labels, visualization);
    }

    imshow3d("labels", visualization);
    waitKey();
}

void ILPProblem::setGroundTruthModel(const std::vector<Point3f> &_groundTruthModel)
{
    groundTruthModel = _groundTruthModel;
}

void ILPProblem::supperimposeGroundTruth(const cv::Mat &image3d, cv::Mat &supperimposedImage) const
{
  //TODO: move up
  const Vec3b groundTruthColor(0, 255, 0);
  const float groundTruthWeight = 0.2f;

  CV_Assert(image3d.channels() == 1);
  Mat image3d_uchar = image3d;
  if (image3d.type() == CV_32FC1)
  {
    image3d.convertTo(image3d_uchar, CV_8UC1, 255);
  }
  CV_Assert(image3d_uchar.type() == CV_8UC1);

  cvtColor3d(image3d_uchar, supperimposedImage, CV_GRAY2BGR);

  Vec3f inverseStep;
  for (int i = 0; i < volumeParams.step.channels; ++i)
  {
    inverseStep[i] = 1.0 / volumeParams.step[i];
  }

  for (size_t i = 0; i < groundTruthModel.size(); ++i)
  {
    Vec3f pt(groundTruthModel[i]);
    Vec3i raw_indices = (pt - volumeParams.minBound).mul(inverseStep);
    Vec3i indices(raw_indices[2], raw_indices[0], raw_indices[1]);

    bool isInside = true;
    for (int j = 0; j < indices.channels; ++j)
    {
      isInside = isInside && indices[j] >= 0 && indices[j] < supperimposedImage.size.p[j];
    }

    if (isInside)
    {
      supperimposedImage.at<Vec3b>(indices) *= 1.0 - groundTruthWeight;
      supperimposedImage.at<Vec3b>(indices) += groundTruthWeight * groundTruthColor;
    }
  }
}

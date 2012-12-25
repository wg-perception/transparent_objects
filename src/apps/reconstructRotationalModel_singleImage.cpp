/*
 * reconstructRotationalModel_singleImage.cpp
 *
 *  Created on: 12/18/2012
 *      Author: ilysenkov
 */

#include <opencv2/opencv.hpp>
#include "edges_pose_refiner/tableSegmentation.hpp"
#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/glassSegmentator.hpp"
#include <opencv2/rgbd/rgbd.hpp>

//#define VISUALIZE_MODEL_CONSTRUCTION_3D
#define SAVE_USER_INPUT
//#define LOAD_USER_INPUT

using namespace cv;
using std::cout;
using std::endl;

void collectUserInput(const cv::Mat &image,
                      std::vector<cv::Point> &topEllipseContour, std::vector<cv::Point> &bottomEllipseContour)
{
#ifndef LOAD_USER_INPUT
    markContourByUser(image, topEllipseContour, "mark the top ellipse");
    markContourByUser(image, bottomEllipseContour, "mark the bottom ellipse");
#else
    topEllipseContour = getFromCache("userInput_top");
    bottomEllipseContour = getFromCache("userInput_bottom");
#endif

#ifdef SAVE_USER_INPUT
    saveToCache("userInput_top", Mat(topEllipseContour));
    saveToCache("userInput_bottom", Mat(bottomEllipseContour));
#endif
}

int main(int argc, char *argv[])
{
    CV_Assert(argc == 3);
    const string baseFolder = argv[1];
    const string objectName = argv[2];
    const string testFolder = baseFolder + "/" + objectName + "/";

    TODBaseImporter baseImporter(baseFolder, testFolder);
    Mat image, depth;
    baseImporter.importBGRImage(0, image);
    baseImporter.importDepth(0, depth);
    PinholeCamera camera;
    baseImporter.importCamera(camera);

    Mat glassMask;
#ifdef USE_AUTOMATIC_SEGMENTATION
    Mat registrationMask;
    baseImporter.importRegistrationMask(registrationMask);
    GlassSegmentator glassSegmentator;
    int numberOfComponents;
    glassSegmentator.segment(image, depth, registrationMask, numberOfComponents, glassMask);
#else

#ifndef LOAD_USER_INPUT
    segmentGlassManually(image, glassMask);
#else
    glassMask = getFromCache("userMask");
#endif
#ifdef SAVE_USER_INPUT
    saveToCache("userMask", glassMask);
#endif
    Mat refinedGlassMask;
    GlassSegmentatorParams params;
    params.grabCutDilationsIterations = 1;
    params.grabCutErosionsIterations = 1;
    refineSegmentationByGrabCut(image, glassMask, refinedGlassMask, params);
    glassMask = refinedGlassMask;
#endif

    showSegmentation(image, glassMask);
    waitKey();

#ifdef VISUALIZE_MODEL_CONSTRUCTION_3D
    Mat points3d;
    depthTo3d(depth, camera.cameraMatrix, points3d);
    vector<vector<Point3f> > visualizationPoints;
    visualizationPoints.push_back(points3d.reshape(3, 1));
#endif

    vector<Point> topEllipsePoints, bottomEllipsePoints;
    collectUserInput(image, topEllipsePoints, bottomEllipsePoints);
    RotatedRect topEllipse = fitEllipse(topEllipsePoints);
    RotatedRect bottomEllipse = fitEllipse(bottomEllipsePoints);

    Point2f centerTop = topEllipse.center;
    Point2f centerBottom = bottomEllipse.center;
    //ideal glass
    //Point2f centerTop(331.733, 141.999), centerBottom(334.201, 191.253);

    cout << "centers: " << centerTop << " " << centerBottom << endl;

    Mat drawImage = image.clone();
    circle(drawImage, centerTop, 1, Scalar(255, 0, 255), -1);
    circle(drawImage, centerBottom, 1, Scalar(255, 0, 255), -1);
    ellipse(drawImage, topEllipse, Scalar(0, 255, 0));
    ellipse(drawImage, bottomEllipse, Scalar(0, 255, 0));
    imshow("draw", drawImage);
    waitKey();

    Vec4f tablePlane;
    vector<Point> tableHull;
    computeTableOrientationByRGBD(depth, camera, tablePlane, &tableHull);
//    computeTableOrientationByFiducials(camera, image, tablePlane);
    Point3f tableNormal(tablePlane[0], tablePlane[1], tablePlane[2]);
    tableNormal *= 1.0 / norm(tableNormal);

    polylines(drawImage, tableHull, true, Scalar(255, 0, 0), 2);
    imshow("table", drawImage);
    waitKey();


    Point3f centerBottom3D = camera.reprojectPointsOnTable(centerBottom, tablePlane);
    Point3f centerTopRay = camera.reprojectPoints(centerTop);
    centerTopRay *= 1.0 / norm(centerTopRay);
    cout << "center bottom 3D: " << centerBottom3D << endl;
    cout << "center top ray: " << centerTopRay << endl;

    Point3f basis_x(tableNormal.y, -tableNormal.x, 0.0f);
    Point3f basis_y = basis_x.cross(tableNormal);
    basis_x *= 1.0 / norm(basis_x);
    basis_y *= 1.0 / norm(basis_y);

    const float eps = 1e-4;
    CV_Assert(fabs(basis_x.dot(basis_y)) < eps);
    CV_Assert(fabs(basis_x.dot(tableNormal)) < eps);
    CV_Assert(fabs(basis_y.dot(tableNormal)) < eps);

    //TODO: use a smarter approach
    //TODO: move up
    const float maxBottomOffset = 0.02f;
    const float minHeight = 0.05f;
    const float maxHeight = 0.4f;
    const float step = 0.0005f;

    float minError = std::numeric_limits<float>::max();
    Point3f bottomBest3D, topBest3D;
    bool useReprojectionError = true;

    for (float x = -maxBottomOffset; x < maxBottomOffset; x += step)
    {
        for (float y = -maxBottomOffset; y < maxBottomOffset; y += step)
        {
            Point3f ptBottom = centerBottom3D + x * basis_x + y * basis_y;
            Point2f ptBottomProjected = camera.projectPoints(ptBottom, PoseRT());

            for (float z = minHeight; z < maxHeight; z += step)
            {
                Point3f ptTop = ptBottom + z * tableNormal;
                Point2f ptTopProjected = camera.projectPoints(ptTop, PoseRT());

                float error;
                if (useReprojectionError)
                {
                    error = norm(ptBottomProjected - centerBottom) + norm(ptTopProjected - centerTop);
                }
                else
                {
                    error = norm(ptBottom - centerBottom3D) + norm(ptTop.cross(centerTopRay));
                }

                if (error < minError)
                {
                    minError = error;
                    bottomBest3D = ptBottom;
                    topBest3D = ptTop;
                }
            }
        }
    }

    cout << "original bottom: " << centerBottom3D << endl;
    centerBottom3D = bottomBest3D;
    Point3f centerTop3D = topBest3D;
    cout << "new 3D points: " << centerBottom3D << " " << centerTop3D << endl;;
    cout << "error: " << minError << endl;

#ifdef VISUALIZE_MODEL_CONSTRUCTION_3D
    visualizationPoints.push_back(vector<Point3f>(1, centerBottom3D));
    visualizationPoints.push_back(vector<Point3f>(1, centerTop3D));
#endif

    Point2f projectedCenterTop3D = camera.projectPoints(centerTop3D, PoseRT());
    Point2f projectedCenterBottom3D = camera.projectPoints(centerBottom3D, PoseRT());
    circle(drawImage, projectedCenterTop3D, 1, Scalar(0, 255, 0), -1);
    circle(drawImage, projectedCenterBottom3D, 1, Scalar(0, 255, 0), -1);
    imshow("corrected centers", drawImage);
    waitKey();

    float objectHeight = norm(centerTop3D - centerBottom3D);
    cout << "objectHeight: " << objectHeight << endl;

    //TODO: move up
    const float heightStep = 0.001f;
    const float radiusStep = 0.0005f;
    const float pointsCount = 100;
    const float phiStep = 2.0 * CV_PI / pointsCount;

    vector<Point3f> modelPoints;
    vector<Point3f> circlesModelPoints;
#ifdef VISUALIZE_MODEL_CONSTRUCTION_3D
    vector<Point3f> allBasePoints;
#endif
    //TODO: use a smarter approach
    for (float h = 0.0f; h <= objectHeight; h += heightStep)
    {
        Point3f basePoint = centerBottom3D + h * tableNormal;
#ifdef VISUALIZE_MODEL_CONSTRUCTION_3D
        allBasePoints.push_back(basePoint);
#endif
        float r = radiusStep;
        while (true)
        {
            vector<Point3f> circlePoints;
            for (int i = 0; i < pointsCount; ++i)
            {
                float phi = i * phiStep;
                float x = r * cos(phi);
                float y = r * sin(phi);

                Point3f pt = basePoint + x * basis_x + y * basis_y;
                circlePoints.push_back(pt);
            }

            vector<Point2f> projectedCirclePoints;
            camera.projectPoints(circlePoints, PoseRT(), projectedCirclePoints);
            bool isInside = true;
            for (size_t i = 0; i < projectedCirclePoints.size(); ++i)
            {
                Point pt = projectedCirclePoints[i];
                CV_Assert(isPointInside(glassMask, pt));

                if (!glassMask.at<uchar>(pt))
                {
                    isInside = false;
#ifdef VISUALIZE_MODEL_CONSTRUCTION_3D
                    visualizationPoints.push_back(circlePoints);
#endif
                    break;
                }
            }

            if (!isInside)
            {
                r -= radiusStep / 2.0;
                break;
            }

            r += radiusStep;
        }

        //TODO: remove code duplication
        for (int i = 0; i < pointsCount; ++i)
        {
            float phi = i * phiStep;
            float x = r * cos(phi);
            float y = r * sin(phi);

            Point3f pt(x, y, h);
            modelPoints.push_back(pt);
            circlesModelPoints.push_back(basePoint + x * basis_x + y * basis_y);
        }
    }
    writePointCloud("model_singleImage.asc", modelPoints);

    Mat projectedModel = drawEdgels(image, circlesModelPoints, PoseRT(), camera);
    imshow("final model", projectedModel);
    waitKey();

#ifdef VISUALIZE_MODEL_CONSTRUCTION_3D
    Mat projectedNormal = drawEdgels(drawImage, allBasePoints, PoseRT(), camera);
    imshow("projectedNormal", projectedNormal);
    waitKey();

    visualizationPoints.push_back(allBasePoints);
    publishPoints(visualizationPoints);
//    publishPoints(modelPoints);
#endif

    return 0;
}


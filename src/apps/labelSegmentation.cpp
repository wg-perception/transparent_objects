#include <opencv2/opencv.hpp>

#include "edges_pose_refiner/TODBaseImporter.hpp"
#include "edges_pose_refiner/glassSegmentator.hpp"
#include "edges_pose_refiner/tableSegmentation.hpp"

//#include <opencv2/rgbd/rgbd.hpp>
#include <iomanip>

using namespace cv;
using std::cout;
using std::endl;

Point offsetPt;

void onMouse(int event, int x, int y, int, void *)
{
  if (event == CV_EVENT_LBUTTONDOWN)
  {
      offsetPt = Point(x, y);
      cout << x << " " << y << endl;
  }
}

const string mainWindowName = "dataset image";

//TODO: don't use global variables. Pass it as the user data in callbacks
string testFolder;
int imageIndex = 0;
vector<int> testIndices;
TODBaseImporter dataImporter;
PinholeCamera camera;

void onTrackbar(int, void*)
{
    const char segmentationKey = 's';
    const char saveKey = 's';
    const char offsetKey = 'o';
    const string userSegmentationWindowName = "user segmentation";
    const string refinedSegmentationWindowName = "refined segmentation";

    Mat bgrImage;
    dataImporter.importBGRImage(testIndices[imageIndex], bgrImage);
    imshow(mainWindowName, bgrImage);
/*
    Mat depth;
    dataImporter.importDepth(testIndices[imageIndex], depth);
    imshow(mainWindowName + "_depth", depth);
*/

    int key = waitKey();
    if (key == segmentationKey)
    {
        Mat userMask;
        segmentGlassManually(bgrImage, userMask);
        showSegmentation(bgrImage, userMask, userSegmentationWindowName);

        GlassSegmentatorParams glassSegmentatorParams;
        glassSegmentatorParams.grabCutErosionsIterations = 2;
        glassSegmentatorParams.grabCutDilationsIterations = 2;
        Mat refinedMask;
        refineSegmentationByGrabCut(bgrImage, userMask, refinedMask, glassSegmentatorParams);
        showSegmentation(bgrImage, refinedMask, refinedSegmentationWindowName);

        key = waitKey();
        if (key = saveKey)
        {
            std::stringstream maskFilename;
            maskFilename << testFolder << "/image_" << std::setfill('0') << std::setw(5) << testIndices[imageIndex] << ".png.user_mask.png";
            imwrite(maskFilename.str(), refinedMask);
        }

        destroyWindow(userSegmentationWindowName);
        destroyWindow(refinedSegmentationWindowName);
    }

    //TODO: move it to a separate app
    if (key == offsetKey)
    {
        const string offsetWindowName = "label offset";
        namedWindow(offsetWindowName, WINDOW_NORMAL);
        imshow(offsetWindowName, bgrImage);
        setMouseCallback(offsetWindowName, onMouse);
        waitKey();
        destroyWindow(offsetWindowName);

        Vec4f tablePlane;
        int numberOfPatternsFound = computeTableOrientationByFiducials(camera, bgrImage, tablePlane);
        CV_Assert(numberOfPatternsFound == 2);

        Mat blackBlobs, whiteBlobs;
        detectFiducial(bgrImage, blackBlobs, whiteBlobs);
        CV_Assert(!blackBlobs.empty() && !whiteBlobs.empty());

        vector<Point2f> imagePoints;
        //TODO: move up
        const int originIndex = 0;
        const int xIndex = 3;
        const int yIndex = 40;

        imagePoints.push_back(whiteBlobs.at<Point2f>(originIndex));
        imagePoints.push_back(blackBlobs.at<Point2f>(xIndex));
        imagePoints.push_back(whiteBlobs.at<Point2f>(yIndex));
        imagePoints.push_back(offsetPt);


        vector<Point3f> reprojectedPoints;
        camera.reprojectPointsOnTable(imagePoints, tablePlane, reprojectedPoints);
        Point3f xAxis = reprojectedPoints[1] - reprojectedPoints[0];
        Point3f yAxis = reprojectedPoints[2] - reprojectedPoints[0];
        Point3f offsetVector = reprojectedPoints[3] - reprojectedPoints[0];

        float x = offsetVector.dot(xAxis) / norm(xAxis);
        float y = offsetVector.dot(yAxis) / norm(yAxis);

        cout << "Offset: " << x << " " << y << endl;

/*
        Mat drawImage = bgrImage.clone();
        circle(drawImage, imagePoints[0], 2, Scalar(255, 0, 255), -1);
        circle(drawImage, imagePoints[2], 2, Scalar(255, 0, 255), -1);
        circle(drawImage, blackBlobs.at<Point2f>(0), 2, Scalar(0, 255, 0), -1);
        imshow("circle", drawImage);
        Mat points3d;
        depthTo3d(depth, camera.cameraMatrix, points3d);
        cout << points3d.at<Point3f>(imagePoints[0]) << endl;
        cout << points3d.at<Point3f>(imagePoints[2]) << endl;
        cout << "distance: " << norm(points3d.at<Point3f>(imagePoints[0]) - points3d.at<Point3f>(imagePoints[2])) << endl;
        cout << "distance between patterns: " << norm(points3d.at<Point3f>(imagePoints[0]) - points3d.at<Point3f>(blackBlobs.at<Point2f>(0))) << endl;
        //cout << "distance: " << norm(reprojectedPoints[0] - reprojectedPoints[1]) << endl;

        vector<Point3f> cvCloud = points3d.reshape(3, points3d.total());
        writePointCloud("test.asc", cvCloud);

        waitKey();
*/
    }
}

int main(int argc, char *argv[])
{
    CV_Assert(argc == 3);
    const string baseFolder = argv[1];
    const string objectName = argv[2];
    testFolder = baseFolder + "/" + objectName + "/";
    dataImporter = TODBaseImporter(baseFolder, testFolder);

    Mat registrationMask;
    dataImporter.importAllData(0, 0, &camera, &registrationMask, 0, &testIndices);
    namedWindow(mainWindowName, WINDOW_NORMAL);
    int lastIndex = static_cast<int>(testIndices.size()) - 1;
    createTrackbar("image index", mainWindowName, &imageIndex, lastIndex, onTrackbar);
    onTrackbar(0, 0);

    return 0;
}

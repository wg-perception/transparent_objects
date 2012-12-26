/*
 * writeVideo.cpp
 *
 *  Created on: 12/16/2012
 *      Author: ilysenkov
 */

#include <opencv2/opencv.hpp>
#include <iomanip>
#include <fstream>

using namespace cv;
using std::cout;
using std::endl;

void readQualities(const std::string &path, int endIndex, std::vector<float> &allQualities)
{
    std::string qualitiesFilename = path + "/qualities.txt";
    std::ifstream log(qualitiesFilename.c_str());
    CV_Assert(log.is_open());
    allQualities.resize(endIndex, std::numeric_limits<float>::max());
    while (!log.eof())
    {
        int testIndex;
        float quality;
        log >> testIndex >> quality;
        cout << testIndex << " " << quality << endl;
        if (testIndex < endIndex)
        {
            allQualities[testIndex] = quality;
        }
    }
}

int main(int argc, char *argv[])
{
    CV_Assert(argc == 6);

    const string path = argv[1];
    const string objectName = argv[2];
    const int startIndex = atoi(argv[3]);
    const int endIndex = atoi(argv[4]);
    const float maxQuality = atof(argv[5]); //1.2f

    const int fps = 10;

    std::vector<float> allQualities;
    readQualities(path, endIndex, allQualities);

    VideoWriter writer;
    for (int i = startIndex; i < endIndex; ++i)
    {
        cout << i << " " << allQualities[i] << endl;
        std::stringstream index;
        index << std::setw(5) << std::setfill('0') << i;

        Mat bgrImage = imread(path + "/image_" + index.str() + ".png");
        CV_Assert(!bgrImage.empty());
        Mat detection = imread(path + "/image_" + index.str() + "_detection.png");
        CV_Assert(!detection.empty());

        if (allQualities[i] > maxQuality)
        {
            detection = bgrImage;
        }

        Mat concatImage;
        hconcat(bgrImage, detection, concatImage);

        if (i == startIndex)
        {
            writer.open(objectName + ".avi", CV_FOURCC('M','J','P','G'), fps, concatImage.size());
            CV_Assert(writer.isOpened());
        }

        writer << concatImage;
    }
    writer.release();

    return 0;
}


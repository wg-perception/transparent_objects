#!/usr/bin/python
import sys
import glob
import re
import cv2
from subprocess import call
import numpy as np


if __name__ == '__main__':
    assert len(sys.argv) == 5, sys.argv[0] + ' <testImagesFilename> <classifierFilename> <segmentationFilesList> <propagationScaligsList>'

    segmentImage = '/home/ilysenkov/itseezMachine/home/ilysenkov/ecto/server_build/bin/enhancedGlassSegmenter'
    tmpSegmentationFilename = 'segmentation.png'
    groundTruthPrefix = 'glassMask'

    testImagesFilename = sys.argv[1]
    classifierFilename = sys.argv[2]
    segmentationFilesList = sys.argv[3]
    propagationScalingsList = sys.argv[4]

    segmentationFilenames = open(segmentationFilesList, 'r').read().splitlines()
    testFilenames = open(testImagesFilename, 'r').read().splitlines()
    
    allTrueGlassArea = [0] * len(segmentationFilenames)
    allPredictedGlassArea = [0] * len(segmentationFilenames)
    allValidPredictedGlassArea = [0] * len(segmentationFilenames)

    for imageFilename in testFilenames[:2]:
        print imageFilename

        match = re.search('_(0*([1-9][0-9]*))\.', imageFilename)
        if (match == None):
            match = re.search('_(0000(0))\.', imageFilename)
        assert match != None, 'Cannot parse an image index'

        fullImageIndex = match.group(1)
        imageIndex = match.group(2)

        groundTruthFilename = imageFilename.replace('/image_', '/' + groundTruthPrefix + '_')
        testFolder = imageFilename.replace('/image_' + fullImageIndex + '.png', '/')

        call([segmentImage, testFolder, fullImageIndex, classifierFilename, segmentationFilesList, propagationScalingsList])

        groundTruth = cv2.imread(groundTruthFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        for (index, filename) in enumerate(segmentationFilenames):
            segmentation = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)

            glassLabel = 255
            trueGlassArea = len(np.nonzero(groundTruth == glassLabel)[0])
            predictedGlassArea = len(np.nonzero(segmentation == glassLabel)[0])
            validPredictedGlassArea = len(np.nonzero(np.logical_and(groundTruth == glassLabel, segmentation == groundTruth))[0])
            allTrueGlassArea[index] += trueGlassArea
            allPredictedGlassArea[index] += predictedGlassArea
            allValidPredictedGlassArea[index] += validPredictedGlassArea

            print 'precision =', float(validPredictedGlassArea) / predictedGlassArea
            print 'recall =', float(validPredictedGlassArea) / trueGlassArea

    for i in range(0, len(allTrueGlassArea)):
        print 'overall precision =', float(allValidPredictedGlassArea[i]) / allPredictedGlassArea[i]
        print 'overall recall =', float(allValidPredictedGlassArea[i]) / allTrueGlassArea[i]


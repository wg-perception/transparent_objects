#!/usr/bin/python
import sys
import glob
import re
import cv2
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

if __name__ == '__main__':
    assert len(sys.argv) == 5, sys.argv[0] + ' <testListFilename> <classifierFilename> <propagationScaligsList> <algorithmName>'

    segmentImage = '/home/ilysenkov/itseezMachine/home/ilysenkov/ecto/server_build/bin/enhancedGlassSegmenter'
    groundTruthPrefix = 'glassMask'

    testListFilename = sys.argv[1]
    classifierFilename = sys.argv[2]
    propagationScalingsList = sys.argv[3]
    algorithmName = sys.argv[4]

    propagationScalings = open(propagationScalingsList, 'r').read().splitlines()

    segmentationTemporaryFiles = [tempfile.mkstemp('.png', 'segmentation_') for scale in propagationScalings]

    (segmentationListFD, segmentationListFilename) = tempfile.mkstemp('.txt', 'segmentationList_', text=True)
    segmentationListFile = open(segmentationListFilename, 'w')
    for (fd, filename) in segmentationTemporaryFiles:
        segmentationListFile.write(filename + '\n')
    segmentationListFile.close()

    testFilenames = open(testListFilename, 'r').read().splitlines()
    
    allTrueGlassArea = [0] * len(segmentationTemporaryFiles)
    allPredictedGlassArea = [0] * len(segmentationTemporaryFiles)
    allValidPredictedGlassArea = [0] * len(segmentationTemporaryFiles)

    for imageFilename in testFilenames:
        print imageFilename

        match = re.search('_(0*([1-9][0-9]*))\.', imageFilename)
        if (match == None):
            match = re.search('_(0000(0))\.', imageFilename)
        assert match != None, 'Cannot parse an image index'

        fullImageIndex = match.group(1)
        imageIndex = match.group(2)

        groundTruthFilename = imageFilename.replace('/image_', '/' + groundTruthPrefix + '_')
        testFolder = imageFilename.replace('/image_' + fullImageIndex + '.png', '/')

        call([segmentImage, testFolder, fullImageIndex, classifierFilename, segmentationListFilename, propagationScalingsList, algorithmName])

        groundTruth = cv2.imread(groundTruthFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        for (index, temporaryFile) in enumerate(segmentationTemporaryFiles):
            segmentationImage = cv2.imread(temporaryFile[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)

            glassLabel = 255
            trueGlassArea = len(np.nonzero(groundTruth == glassLabel)[0])
            predictedGlassArea = len(np.nonzero(segmentationImage == glassLabel)[0])
            validPredictedGlassArea = len(np.nonzero(np.logical_and(groundTruth == glassLabel, segmentationImage == groundTruth))[0])
            allTrueGlassArea[index] += trueGlassArea
            allPredictedGlassArea[index] += predictedGlassArea
            allValidPredictedGlassArea[index] += validPredictedGlassArea
            if predictedGlassArea != 0:
                print 'precision =', float(validPredictedGlassArea) / predictedGlassArea
            if trueGlassArea != 0:
                print 'recall =', float(validPredictedGlassArea) / trueGlassArea

    invertedPrecisions = [None] * len(allTrueGlassArea)
    recalls = [None] * len(allTrueGlassArea)
    for i in range(len(allTrueGlassArea)):
        if allPredictedGlassArea[i] != 0:
            invertedPrecisions[i] = 1.0 - float(allValidPredictedGlassArea[i]) / allPredictedGlassArea[i]
        if allTrueGlassArea[i] != 0:
            recalls[i] = float(allValidPredictedGlassArea[i]) / allTrueGlassArea[i]
        print 'overall precision =', 1.0 - invertedPrecisions[i]
        print 'overall recall =', recalls[i]

    plt.plot(invertedPrecisions, recalls)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('1 - precision')
    plt.ylabel('recall')
    plt.title('Evaluation of Glass Segmentation Algorithms')
    plt.grid(True)
    plt.legend(['RGB data'], title='Algorithm', loc='best')
    plt.savefig('evaluation.png')
    plt.show()

    for segmentationFile in segmentationTemporaryFiles:
        os.close(segmentationFile[0])
        os.remove(segmentationFile[1])
    os.close(segmentationListFD)
    os.remove(segmentationListFilename)


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

class Evaluator:

    def __init__(self, curvePointsCount):
        self._recallValid = [0] * curvePointsCount
        self._recallAll = [0] * curvePointsCount
        self._precisionValid = [0] * curvePointsCount
        self._precisionAll = [0] * curvePointsCount

        self._invertedPrecisions = [None] * curvePointsCount
        self._recalls = [None] * curvePointsCount

    def addObservation(self, pointIndex, recallValid, recallAll, precisionValid, precisionAll):
        self._recallValid[pointIndex] += recallValid
        self._recallAll[pointIndex] += recallAll
        self._precisionValid[pointIndex] += precisionValid
        self._precisionAll[pointIndex] += precisionAll

    def computeStatistics(self):
        for i in range(len(self._recalls)):
            if self._recallAll[i] != 0:
                self._recalls[i] = float(self._recallValid[i]) / self._recallAll[i]
            if self._precisionAll[i] != 0:
                self._invertedPrecisions[i] = 1.0 - float(self._precisionValid[i]) / self._precisionAll[i]

    def printStatistics(self):
        print self._invertedPrecisions
        print self._recalls
        for i in range(len(self._recalls)):
            print 'recall =', self._recalls[i]
            print '1 - precision =', self._invertedPrecisions[i]

    def plot(self, title):
        plt.clf()
        plt.plot(self._invertedPrecisions, self._recalls, 'o-')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('1 - precision')
        plt.ylabel('recall')
        plt.title('Evaluation of Glass Segmentation Algorithms: ' + title)
        plt.grid(True)
        plt.legend(['RGB data'], title='Algorithm', loc='best')
        plt.savefig('evaluation_' + title + '.png')
#        plt.show()


def mask2dtAndEdgelsCount(mask):
    maskCopy = mask.copy()
    contours, hierarchy = cv2.findContours(maskCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contoursImage = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(contoursImage, contours, -1, (255))
    edgelsCount = len(np.nonzero(contoursImage == 255)[0])
    dt = cv2.distanceTransform(~contoursImage, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
    return [dt, edgelsCount]

def computeRecallPrecisionEdgels(groundTruthDT, segmentedImageDT):
    maxDistance = 2
    eps = 0.1

    recallEdgelsCount = len(np.nonzero(np.logical_and(groundTruthDT < eps, segmentedImageDT < maxDistance + eps))[0])
    precisionEdgelsCount = len(np.nonzero(np.logical_and(groundTruthDT < maxDistance + eps, segmentedImageDT < eps))[0])

    return [recallEdgelsCount, precisionEdgelsCount]


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
    
    areaEvaluator = Evaluator(len(segmentationTemporaryFiles))
    boundaryEvaluator = Evaluator(len(segmentationTemporaryFiles))

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
        glassLabel = 255
        groundTruthGlassMask = (groundTruth == glassLabel).astype(np.uint8)
        [groundTruthDT, groundTruthEdgelsCount] = mask2dtAndEdgelsCount(groundTruthGlassMask)

        for (index, temporaryFile) in enumerate(segmentationTemporaryFiles):
            segmentationImage = cv2.imread(temporaryFile[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
            [segmentationImageDT, segmentationImageEdgelsCount] = mask2dtAndEdgelsCount(segmentationImage)
            [recallEdgelsCount, precisionEdgelsCount] = computeRecallPrecisionEdgels(groundTruthDT, segmentationImageDT)

            trueGlassArea = len(np.nonzero(groundTruth == glassLabel)[0])
            predictedGlassArea = len(np.nonzero(segmentationImage == glassLabel)[0])
            validPredictedGlassArea = len(np.nonzero(np.logical_and(groundTruth == glassLabel, segmentationImage == groundTruth))[0])

            boundaryEvaluator.addObservation(index, recallEdgelsCount, groundTruthEdgelsCount, precisionEdgelsCount, segmentationImageEdgelsCount)
            areaEvaluator.addObservation(index, validPredictedGlassArea, trueGlassArea, validPredictedGlassArea, predictedGlassArea)

        areaEvaluator.computeStatistics()
        areaEvaluator.printStatistics()
        areaEvaluator.plot('area')

        boundaryEvaluator.computeStatistics()
        boundaryEvaluator.plot('boundary')


    areaEvaluator.computeStatistics()
    print 'final area: '
    areaEvaluator.printStatistics()
    areaEvaluator.plot('area')

    boundaryEvaluator.computeStatistics()
    boundaryEvaluator.printStatistics()
    boundaryEvaluator.plot('boundary')

    for segmentationFile in segmentationTemporaryFiles:
        os.close(segmentationFile[0])
        os.remove(segmentationFile[1])
    os.close(segmentationListFD)
    os.remove(segmentationListFilename)


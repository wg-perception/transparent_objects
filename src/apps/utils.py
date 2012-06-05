import cv2
import numpy as np
import glob
import re
import matplotlib.pyplot as plt

def mask2dtAndEdgelsCount(mask):
    maskCopy = mask.copy()
    contours, hierarchy = cv2.findContours(maskCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contoursImage = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(contoursImage, contours, -1, (255))
    edgelsCount = len(np.nonzero(contoursImage == 255)[0])
    dt = cv2.distanceTransform(~contoursImage, cv2.cv.CV_DIST_L1, cv2.cv.CV_DIST_MASK_PRECISE)
    return [dt, edgelsCount]

def computeRecallPrecisionEdgels(groundTruthDT, segmentedImageDT):
#    maxDistance = 2
    maxDistance = 4
    eps = 0.1

    recallEdgelsCount = len(np.nonzero(np.logical_and(groundTruthDT < eps, segmentedImageDT < maxDistance + eps))[0])
    precisionEdgelsCount = len(np.nonzero(np.logical_and(groundTruthDT < maxDistance + eps, segmentedImageDT < eps))[0])

    return [recallEdgelsCount, precisionEdgelsCount]

class Sample:
    def __init__(self):
        self._glassLabel = 255

    def getImage(self):
        image = cv2.imread(self.imageFilename, cv2.CV_LOAD_IMAGE_COLOR)
        return image

    def getGlassMask(self):
        fullGlassMask = cv2.imread(self.glassMaskFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if (fullGlassMask == None):
            return None
        glassMask = (fullGlassMask == self._glassLabel).astype(np.uint8)
        return glassMask

class TODBase:
    def __init__(self, basePath, objects):
        self._basePath = basePath
        self._objects = objects

    def getSamples(self):
        allSamples = []
        for obj in self._objects:
            testFolder = self._basePath + '/' + obj + '/'
            for imageFilename in sorted(glob.glob(testFolder + '/image_*.png')):
                match = re.search('_(0*([1-9][0-9]*))\.', imageFilename)
                if (match == None):
                    match = re.search('_(0000(0))\.', imageFilename)

                assert match != None, 'Cannot parse an image index'

                fullImageIndex = match.group(1)
                imageIndex = match.group(2)

                sample = Sample()
                sample.imageFilename = testFolder + '/image_' + fullImageIndex + '.png'
                sample.glassMaskFilename = testFolder + '/glassMask_' + fullImageIndex + '.png'
                allSamples.append(sample)

        return allSamples         
 
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


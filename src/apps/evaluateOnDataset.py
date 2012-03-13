#!/usr/bin/python
import sys
import glob
import re
import cv2
from subprocess import call
import numpy as np


if __name__ == '__main__':
    assert len(sys.argv) == 3, sys.argv[0] + ' <baseFolder> <classifierFilename>'

    objects = ('bank', 'bottle', 'bucket', 'glass', 'wineglass')
    segmentImage = '/home/ilysenkov/itseezMachine/home/ilysenkov/ecto/server_build/bin/enhancedGlassSegmenter'

    tmpSegmentationFilename = 'segmentation.png'
    groundTruthPrefix = 'glassMask'

    baseFolder = sys.argv[1]
    classifierFilename = sys.argv[2]

    testIndices = (27, 29, 31, 33, 36, 38, 40, 44, 46, 50, 52, 54, 56, 61, 67, 74, 78, 82, 85, 88)

    
    allTrueGlassArea = 0
    allPredictedGlassArea = 0
    allValidPredictedGlassArea = 0
    for obj in objects:
        if (obj != 'bank'):
            continue
        print obj
        testFolder = baseFolder + '/' + obj + '/'

        for imageFilename in sorted(glob.glob(testFolder + '/image_[0-9]*.png')):
            match = re.search('_(0*([1-9][0-9]*))\.', imageFilename)
            if (match == None):
                match = re.search('_(0000(0))\.', imageFilename)
            assert match != None, 'Cannot parse an image index'

            fullImageIndex = match.group(1)
            imageIndex = match.group(2)

            groundTruthFilename = imageFilename.replace('/image_', '/' + groundTruthPrefix + '_')
 
            if (int(imageIndex) not in testIndices):
                continue

            print imageFilename

            call([segmentImage, testFolder, fullImageIndex, classifierFilename, tmpSegmentationFilename])
            segmentation = cv2.imread(tmpSegmentationFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            groundTruth = cv2.imread(groundTruthFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE)

            glassLabel = 255
            trueGlassArea = len(np.nonzero(groundTruth == glassLabel)[0])
            predictedGlassArea = len(np.nonzero(segmentation == glassLabel)[0])
            validPredictedGlassArea = len(np.nonzero(np.logical_and(groundTruth == glassLabel, segmentation == groundTruth))[0])
            allTrueGlassArea += trueGlassArea
            allPredictedGlassArea += predictedGlassArea
            allValidPredictedGlassArea += validPredictedGlassArea

            print 'precision =', float(validPredictedGlass) / predictedGlassArea 
            print 'recall =', float(validPredictedGlass) / trueGlassArea

    print 'overall precision =', float(validPredictedGlass) / predictedGlassArea 
    print 'overall recall =', float(validPredictedGlass) / trueGlassArea


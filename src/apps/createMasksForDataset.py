#!/usr/bin/python
import cv2 
import sys 
import numpy as np
from subprocess import call
import glob
import re
import os
import tempfile

if __name__ == '__main__':
    assert len(sys.argv) == 3, sys.argv[0] + ' <baseFolder> <trainedModelsFolder>'

    backgroundValue = 127
    objects=('bank', 'bottle', 'bucket', 'glass', 'wineglass')
    objectMasker='/home/ilysenkov/itseezMachine/home/ilysenkov/ecto/server_build/bin/createObjectMask'
    backgroundMasker='/home/ilysenkov/itseezMachine/home/ilysenkov/ecto/recognition_kitchen/transparent_objects/src/apps/createBackgroundMask.py'

    baseFolder = sys.argv[1]
    trainedModelsFolder = sys.argv[2]

    (objectMaskFD, objectMaskFilename) = tempfile.mkstemp('.png', 'objMask_')
    (backgroundMaskFD, backgroundMaskFilename) = tempfile.mkstemp('.png', 'backMask_')
    for obj in objects:
        modelFilename = trainedModelsFolder + '/' + obj + '.xml' 
        testFolder = baseFolder + '/' + obj + '/'

        for imageFilename in sorted(glob.glob(testFolder + '/image_*.png')):
            print imageFilename
            match = re.search('_(0*([1-9][0-9]*))\.', imageFilename)
            if (match == None):
                match = re.search('_(0000(0))\.', imageFilename)

            assert match != None, 'Cannot parse an image index'

            fullImageIndex = match.group(1)
            imageIndex = match.group(2)

            call([objectMasker, testFolder, imageIndex, modelFilename, objectMaskFilename])
            imageFilename = testFolder + '/image_' + fullImageIndex + '.png'
            exitStatus = call([backgroundMasker, imageFilename, backgroundMaskFilename])
            if (exitStatus != 0):
                continue

            objectMask = cv2.imread(objectMaskFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            backgroundMask = cv2.imread(backgroundMaskFilename, cv2.CV_LOAD_IMAGE_GRAYSCALE)

            mask = objectMask
            mask[np.nonzero(backgroundMask)] = backgroundValue
            maskFilename = testFolder + '/glassMask_' + fullImageIndex + '.png'
            cv2.imwrite(maskFilename, mask)

            image = cv2.imread(imageFilename, cv2.CV_LOAD_IMAGE_UNCHANGED)
            image[np.nonzero(mask == backgroundValue)] = 0
            image[np.nonzero(mask == 255)] /= 1.5
            maskedImageFilename = testFolder + '/maskedImage_' + fullImageIndex + '.png'
            cv2.imwrite(maskedImageFilename, image)
#            cv2.imshow('masked image', image)
#            cv2.waitKey()
#            cv2.imshow('mask', mask)
#            cv2.waitKey()
    os.close(objectMaskFD)
    os.close(backgroundMaskFD)
    os.remove(objectMaskFilename)
    os.remove(backgroundMaskFilename)
    

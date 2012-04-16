import sys
import tempfile
import subprocess
import cv2
import numpy as np
import os
import utils

def computeVeksler(image, patchSize):
#    programExecutable='superpixels'
    programExecutable='/home/ilysenkov/itseezMachine/home/ilysenkov/ecto/server_build/superpixels'

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (inputImageFD, inputImageFilename) = tempfile.mkstemp('.pgm', 'image_')
    (outputImageFD, outputImageFilename) = tempfile.mkstemp('.ppm', 'image_')
    cv2.imwrite(inputImageFilename, grayImage)

    subprocess.call([programExecutable, inputImageFilename, outputImageFilename, str(patchSize)])
    segmentationFilename = outputImageFilename + '.txt'
    
    segmentation = np.loadtxt(segmentationFilename, np.int32)

    os.close(inputImageFD)
    os.remove(inputImageFilename)
    os.close(outputImageFD)
    os.remove(outputImageFilename)

    return segmentation

def computeFelzenszwalb(image, k):
    programExecutable='/home/ilysenkov/itseezMachine/home/ilysenkov/ecto/server_build/segment'
    sigma = 0.4
    minSize = 200

    (inputImageFD, inputImageFilename) = tempfile.mkstemp('.ppm', 'image_')
    (outputImageFD, outputImageFilename) = tempfile.mkstemp('.ppm', 'image_')
    cv2.imwrite(inputImageFilename, image)

    subprocess.call([programExecutable, str(sigma), str(k), str(minSize), inputImageFilename, outputImageFilename])
#    segmentationFilename = outputImageFilename + '.txt'
    segmentationFilename = 'segmentation.txt'
    
    segmentation = np.loadtxt(segmentationFilename, np.int32)

    os.close(inputImageFD)
    os.remove(inputImageFilename)
    os.close(outputImageFD)
    os.remove(outputImageFilename)

    return segmentation


def computeTurbopixels(imageFilename, turbopixelsCount):
    image = mlab.im2double(mlab.imread(imageFilename))
    boundary = mlab.superpixels(image, turbopixelsCount)
    return boundary

def main():
    glassLabel = 255
    thicknessForEvaluation = 20
    objects=('bank', 'bottle', 'bucket', 'glass', 'wineglass')
    basePath = '/media/2Tb/transparentBases/fixedOnTable/base/'

    base = utils.TODBase(basePath, objects)
    patchSizes = (5, 10, 20, 40, 100, 200, 400)
    fks = (10, 20, 50, 100, 200, 400, 800, 1600)

    allSamples = base.getSamples()
    evaluator = utils.Evaluator(len(patchSizes))
#    evaluator = utils.Evaluator(len(fks))
    for sample in allSamples:
        print sample.imageFilename
 
        groundTruthMask = sample.getGlassMask()
        if (groundTruthMask == None):
            continue
        [groundTruthDT, groundTruthEdgelsCount] = utils.mask2dtAndEdgelsCount(groundTruthMask)
        glassMask = groundTruthMask.copy()
        glassContours, hierarchy = cv2.findContours(glassMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        glassContoursImage = np.zeros(glassMask.shape, np.uint8)
        cv2.drawContours(glassContoursImage, glassContours, -1, (255), thicknessForEvaluation)

        image = sample.getImage()
        for idx, size in enumerate(patchSizes):
            segmentation = computeVeksler(image, size)
#        for idx, k in enumerate(fks):
#            segmentation = computeFelzenszwalb(image, k)
            minSegmentation = cv2.erode(segmentation.astype(np.float32), None)
            maxSegmentation = cv2.dilate(segmentation.astype(np.float32), None)
            boundaries = 255 * (maxSegmentation - minSegmentation != 0).astype(np.uint8)
            boundaries[glassContoursImage == 0] = 0
            boundariesEdgelsCount = len(np.nonzero(boundaries == 255)[0])
            boundariesDT = cv2.distanceTransform(~boundaries, cv2.cv.CV_DIST_L1, cv2.cv.CV_DIST_MASK_PRECISE)

            [recallEdgels, precisionEdgels] = utils.computeRecallPrecisionEdgels(groundTruthDT, boundariesDT)
            evaluator.addObservation(idx, recallEdgels, groundTruthEdgelsCount, precisionEdgels, boundariesEdgelsCount)
        evaluator.computeStatistics()
        evaluator.printStatistics()


    evaluator.plot('veksler')
#    evaluator.plot('felz')
    utils.plt.show()


#    cv2.imshow('boundaries', boundaries.astype(np.uint8))
#        cv2.imshow('boundaries', boundaries)
#    print boundaries.astype(np.uint8)
#        cv2.waitKey()


if __name__ == "__main__":
    main()
    

#!/usr/bin/python
import cv2 
import sys 
import numpy as np

def isInside(mat, point):
    x = point[0]
    y = point[1]
    return x >= 0 and y >= 0 and x < mat.shape[1] and y < mat.shape[0]

def drawFullLine(image, leftLine):
    leftStart = leftLine[0][0].copy()
    leftEnd = leftLine[-1][0].copy()
    leftVector = leftEnd - leftStart

    leftMin = leftStart
    while (isInside(image, leftMin)):
        leftMin -= leftVector

    leftMax = leftEnd
    while (isInside(image, leftMax)):
        leftMax += leftVector

    cv2.line(image, (leftMin[0], leftMin[1]), (leftMax[0], leftMax[1]), (255))


if __name__ == '__main__':
    filename = sys.argv[1]
    if len(sys.argv) > 2:
        outFilename = sys.argv[2]
    else:
        outFilename = None


    topLineShift = (0, 90)
    topLinePointShift = 10
    thresholdValue = 120
    minContourLength = 300
    patternSize = (4, 11)
    dilationIterationCount = 5
    visualize = False

    image = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    if visualize:
        cv2.imshow("image", image)
        cv2.waitKey()
    assert image != None, 'Cannot read ' + filename

    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 5.0
    params.minArea = 15.0
    blackDetector = cv2.SimpleBlobDetector(params)
    params.blobColor = 255

    whiteDetector = cv2.SimpleBlobDetector(params)
    whiteCenters = np.empty((0, 0))
    blackCenters = np.empty((0, 0))
    isBlackFound, blackCenters = cv2.findCirclesGrid(image, patternSize, blackCenters, cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING, blackDetector)
    isWhiteFound, whiteCenters = cv2.findCirclesGrid(image, patternSize, whiteCenters, cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING, whiteDetector)

    circlesImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(circlesImage, patternSize, blackCenters, isBlackFound)
    cv2.drawChessboardCorners(circlesImage, patternSize, whiteCenters, isWhiteFound)
    if visualize:
        cv2.imshow("corners", circlesImage)
        cv2.waitKey()

    assert (isBlackFound and isWhiteFound), 'Cannot find two circles grids'
    whiteCentroid = np.mean(whiteCenters, 0)
    blackCentroid = np.mean(blackCenters, 0)


    mask = np.zeros(image.shape, np.uint8)
    """    
    whiteHull = cv2.convexHull(whiteCenters)
    blackHull = cv2.convexHull(blackCenters)
    cv2.fillConvexPoly(mask, whiteHull.astype(np.int32), (255))
    cv2.fillConvexPoly(mask, blackHull.astype(np.int32), (255))
    dilationIterationsCount = 10
    mask = cv2.dilate(mask, None, mask, (-1, -1), dilationIterationsCount)
    cv2.imshow("dilated mask", mask)
    """

    leftLine = blackCenters[2*patternSize[0] - 1:len(blackCenters):2*patternSize[0]]
    rightLine = whiteCenters[0:len(whiteCenters):2*patternSize[0]]
    topLeftLine = blackCenters[-patternSize[0]:]
    topLeftLine -= topLineShift

    drawFullLine(mask, leftLine)
    drawFullLine(mask, rightLine)
    enlargedMaskShape = (mask.shape[0] + 2, mask.shape[1] + 2)
    floodFillMask = np.zeros(enlargedMaskShape, np.uint8)
    cv2.floodFill(mask, floodFillMask, (0, 0), 255)
    cv2.floodFill(mask, floodFillMask, (mask.shape[1] - 1, 0), 255)
    drawFullLine(mask, topLeftLine)
    cv2.floodFill(mask, floodFillMask, (int(topLeftLine[0][0][0]), int(topLeftLine[0][0][1])- topLinePointShift), 255)
    if visualize:
        cv2.imshow("filled mask", mask)

    retval, thresholdedImage = cv2.threshold(image, thresholdValue, 255, cv2.THRESH_BINARY )
    thresholdedImage[np.nonzero(mask)] = 0
    if visualize:
        cv2.imshow('gray image', image)
        cv2.imshow('thresholded image', thresholdedImage)

    contours, hierarchy = cv2.findContours(thresholdedImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contoursImage = np.zeros(image.shape, np.uint8)
    cv2.drawContours(contoursImage, contours, -1, (255))
    if visualize:
        cv2.imshow('contours', contoursImage)
        print [len(c) for c in contours]

    blackBorder = np.empty((0, 1, 2), np.int32)
    whiteBorder = np.empty((0, 1, 2), np.int32)
    for c in contours:
        if (len(c) < minContourLength):
           continue 
        centroid = np.mean(c, 0)
        if (np.linalg.norm(centroid - blackCentroid) < np.linalg.norm(centroid - whiteCentroid)):
            blackBorder = np.vstack((blackBorder, c)) 
        else:
            whiteBorder = np.vstack((whiteBorder, c)) 

    fiducialMask = np.zeros(image.shape, np.uint8)
    blackHull = cv2.convexHull(blackBorder)
    whiteHull = cv2.convexHull(whiteBorder)
    cv2.fillConvexPoly(fiducialMask, blackHull.astype(np.int32), (255))
    cv2.fillConvexPoly(fiducialMask, whiteHull.astype(np.int32), (255))
    if visualize:
        cv2.imshow("fiducial mask", fiducialMask)


    mask[np.nonzero(fiducialMask)] = 255
    if visualize:
        cv2.imshow("mask with fiducials", mask)

    mask = cv2.dilate(mask, None, mask, (-1, -1), dilationIterationCount)
    if visualize:
        cv2.imshow("final mask", mask)

    maskedImage = image.copy()
    maskedImage[np.nonzero(mask)] = 0

    if visualize:
        cv2.imshow("masked image", maskedImage)
        cv2.waitKey()

    if (outFilename != None):
        cv2.imwrite(outFilename, mask)


#!/usr/bin/python
import sys
import glob
from subprocess import call

if __name__ == '__main__':
    assert len(sys.argv) == 2, sys.argv[0] + ' <baseFolder>'

    objects = ('bank', 'bottle', 'bucket', 'glass', 'wineglass')
    createSegmentedImage = '/home/ilysenkov/itseezMachine/home/ilysenkov/ecto/server_build/bin/createSegmentedImage'
    segmentationPrefix = 'turbopixels'
    segmentedImagePrefix = 'segmentedImage'

    baseFolder = sys.argv[1]

    for obj in objects:
        testFolder = baseFolder + '/' + obj + '/'

        for imageFilename in sorted(glob.glob(testFolder + '/image_[0-9]*.png')):
            print imageFilename
            segmentationFilename = imageFilename.replace('/image_', '/' + segmentationPrefix + '_')
            segmentationFilename = segmentationFilename.replace('.png', '.txt')

            segmentedImageFilename = imageFilename.replace('/image_', '/' + segmentedImagePrefix + '_')
            segmentedImageFilename = segmentedImageFilename.replace('.png', '.xml')
            
            call([createSegmentedImage, imageFilename, segmentationFilename, segmentedImageFilename])

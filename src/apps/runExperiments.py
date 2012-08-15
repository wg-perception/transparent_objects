import subprocess
import os
import sys

runner='/home/ilysenkov/ecto_fuerte/build/bin/transparentExperiments'
trainedModelsPath='/media/2Tb/transparentBases/trainedModels/'

dataset='/media/2Tb/transparentBases/different_clutter/base_3/' 
datasetName = 'different_clutter_3'
allObjects = ['bank', 'bucket', 'bottle', 'glass', 'wineglass']
#dataset='/media/2Tb/transparentBases/good_clutter/base/' 
#datasetName = 'good_clutter'
#allObjects = ['bank', 'bucket', 'bottle', 'glass', 'wineglass', 'sourCream']

baseLogsPath = '/home/ilysenkov/results/occlusions/'

if __name__ == '__main__':
    assert len(sys.argv) == 2, sys.argv[0] + ' <experimentsName>'
    baseLogsPath += sys.argv[1]

    logsPath = baseLogsPath + '/' + datasetName

    if not os.path.exists(baseLogsPath):
        os.makedirs(baseLogsPath)

    if not os.path.exists(logsPath):
        os.makedirs(logsPath)

    for obj in allObjects:
        logFilename = logsPath + '/' + obj
        logFile = open(logFilename, 'w')

        command = [runner, trainedModelsPath, dataset, obj]
        process = subprocess.Popen(command, stdout=logFile, stderr=logFile)

        #process = subprocess.Popen(command, stdout=subprocess.PIPE)
        #print process.stdout.read()

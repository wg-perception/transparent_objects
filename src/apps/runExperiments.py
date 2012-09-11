import subprocess
import os
import sys
import psutil
import time
import re

runner='/home/ilysenkov/ecto_fuerte/build/bin/transparentExperiments'
#runner='/home/ilysenkov/itseezMachine/home/ilysenkov/ecto/server_build/bin/transparentExperiments'

runnerName = 'transparentExperiments'
trainedModelsPath='/media/2Tb/transparentBases/trainedModels/'
experimentsCoresCount = 7

#dataset='/media/2Tb/transparentBases/different_clutter/base_3/'
#datasetName = 'different_clutter_3'
#allObjects = ['bank', 'bucket', 'bottle', 'glass', 'wineglass']

#dataset='/media/2Tb/transparentBases/different_clutter/base_ocl/'
#datasetName = 'different_clutter_ocl'
#allObjects = ['bank', 'bucket', 'bottle', 'glass', 'wineglass']

#dataset='/media/2Tb/transparentBases/good_clutter/base/'
#datasetName = 'good_clutter'
#allObjects = ['bank', 'bucket', 'bottle', 'glass', 'wineglass', 'sourCream']

#dataset='/media/2Tb/transparentBases/fixedOnTable/base/'
#datasetName = 'fixed_on_table'
#allObjects = ['bank', 'bucket', 'bottle', 'glass', 'wineglass', 'sourCream']

dataset='/media/2Tb/transparentBases/finalClutter/base/'
datasetName = 'finalClutter'
allObjects = ['bank', 'bucket', 'bottle', 'glass']

baseLogsPath = '/home/ilysenkov/results/occlusions/'
bigSleepTime = 10
smallSleepTime = 1

def getRunProcessesCount():
    processes = psutil.get_process_list()
    runProcessesCount = 0
    for proc in processes:
        match = re.match(runnerName, proc.name)
        if (match != None):
            runProcessesCount += 1
    return runProcessesCount

if __name__ == '__main__':
    assert len(sys.argv) == 2, sys.argv[0] + ' <experimentsName>'
    baseLogsPath += sys.argv[1]

    logsPath = baseLogsPath + '/' + datasetName

    if not os.path.exists(baseLogsPath):
        os.makedirs(baseLogsPath)

    if not os.path.exists(logsPath):
        os.makedirs(logsPath)


    for obj in allObjects:
        runProcessesCount = getRunProcessesCount()
        while (runProcessesCount >= experimentsCoresCount):
            time.sleep(bigSleepTime)
            runProcessesCount = getRunProcessesCount()

        logFilename = logsPath + '/' + obj
        logFile = open(logFilename, 'w')

        command = [runner, trainedModelsPath, dataset, obj]
        process = subprocess.Popen(command, stdout=logFile, stderr=logFile)

        print obj
        time.sleep(smallSleepTime)

        #process = subprocess.Popen(command, stdout=subprocess.PIPE)
        #print process.stdout.read()

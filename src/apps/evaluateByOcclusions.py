import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import bisect
import os.path

if __name__ == '__main__':
    boundaries = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]

#final
    maxFinalTranslation = 0.02
    finalTranslationIndex = 4

#initial
    maxInitialTranslation = 0.06
    initialTranslationIndex = 2

    assert len(sys.argv) == 2, sys.argv[0] + ' <resultsPath>'
    basePath = sys.argv[1]

#    allDatasets = ['good_clutter']
#    allDatasets = ['different_clutter_3']
#    allDatasets = ['different_clutter_ocl']
#    allDatasets = ['finalClutter']

#    allDatasets = ['good_clutter', 'different_clutter_3']
#    allDatasets = ['fixedOnTable_3', 'good_clutter', 'different_clutter_3']
#    allDatasets = ['good_clutter', 'different_clutter_3', 'finalClutter']

#    allDatasets = ['fixedOnTable_3', 'fixed_on_table', 'good_clutter', 'different_clutter_3']
    allDatasets = ['fixedOnTable_3', 'fixed_on_table', 'good_clutter', 'different_clutter_3', 'finalClutter']

    allObjects = ['bank', 'bucket', 'bottle', 'glass', 'wineglass', 'sourCream']

    csvData = []
    for dataset in allDatasets:
        for obj in allObjects:
            filename = basePath + "/" + dataset + "/" + obj
            if not os.path.exists(filename):
                continue

            allLines = open(filename, 'r').read().splitlines()
            for line in allLines:
                match = re.match('Result: (.*)', line)
                if (match != None):
                    csvData.append(match.group(1))

    data = np.loadtxt(csvData, delimiter=", ")

    binCount = len(boundaries)
    binCardinalities = [0] * binCount
    bins = np.zeros((binCount, data.shape[1]), data.dtype)

    data[:, finalTranslationIndex] = data[:, finalTranslationIndex] < maxFinalTranslation
    data[:, initialTranslationIndex] = data[:, initialTranslationIndex] < maxInitialTranslation

    for row in data:
        binIndex = bisect.bisect_left(boundaries, row[0])
        binCardinalities[binIndex] += 1
        bins[binIndex] += row

    print binCardinalities

    for idx, val in enumerate(binCardinalities):
        if (val != 0):
            bins[idx] /= val

    validBins = [idx for idx, val in enumerate(binCardinalities) if val != 0]

    plt.plot(bins[validBins, 0], bins[validBins, initialTranslationIndex], 'o-')
    plt.plot(bins[validBins, 0], bins[validBins, finalTranslationIndex], 'o-')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.xlabel("Occlusion percentage")
    plt.ylabel("Success rate")
    plt.legend(['initial', 'final'])
    plt.title('Evaluation of pose estimation accuracy')
    plt.show()


#    y1 = [ 0.98791019,  0.96,       0.92622951,  0.77586207,  0.784, 0.60714286,  0.30645161,  0.17910448,  0.04166667]
#    y2 = [ 0.97769517,  0.965,       0.92622951,  0.74137931,  0.08196721, 0.  ]
#    print validBins
#    usedBins2 = [0, 2, 3, 4, 5, 6]
#    usedBins1 = [0, 2, 3, 4, 5, 6, 7, 8, 9]
#    plt.plot(bins[usedBins1, 0], y1, 'o-')
#    plt.plot(bins[usedBins2, 0], y2, 'o-')
#    plt.legend(['FDCM LM-ICP', 'last week results'])

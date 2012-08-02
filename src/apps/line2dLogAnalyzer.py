import sys
import re
import numpy as np
import matplotlib.pyplot as plt

def getStatistics(logFilename):
    lines = open(logFilename, 'r').read().splitlines()
    templatesCount = []
    successRates = []
    for line in lines:
        match = re.match('templates count: ([0-9]*)', line)
        if (match != None):
            templatesCount.append(int(match.group(1)))

        match = re.match('Success rate: (.*)', line)
        if (match != None):
            successRates.append(float(match.group(1)))

    a1 = np.array((templatesCount))
    a2 = np.array((successRates))
    statistics = np.vstack((a1, a2)).T
    return statistics

if __name__ == '__main__':
    assert len(sys.argv) > 2, sys.argv[0] + ' <rowsCount> <logFilename> [...]'

    rowsCount = int(sys.argv[1])
    statistics = np.zeros((rowsCount, 2))
    for filename in sys.argv[2:]:
        print filename
        currentStatistics = getStatistics(filename)
        statistics += currentStatistics[:rowsCount, :]

    statistics /= len(sys.argv) - 2.0

    plt.plot(statistics[:, 0], statistics[:, 1], 'o-')
    plt.ylabel('Success rate')
    plt.xlabel('Templates count (templates which are nearest to ground truth are used first)')
    plt.grid()
    plt.title('Evaluation of Line2D')
    plt.ylim(0, 1)
    plt.show()

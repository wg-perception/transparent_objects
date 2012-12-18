import sys
import re

if __name__ == '__main__':
    assert len(sys.argv) == 2, sys.argv[0] + ' <logFilename>'

    logFilename = sys.argv[1]
    lines = open(logFilename, 'r').read().splitlines()
    testIndex = -1
    for line in lines:
        match = re.match('Test: ([0-9]*) ([0-9]*)', line)
        if (match != None):
            testIndex = int(match.group(1))

        match = re.match('quality: (.*)', line)
        if (match != None):
            print testIndex, float(match.group(1))

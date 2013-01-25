import sys
import re
import numpy as np
import openopt

from scipy.sparse import lil_matrix

def readProblemInstance(filename):
  lines = open(filename, 'r').read().splitlines()

  isReadingConstraints = False
  for line in lines:
    if (isReadingConstraints):
      words = line.split()
      pairs = zip(words[1::2], words[2::2])
      for pair in pairs:
        A[constraintIndex, int(pair[0])] = pair[1]

      b[constraintIndex] = words[0]
      constraintIndex += 1

    match = re.match('Volume variables: (.*)', line)
    if (match != None):
      volumeVariablesCount = int(match.group(1)) 

    match = re.match('Pixel variables: (.*)', line)
    if (match != None):
      pixelVariablesCount = int(match.group(1)) 

    match = re.match('Constraints: (.*)', line)
    if (match != None):
      constraintsCount = int(match.group(1))
      variablesCount = volumeVariablesCount + pixelVariablesCount
      print "Problem dimesions: %d x %d" % (constraintsCount, variablesCount)
#      A = np.zeros((constraintsCount, variablesCount), dtype=np.int8)
      A = lil_matrix((constraintsCount, variablesCount))
      b = np.empty((constraintsCount))
      isReadingConstraints = True
      constraintIndex = 0

  f_volume = np.zeros((volumeVariablesCount))
  f_pixel = np.ones((pixelVariablesCount))
  f = np.hstack((f_volume, f_pixel))

  return (f, A, b)


if __name__ == '__main__':
  assert len(sys.argv) == 2, sys.argv[0] + ' <problemInstanceFilename>'

  (f, A, b) = readProblemInstance(sys.argv[1])
  print f
#  print A
  print b
 
  lb = np.zeros(len(f))
  ub = np.ones(len(f))
  print lb 
  print ub 

  p = openopt.LP(f, A=A, b=b, lb=lb, ub=ub)


#  intVars = range(0, len(f))
#  p = openopt.MILP(f, A=A, b=b, lb=lb, ub=ub, intVars=intVars)
#  print 'Saving in MPS format...'
#  isExported = p.exportToMPS('lp.mps')
#  print 'Done:', isExported
#  sys.exit(0)

  r = p.maximize('glpk')
#  r = p.maximize('cvxopt_lp')
#  r = p.maximize('pclp')
#  r = p.maximize('lpSolve')

  print('objFunValue: %f' % r.ff) # should print 204.48841578
  print('x_opt: %s' % r.xf) # should print [ 9.89355041 -8.          1.5010645 ]

  np.savetxt('solution.csv', r.xf)

  '''
  solution = np.loadtxt('solution.csv')
  print '<?xml version = "1.0" standalone="yes"?>\n<variables>'
  for idx, val in enumerate(solution):
      print '   <variable name="C%d" index="%d" value="%f"/>' % (idx + 1, idx + 1, val)
  print '</variables>'
  '''

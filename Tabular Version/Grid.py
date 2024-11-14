from Run import *

gamma   = 0.95
noSteps = 10**4
noRuns  = 10000
exp     = "egreedy1divn^0.5"
# lrs     = [ "1divn^0.8", "1divn" ]
lrs     = [  "1divn" ]


widths = [ 22 ]
for lr in lrs:
    for width in widths:
        World   = Grid( 3, 3, width )
        runExperiment(World, gamma, noSteps, noRuns, exp, lr)


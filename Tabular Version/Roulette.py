from Run import *

gamma   = 0.95
World   = Roulette( 0 )
noSteps = 10**5
noRuns  = 10

exp = "onebyone"
lrs  = ["1divn^0.8" ]


# lrs     = [  "1divn" ]
    
for lr in lrs:
    runExperiment(World, gamma, noSteps, noRuns, exp, lr)

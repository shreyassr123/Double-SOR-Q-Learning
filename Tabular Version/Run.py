from RL import *
import math

def notate( x ):
    suffices =[ "", "K", "M", "B" ]
    index = 0

    while math.log(x,1000) >= 2:
        index += 1
        x /= 1000

    return str(x) + suffices[index]

def runExperiment( World, gamma, noSteps, noRuns, exp, lr):
    
    S = World.S
    A = World.A
    indices = [ (i+1)*noSteps/100 for i in range(100) ]
    
    nIndices = len( indices )
    
    pos = 0
    #algs = [ "Q", "DoubleQ","SORQ","SORDQ" ]
    
    algs = [ "Q","SORQw","DoubleQ","SORDQw"]
    last = {}
    R = {}
    
    if exp == "onebyone":
        noSteps *= sum(A)
        indices = [ sum(A)*i for i in indices ]

    cleanstring = lambda string: string.replace(".","").replace("^","")
    
    outFile = World.name + "_E"+cleanstring(exp)+"_L"+cleanstring(lr)+"_S"+notate(noSteps)+"_R"+notate(noRuns)+"_D"+cleanstring(str(gamma))+"_A"+str(A[0])+"hmm3"+".py"
    print (outFile)

    MaxQ = {}
    MeanQ = {}
    MinQ = {}
    Qs = {}
    Counts = {}

    for alg in algs:

        R[alg]      = [ 0. for i in range( nIndices ) ]
        Qs[alg]     = [ [ [ 0. for i in range( nIndices ) ] for a in range( A[s] ) ] for s in range( S ) ]
        
        MaxQ[alg] = [ [ 0. for i in range( nIndices ) ] for s in range( S ) ]
        MeanQ[alg] = [ [ 0. for i in range( nIndices ) ] for s in range( S ) ]
        MinQ[alg] = [ [ 0. for i in range( nIndices ) ] for s in range( S ) ]
        
        for run in range( noRuns ):
            
            World.reset()
            
            if lr == "1divn":
                alpha   = LearningRateDecreasing( S, A, lambda n: 1.0/n, lambda n: 1.0/n )
                beta    = LearningRateDecreasing( S, A, lambda n: 1.0/n, lambda n: 1.0/n )
            elif lr[:6] == "1divn^":
                pow     = float( lr[6:] )
                alpha   = LearningRateDecreasing( S, A, lambda n: 1.0/n**pow, lambda n: 1.0/n**pow )
                beta    = LearningRateDecreasing( S, A, lambda n: 1.0/n**pow, lambda n: 1.0/n**pow )
            elif lr[:5] == "const":
                const   = float( lr[5:] )
                alpha   = LearningRateDecreasing( S, A, lambda n: const, lambda n: const )
                beta    = LearningRateDecreasing( S, A, lambda n: const, lambda n: const )
            
            RL      = RLAgent( S, A, alg, exp, gamma, alpha, beta )

            st  = World.st
            at  = RL.getAction( st )

            i = 0

            index = 0

            while i < noSteps:
                if i < noSteps - 10:
                    st_     = World.act( at )
                else:
                    st_     = World.act( at )
                
                eoe     = World.endOfEpisode
                rt      = World.rt
                
                R[alg][index] += rt/noRuns
                
                if exp=="onebyone":
                    if (i+1) % sum(A) == 0:
                        un = "now"
                    else:
                        un = "no"
                else:
                    un = "always"
                
                
                if alg == "SORDQw":
                    
                    at_ = RL.step(st, at, rt, st_, eoe, un, i)
                    
                elif alg == "SORQw" :  
                    
                    at_ = RL.step(st, at, rt, st_, eoe, un, i)
                    
                else:
                    
                    at_     = RL.step( st, at, rt, st_, eoe, updatenext=un )    
                
                st      = st_
                at      = at_
                
                i += 1
                
                if i in indices:
                    
                    for s in range(S):
                        MaxQ[alg][s][index]     += max(RL.Q[s])/noRuns
                        MeanQ[alg][s][index]    += sum(RL.Q[s])/(noRuns*A[s])
                        MinQ[alg][s][index]     += min(RL.Q[s])/noRuns
                        
                        for a in range(A[s]):
                            Qs[alg][s][a][index] += RL.Q[s][a]/noRuns
                    
                    index += 1
            
        
        print (alg)
        print (max(R[alg]), sum(R[alg])/len(R[alg]), min(R[alg]), R[alg][-1])
        
        Counts[alg] = alpha.saCounts
        
    outStr = str( tuple( algs ) ) + "\n"

    outStr += "import pylab\n"
    outStr += "x = " + str(indices) + "\n"
    outStr += "R = [\n"
    for alg in algs:
        outStr += " ["+",".join([ "%1.3f" % rw for rw in R[alg] ]) +"],\n"
    outStr += "]\n"
    outStr += "Opt = 4.95\n"

    outStr += "Counts = [\n"
    for alg in algs:
        outStr += " [\n"
        for s in range(World.S):
            outStr += "   [" + ",".join( [ "%1.0f" % Counts[alg][s][a] for a in range( World.A[s] ) ] ) + "],\n"
        outStr += " ],\n"
    outStr += "]\n"

    outStr += "Q = [\n"
    for alg in algs:
        outStr += " [\n"
        for s in range(World.S):
            outStr += "  [\n"
            for a in range(World.A[s]):
                outStr += "   [" + ",".join( [ "%1.3f" % Qs[alg][s][a][n] for n in range( nIndices ) ] ) + "],\n"
            outStr += "  ],\n"
        outStr += " ],\n"
    outStr += "]\n"
    
    outStr += "MaxQ = [\n"
    for alg in algs:
        outStr += " [\n"
        for s in range(World.S):
            outStr += "   [" + ",".join( [ "%1.3f" % MaxQ[alg][s][n] for n in range( nIndices ) ] ) + "],\n"
        outStr += " ],\n"
    outStr += "]\n"
    
    outStr += "MeanQ = [\n"
    for alg in algs:
        outStr += " [\n"
        for s in range(World.S):
            outStr += "   [" + ",".join( [ "%1.3f" % MeanQ[alg][s][n] for n in range( nIndices ) ] ) + "],\n"
        outStr += " ],\n"
    outStr += "]\n"
    
    outStr += "MinQ = [\n"
    for alg in algs:
        outStr += " [\n"
        for s in range(World.S):
            outStr += "   [" + ",".join( [ "%1.3f" % MinQ[alg][s][n] for n in range( nIndices ) ] ) + "],\n"
        outStr += " ],\n"
    outStr += "]\n"
    
    f = open( outFile, "w" )
    f.write( outStr )
    f.close()
    
    print (outFile)

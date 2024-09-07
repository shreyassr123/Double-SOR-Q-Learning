import random
import numpy as np
def choice( X, N ):
    return X[ int(N*random.random()) ]

def randrange( N ):
    return int(N*random.random())

def argmax( X ):
    maxX = X[0]
    maxI = [0]
    maxN = 1

    for i in range( 1, len( X ) ):
        if X[i] > maxX:
            maxX = X[i]
            maxI = [i]
        elif X[i] == maxX:
            maxI.append(i)
            maxN += 1

    return (maxI,maxN)

class LearningRateConstantPer:

    def __init__( self, S, A, value ):
        self.sValue = [ value for s in range( S ) ]
        self.saValue = [ [ value for a in range( A[s] ) ] for s in range( S ) ]

        self.sCounts     = [ 0 for s in range( S ) ]
        self.saCounts    = [ [ 0 for a in range( A[s] ) ] for s in range( S ) ]

    def getS( self, st ):
        self.sCounts[ st ] += 1
        return self.sValue[ st ]

    def getSA( self, st, at ):
        self.saCounts[ st ][ at ] += 1
        return self.saValue[ st ][ at ]


class LearningRateConstant:

    def __init__( self, value ):
        self.value = value

    def getS( self, st ):
        return self.value

    def getSA( self, st, at ):
        return self.value


class LearningRateDecreasing:

    def __init__( self, noStates, noActions, sFunction, saFunction ):
        self.sFunction  = sFunction
        self.saFunction = saFunction

        self.S = noStates
        self.A = noActions

        self.reset()

    def reset( self ):
        self.sCounts     = [ 0 for s in range( self.S ) ]
        self.saCounts    = [ [ 0 for a in range( self.A[s] ) ] for s in range( self.S ) ]

    def getS( self, st ):
        self.sCounts[ st ] += 1
        return self.sFunction( self.sCounts[ st ] )

    def getSA( self, st, at ):
        self.saCounts[ st ][ at ] += 1
        return self.saFunction( self.saCounts[ st ][ at ] )

class E_Greedy:

    def __init__( self, noStates, noActions, sFunction, saFunction ):
        self.sFunction  = sFunction
        self.saFunction = saFunction

        self.S = noStates
        self.A = noActions

        self.reset()

    def reset( self ):
        self.sCounts     = [ 0 for s in range( self.S ) ]
        self.saCounts    = [ [ 0 for a in range( self.A[s] ) ] for s in range( self.S ) ]

    def getAction( self, st, Q, maxQ, maxA, maxN ):

        self.sCounts[st]        += 1

        exp = self.sFunction( self.sCounts[ st ] )

        if random.random() >= exp:
            if maxN > 1:
                at = choice( maxA, maxN )
            else:
                at = maxA[0]
        
        else:
            at = randrange( self.A[st] )
        
        self.saCounts[st][at]   += 1

        return at

    def getPolicy( self, st, Q, maxQ, maxA, maxN ):

        exp         = self.sFunction( self.sCounts[ st ] + 1 )

        policy      = [ exp/self.A[st] for a in range( self.A[st] ) ]

        for a in maxA:
            policy[a] += ( 1.0 - exp )/maxN

        return policy

class OneByOne:

    def __init__( self, noStates, noActions ):
        self.S = noStates
        self.A = noActions
        
        self.reset()

    def reset( self ):
        self.sCounts     = [ 0 for s in range( self.S ) ]
        self.saCounts    = [ [ 0 for a in range( self.A[s] ) ] for s in range( self.S ) ]

    def getAction( self, st, Q, maxQ, maxA, maxN ):

        at = self.sCounts[st] % self.A[st]

        self.sCounts[st]        += 1
        self.saCounts[st][at]   += 1


        return at

    def getPolicy( self, st, Q, maxQ, maxA, maxN ):

        # This doesn't give the actual (deterministic) policy, but rather
        # a stochastic equiprobable version. This way, Sarsa and E-Sarsa
        # differ and it is in the spirit of the exploration.

        policy      = [ 1.0/self.A[st] for a in range( self.A[st] ) ]

        return policy

class Grid:

    def __init__( self, noRows, noCols, width ):
        self.name   = "Grid"+str(width)
        self.R      = noRows
        self.C      = noCols
        self.S      = self.R*self.C
        self.A      = [ 4 for s in range( self.S ) ]
        self.W      = width
        
        self.st     = 0
        

    def reset( self ):
        self.st     = 0
        
    def act( self, at ):
        
        self.endOfEpisode = 0
        
        R = self.R
        C = self.C
        
        if self.st == R*C - 1:
            #Goal
            self.st = 0
            self.rt = 5.
            self.endOfEpisode = 1
        
        elif at == 0 and self.st % R < R - 1:      #up
            if random.random() < 0.9:
                   self.st += 1
            else :
                   self.st = self.st
            
        elif at == 1 and self.st < (C-1)*R:      #right
            if random.random() < 0.9:
                   self.st += R
            else :
                   self.st = self.st
        

        elif at == 2 and self.st % R > 0:        #down
            if random.random() < 0.9:
                   self.st -= 1
            else :
                   self.st = self.st
        

        elif at == 3 and self.st > R:            #left
            if random.random() < 0.9:
                   self.st -= R
            else :
                   self.st = self.st
        
        
        if not self.endOfEpisode:
            if random.random() < 0.5:
                self.rt = -1. - self.W/2.
            else:
                self.rt = -1. + self.W/2.

            self.endOfEpisode = 0

        return self.st




class Roulette:

    def __init__( self, walkaway=0 ):
        self.name   = "Roulette"+str(walkaway)
        self.walkaway = walkaway
        
        self.S = 1
        self.st = 0
        
        # action, multiplier, probablity of success
        self.odds =       [ ('number ' + str(i),                                         35.,1./38.) for i in range(-1,37 ) ]
        self.odds.append(   ('split 00,0',                                               17.,2./38.) )
        self.odds.extend( [ ('split '+str(i+1)+','+str(i+2),                             17.,2./38.) for i in range( 0,36,3 ) ] )
        self.odds.extend( [ ('split '+str(i+1)+','+str(i+2),                             17.,2./38.) for i in range( 1,36,3 ) ] )
        self.odds.extend( [ ('split '+str(i+1)+','+str(i+4),                             17.,2./38.) for i in range( 0,33,3 ) ] )
        self.odds.extend( [ ('split '+str(i+1)+','+str(i+4),                             17.,2./38.) for i in range( 1,33,3 ) ] )
        self.odds.extend( [ ('split '+str(i+1)+','+str(i+4),                             17.,2./38.) for i in range( 2,33,3 ) ] )
        self.odds.extend( [ ('basket 0,1,2',11,3./38.),('basket 0,00,2',                 11.,3./38.),('basket 00,2,3',11,3./37.) ] )
        self.odds.extend( [ ('street '+str(i+1)+','+str(i+2)+','+str(i+3),               11.,3./38.) for i in range(0,36,3) ] )
        self.odds.extend( [ ('corner '+str(i+1)+','+str(i+2)+','+str(i+4)+','+str(i+5),  8., 4./38.) for i in range( 0,36,3 )] )
        self.odds.extend( [ ('corner '+str(i+1)+','+str(i+2)+','+str(i+4)+','+str(i+5),  8., 4./38.) for i in range( 1,36,3 )] )
        self.odds.extend( [ ('top line ',                                                6., 5./38.) for i in range( 1,36,3 )] )
        self.odds.extend( [ ('six line '+str(i+1),                                       5., 6./38.) for i in range( 0,33,3 ) ] )
        self.odds.extend( [ ('column '+str(i+1),                                         2., 12./38.) for i in range( 3 ) ] )
        self.odds.extend( [ ('dozen '+str(i+1),                                          2., 12./38.) for i in range( 3 ) ] )
        self.odds.extend( [ ('odd',                                                      1., 18./38.) ] )
        self.odds.extend( [ ('even',                                                     1., 18./38.) ] )
        self.odds.extend( [ ('red',                                                      1., 18./38.) ] )
        self.odds.extend( [ ('black',                                                    1., 18./38.) ] )
        self.odds.extend( [ ('1-18',                                                     1., 18./38.) ] )
        self.odds.extend( [ ('18-36',                                                    1., 18./38.) ] )
        
        self.A = [len( self.odds ) + 1]
    
    def reset( self ):
        pass
    
    def act( self, at, prt=0 ):
        
        if at == self.A[0] - 1: # leave
            if prt:
                print ("leaving")
            self.endOfEpisode = 1
            self.rt = self.walkaway
        
        else:
            self.endOfEpisode = 0
            bet = self.odds[at]
            if prt:
                print ("placing bet: " + bet[0])
            if random.random() < bet[2]:
                #win!
                if prt:
                    print ("you win $" + str(bet[1]) + "!")
                self.rt = bet[1]
            else:
                #lose...
                if prt:
                    print( "you loose $1...")
                self.rt = -1.
        
        return 0 # state is alway zero

class RLAgent:

    def __init__( self, noStates, noActions, alg, exp, gamma, alpha, beta=0.0, initQ=0.0, initV=0.0 ):
        
        self.lastdir = [ [ 1 for a in range( noActions[s] ) ] for s in range( noStates ) ]
        if alg=="QV":
            self.step   = self.QV
        
        elif alg=="Q":
            self.step   = self.Q
        
        elif alg=="DoubleQ":
            self.step   = self.DoubleQ
        
        elif alg=="Sarsa":
            self.step   = self.Sarsa
        
        elif alg=="ExpectedSarsa":
            self.step   = self.ExpectedSarsa
        
        elif alg=="SORQ":
            self.step   = self.SORQ  
            
        elif alg=="SORQw":
            self.step   = self.SORQw     
        
        elif alg=="SORDQ":
            self.step   = self.SORDQ  
            
        elif alg=="SORDQw":
            self.step   = self.SORDQw     
        else:
            raise ValueError( "undefined value for alg: " + alg )
        
        self.synchronous = 0
        if exp=="egreedy1divn":
            self.exp    = E_Greedy( noStates, noActions, lambda n: 1./n, lambda n: 1./n )
        
        elif exp[:13]=="egreedy1divn^":
            pow         = float( exp[13:] )
            self.exp    = E_Greedy( noStates, noActions, lambda n: 1./n**pow, lambda n: 1./n**pow )
        
        elif exp[:12]=="egreedyconst":
            const       = float( exp[12:] )
            self.exp    = E_Greedy( noStates, noActions, lambda n: const, lambda n: const )
        
        elif exp=="onebyone":
            self.exp    = OneByOne( noStates, noActions )
            self.synchronous = 1
            
        elif exp=="synchronous":
            self.exp    = OneByOne( noStates, noActions )
            self.synchronous = 1
        
        else:
            raise ValueError( "unknown exploration: " + exp )
        
        self.gamma  = gamma
        self.alpha  = alpha
        self.beta   = beta
        
        self.S      = noStates
        self.A      = noActions
        
        self.tot_count = np.zeros((self.S,self.S,len(self.A)))
        
        self.Q      = [ [ initQ for a in range( noActions[s] ) ] for s in range( noStates ) ]
        self.P      = [ [ initQ for a in range( noActions[s] ) ] for s in range( noStates ) ]
        self.maxQ   = [ initQ for s in range( noStates ) ]
        self.maxP   = [ initQ for s in range( noStates ) ]
        self.maxA   = [ [ a for a in range( noActions[s] ) ] for s in range( noStates ) ]
        self.maxB   = [ [ a for a in range( noActions[s] ) ] for s in range( noStates ) ]
        self.maxN   = [ noActions[s] for s in range( noStates ) ]
        self.V      = [ initV for s in range( noStates ) ]
        self.W      = [ initV for s in range( noStates ) ]
        self.w = 1
        
        self.w_history = []  # Initialize an empty list to store w values
        
        self.explore()
    
    def explore( self ):
        self.getAction = lambda st : self.exp.getAction( st, self.Q[st], self.maxQ[st], self.maxA[st], self.maxN[st] )
    
    def exploit( self ):
        self.getAction = lambda st : choice( self.maxA[st], self.maxN[st] )
    
    def getPolicy( self, st ):
        return self.exp.getPolicy( st, self.Q[st], self.maxQ[st], self.maxA[st], self.maxN[st] )
    
    def QV( self, st, at, rt, st_, endOfEpisode=0, update=1, updatenext="always", norm="L2" ):
        alpha   = self.alpha.getSA( st, at )
        beta    = self.beta.getS( st )
        
        if endOfEpisode:
            gamma   = 0.0
        else:
            gamma   = self.gamma
        
        at_     = self.getAction( st_ )
        
        if update:
            self.Q[st][at] += alpha*( rt + gamma*self.V[st_] - self.Q[st][at] )
            self.V[st]     += beta*(  rt + gamma*self.V[st_] - self.V[st] )
            
            if updatenext == "always":
                self.maxQ[st] = max( self.Q[st] )
                self.maxA[st] = [ a for a in range( self.A[st] ) if self.Q[st][a] == self.maxQ[st] ]
                self.maxN[st] = len( self.maxA[st] )
        
        if updatenext == "now":
            for s in range( self.S ):
                self.maxQ[s] = max( self.Q[s] )
                self.maxA[s] = [ a for a in range( self.A[s] ) if self.Q[s][a] == self.maxQ[s] ]
                self.maxN[s] = len( self.maxA[s] )
        
        self.st = st_
        self.at = at_
        
        return self.at
   
    '''
    Q-learning
    '''
    def Q( self, st, at, rt, st_, endOfEpisode=0, update=1, updatenext="always", norm="L2" ):
        alpha   = self.alpha.getSA( st, at )
        #alpha = 1/(i**0.8+1)
        if endOfEpisode:
            gamma   = 0.0
        else:
            gamma   = self.gamma
        
        at_     = self.getAction( st_ )
        
        if update:
            V = self.maxQ[st_]
            
            self.Q[st][at] += alpha*( rt + gamma*V - self.Q[st][at] )
            if updatenext == "always":
                self.maxQ[st] = max( self.Q[st] )
                self.V[st]    = self.maxQ[st]
                self.maxA[st] = [ a for a in range( self.A[st] ) if self.Q[st][a] == self.maxQ[st] ]
                self.maxN[st] = len( self.maxA[st] )
        
        if updatenext == "now":
            for s in range( self.S ):
                self.maxQ[s] = max( self.Q[s] )
                self.V[s]    = self.maxQ[s]
                self.maxA[s] = [ a for a in range( self.A[s] ) if self.Q[s][a] == self.maxQ[s] ]
                self.maxN[s] = len( self.maxA[s] )
        
        self.st = st_
        self.at = at_
        
        return self.at
    
    
    '''
    SOR Q-learning.
    '''

    def SORQ(self, st, at, rt, st_, endOfEpisode=0, update=1, updatenext="always", norm="L2"):
        alpha = self.alpha.getSA(st, at)
        #alpha = 1/(i**0.8+1)
        if endOfEpisode:
            gamma = 0.0
        else:
            gamma = self.gamma
        #w = 1 / (1 - gamma *0.1)
        w = 10
        
        at_ = self.getAction(st_)

        if update:  # for problems except Roulette
            

            self.Q[st][at] = self.Q[st][at] + alpha * (w * (rt + gamma * self.maxQ[st_] ) + (1 - w) * self.maxQ[st] - self.Q[st][at])
            if updatenext == "always":
                self.maxQ[st] = max(self.Q[st])
                self.V[st] = self.maxQ[st]
                self.maxA[st] = [a for a in range(self.A[st]) if self.Q[st][a] == self.maxQ[st]]
                self.maxN[st] = len(self.maxA[st])

        if updatenext == "now":  # only for Roulette
            for s in range(self.S):
                self.maxQ[s] = max(self.Q[s])
                self.V[s] = self.maxQ[s]
                self.maxA[s] = [a for a in range(self.A[s]) if self.Q[s][a] == self.maxQ[s]]
                self.maxN[s] = len(self.maxA[s])

        self.st = st_
        self.at = at_

        return self.at
    
    '''
    Modelf-Free SOR Q-learning.
    '''
    def SORQw(self, st, at, rt, st_, endOfEpisode=0, update=1, i = None, updatenext="always", norm="L2"):
        alpha = self.alpha.getSA(st, at)
        #alpha = 1/(i**0.8+1)
        if endOfEpisode:
            gamma = 0.0
        else:
            gamma = self.gamma
        
        at_ = self.getAction(st_)
        
        self.tot_count[at][st][st_] +=1
        if i >=1000:
            new_w = 1/(1 - gamma)
            for s in range(self.S):
                for a in range(self.A[s]):
                    if np.sum(self.tot_count[a][s][s]) > 0:
                        temp = 1/(1 - (gamma*(self.tot_count[a][s][s]/np.sum(self.tot_count[a][s]))))
                        
                        if new_w > temp:
                            new_w = temp                  
        else:
            new_w = 1.3
            
        m = 1000/(i+1000)                
        self.w = (1 - m) * self.w + m * new_w
        
       
        if update:  # for problems except Roulette
            V = self.maxQ[st_]
            U = self.maxQ[st]

            self.Q[st][at] = self.Q[st][at] + alpha * (self.w * (rt + gamma * V ) + (1 - self.w) * U - self.Q[st][at])
            if updatenext == "always":
                self.maxQ[st] = max(self.Q[st])
                self.V[st] = self.maxQ[st]
                self.maxA[st] = [a for a in range(self.A[st]) if self.Q[st][a] == self.maxQ[st]]
                self.maxN[st] = len(self.maxA[st])

        if updatenext == "now":  # only for Roulette
            for s in range(self.S):
                self.maxQ[s] = max(self.Q[s])
                self.V[s] = self.maxQ[s]
                self.maxA[s] = [a for a in range(self.A[s]) if self.Q[s][a] == self.maxQ[s]]
                self.maxN[s] = len(self.maxA[s])

        self.st = st_
        self.at = at_

        return self.at
    
    '''
    Model Free SOR Double Q-learning.
    '''

    def SORDQw(self, st, at, rt, st_, endOfEpisode=0,  update=1,i = None, updatenext="always", norm="L2"):
        
        if endOfEpisode:
            gamma = 0.0
        else:
            gamma = self.gamma
        #w = 1 / (1 - gamma )
        
        at_ = self.getAction(st_)
        
        
        self.tot_count[at][st][st_] +=1
        
        if i >=1000:
            new_w = 1/(1 - gamma)
            for s in range(self.S):
                for a in range(self.A[s]):
                    if np.sum(self.tot_count[a][s][s]) > 0:
                        temp = 1/(1 - (gamma*(self.tot_count[a][s][s]/np.sum(self.tot_count[a][s]))))
                        
                        if new_w > temp:
                            new_w = temp                  
        else:
            new_w = 1.3
           
        m = 100/(i+100)                
        self.w = (1 - m) * self.w + m * new_w
        
        #alpha = self.alpha.getSA(st, at)
        #self.w = (1-alpha)*self.w+alpha*new_w
        
        # Append the current w value to the history list
        self.w_history.append(self.w)

        
        #self.w = 2
        
        if update:
            if random.random() < 0.5:
                U = self.maxQ[st]
                #alpha = 1/(i**0.8+1)
                alpha = self.alpha.getSA(st, at)
                self.Q[st][at] += alpha * (self.w * (rt + gamma * self.W[st_]) + (1 - self.w) * self.W[st] - self.Q[st][at])
                #self.Q[st][at] += alpha * (w * (rt + gamma * self.W[st_]) + (1 - w) * U - self.Q[st][at])

            else:
                U1 = self.maxP[st]
                beta = self.beta.getSA(st, at)
                #beta = 1/(i**0.8+1)
                self.P[st][at] += beta * (self.w * (rt + gamma * self.V[st_]) + (1 - self.w) *self.V[st] - self.P[st][at])
                # self.P[st][at] += beta * (w * (rt + gamma * self.V[st_]) + (1 - w) *U1 - self.P[st][at])

            if updatenext == "always":  # for problems except Roulette
                self.maxP[st] = max(self.P[st])
                self.maxB[st] = [b for b in range(self.A[st]) if self.P[st][b] == self.maxP[st]]

                self.maxQ[st] = max(self.Q[st])
                self.maxA[st] = [a for a in range(self.A[st]) if self.Q[st][a] == self.maxQ[st]]

                self.V[st] = sum([self.Q[st][b] for b in self.maxB[st]]) / len(self.maxB[st])
                self.W[st] = sum([self.P[st][a] for a in self.maxA[st]]) / len(self.maxA[st])

                X = [self.Q[st][a] + self.P[st][a] for a in range(self.A[st])]
                maxX = max(X)
                self.maxA[st] = [a for a in range(self.A[st]) if X[a] == maxX]
                self.maxN[st] = len(self.maxA[st])

            if updatenext == "now":  # only for Roulette
                for s in range(self.S):
                    self.maxP[s] = max(self.P[s])
                    self.maxB[s] = [b for b in range(self.A[s]) if self.P[s][b] == self.maxP[s]]

                    self.maxQ[s] = max(self.Q[s])
                    self.maxA[s] = [a for a in range(self.A[s]) if self.Q[s][a] == self.maxQ[s]]

                    self.V[s] = sum([self.Q[s][b] for b in self.maxB[s]]) / len(self.maxB[s])
                    self.W[s] = sum([self.P[s][a] for a in self.maxA[s]]) / len(self.maxA[s])

                    X = [self.Q[s][a] + self.P[s][a] for a in range(self.A[s])]
                    maxX = max(X)
                    self.maxA[s] = [a for a in range(self.A[s]) if X[a] == maxX]
                    self.maxN[s] = len(self.maxA[s])

        self.st = st_
        self.at = at_

        return self.at
    
    
    '''
    SOR Double Q-learning.
    '''

    def SORDQ(self, st, at, rt, st_, endOfEpisode=0, update=1, updatenext="always", norm="L2"):
        
        if endOfEpisode:
            gamma = 0.0
        else:
            gamma = self.gamma
        
        w = 10
        at_ = self.getAction(st_)

        if update:
            if random.random() < 0.5:
                #U = self.maxQ[st]
                alpha = self.alpha.getSA(st, at)
                #alpha = 1/(i**0.8+1)
                self.Q[st][at] += alpha * (w * (rt + gamma * self.W[st_]) + (1 - w) * self.W[st] - self.Q[st][at])
                #self.Q[st][at] += alpha * (w * (rt + gamma * self.W[st_]) + (1 - w) * U - self.Q[st][at])

            else:
                #U1 = self.maxP[st]
                beta = self.beta.getSA(st, at)
                #beta = 1/(i**0.8+1)
                self.P[st][at] += beta * (w * (rt + gamma * self.V[st_]) + (1 - w) *self.V[st] - self.P[st][at])
                # self.P[st][at] += beta * (w * (rt + gamma * self.V[st_]) + (1 - w) *U1 - self.P[st][at])

            if updatenext == "always":  # for problems except Roulette
                self.maxP[st] = max(self.P[st])
                self.maxB[st] = [b for b in range(self.A[st]) if self.P[st][b] == self.maxP[st]]

                self.maxQ[st] = max(self.Q[st])
                self.maxA[st] = [a for a in range(self.A[st]) if self.Q[st][a] == self.maxQ[st]]

                self.V[st] = sum([self.Q[st][b] for b in self.maxB[st]]) / len(self.maxB[st])
                self.W[st] = sum([self.P[st][a] for a in self.maxA[st]]) / len(self.maxA[st])

                X = [self.Q[st][a] + self.P[st][a] for a in range(self.A[st])]
                maxX = max(X)
                self.maxA[st] = [a for a in range(self.A[st]) if X[a] == maxX]
                self.maxN[st] = len(self.maxA[st])

            if updatenext == "now":  # only for Roulette
                for s in range(self.S):
                    self.maxP[s] = max(self.P[s])
                    self.maxB[s] = [b for b in range(self.A[s]) if self.P[s][b] == self.maxP[s]]

                    self.maxQ[s] = max(self.Q[s])
                    self.maxA[s] = [a for a in range(self.A[s]) if self.Q[s][a] == self.maxQ[s]]

                    self.V[s] = sum([self.Q[s][b] for b in self.maxB[s]]) / len(self.maxB[s])
                    self.W[s] = sum([self.P[s][a] for a in self.maxA[s]]) / len(self.maxA[s])

                    X = [self.Q[s][a] + self.P[s][a] for a in range(self.A[s])]
                    maxX = max(X)
                    self.maxA[s] = [a for a in range(self.A[s]) if X[a] == maxX]
                    self.maxN[s] = len(self.maxA[s])

        self.st = st_
        self.at = at_

        return self.at

    def DoubleQ( self, st, at, rt, st_, endOfEpisode=0, update=1, updatenext="always", norm="L2" ):
        
        if endOfEpisode:
            gamma   = 0.0
        else:
            gamma   = self.gamma
        
        at_     = self.getAction( st_ )
        
        if update:
            if random.random() < 0.5:
                alpha   = self.alpha.getSA( st, at )
                #alpha = 1/(i**0.8+1)
                self.Q[st][at] += alpha*( rt + gamma*self.V[st_] - self.Q[st][at] )
            
            else:
                beta    = self.beta.getSA( st, at )
                #beta = 1/(i**0.8+1)
                self.P[st][at] += beta*( rt + gamma*self.W[st_] - self.P[st][at] )
                
            if updatenext == "always":
                self.maxP[st] = max( self.P[st] )
                self.maxB[st] = [ b for b in range( self.A[st] ) if self.P[st][b] == self.maxP[st] ]
                
                self.maxQ[st] = max( self.Q[st] )
                self.maxA[st] = [ a for a in range( self.A[st] ) if self.Q[st][a] == self.maxQ[st] ]
                
                self.V[st]    = sum( [ self.Q[st][ b ] for b in self.maxB[st] ] )/len(self.maxB[st])
                self.W[st]    = sum( [ self.P[st][ a ] for a in self.maxA[st] ] )/len(self.maxA[st])
                
                X             = [ self.Q[st][a] + self.P[st][a] for a in range( self.A[st] ) ]
                maxX          = max(X)
                self.maxA[st] = [ a for a in range( self.A[st] ) if X[a] == maxX ]
                self.maxN[st] = len( self.maxA[st] )
                
            
            if updatenext == "now":
                for s in range( self.S ):
                    self.maxP[s] = max( self.P[s] )
                    self.maxB[s] = [ b for b in range( self.A[s] ) if self.P[s][b] == self.maxP[s] ]
                    
                    self.maxQ[s] = max( self.Q[s] )
                    self.maxA[s] = [ a for a in range( self.A[s] ) if self.Q[s][a] == self.maxQ[s] ]
                    
                    self.V[s]    = sum( [ self.Q[s][ b ] for b in self.maxB[s] ] )/len(self.maxB[s])
                    self.W[s]    = sum( [ self.P[s][ a ] for a in self.maxA[s] ] )/len(self.maxA[s])
                    
                    X            = [ self.Q[s][a] + self.P[s][a] for a in range( self.A[s] ) ]
                    maxX         = max(X)
                    self.maxA[s] = [ a for a in range( self.A[s] ) if X[a] == maxX ]
                    self.maxN[s] = len( self.maxA[s] )
                    
                    self.maxN[s] = len( self.maxA[s] )
        
        self.st = st_
        self.at = at_
        
        return self.at

    def Sarsa( self, st, at, rt, st_, endOfEpisode=0, update=1, updatenext="always", norm="L2" ):
        alpha   = self.alpha.getSA( st, at )
        
        if endOfEpisode:
            gamma   = 0.0
        else:
            gamma   = self.gamma

        at_     = self.getAction( st_ )

        if update:
            
            if updatenext == "always":
                V = self.Q[st_][at_]
                self.Q[st][at] += alpha*( rt + gamma*V - self.Q[st][at] )
                self.maxQ[st] = max( self.Q[st] )
                self.V[st]    = self.maxQ[st]
                self.maxA[st] = [ a for a in range( self.A[st] ) if self.Q[st][a] == self.maxQ[st] ]
                self.maxN[st] = len( self.maxA[st] )
        
        self.st = st_
        self.at = at_

        return self.at

    def ExpectedSarsa( self, st, at, rt, st_, endOfEpisode=0, update=1, updatenext="always", norm="L2" ):
        alpha   = self.alpha.getSA( st, at )
        
        if endOfEpisode:
            gamma   = 0.0
        else:
            gamma   = self.gamma
        
        at_     = self.getAction( st_ )
        
        if update:
            
            
            self.Q[st][at] += alpha*( rt + gamma*self.V[st_] - self.Q[st][at] )
            
            if updatenext == "always":
                self.maxQ[st] = max( self.Q[st] )
                
                if self.synchronous:
                    self.V[st] = sum( self.Q[st] )/self.A[st]
                else:
                    policy  = self.getPolicy( st )
                    self.V[st] = sum( [ policy[a]*self.Q[st][a] for a in range( self.A[st] ) ] )
                
                self.maxA[st] = [ a for a in range( self.A[st] ) if self.Q[st][a] == self.maxQ[st] ]
                self.maxN[st] = len( self.maxA[st] )
        
        if updatenext == "now":
            for s in range( self.S ):
                self.maxQ[s] = max( self.Q[s] )
                
                if self.synchronous:
                    self.V[st] = sum( self.Q[st] )/self.A[st]
                else:
                    policy  = self.getPolicy( st )
                    self.V[st] = sum( [ policy[a]*self.Q[st][a] for a in range( self.A[st] ) ] )
                
                self.maxA[s] = [ a for a in range( self.A[s] ) if self.Q[s][a] == self.maxQ[s] ]
                self.maxN[s] = len( self.maxA[s] )
        
        self.st = st_
        self.at = at_
        
        return self.at

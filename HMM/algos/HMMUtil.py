'''
Created on Feb 4, 2015

@author: santhosh
'''

class HMM(object):
    def __init__(self,states, start_p, trans_p, emit_p):
        self.states = states
        self.start_p = start_p
        self.trans_p = trans_p
        self.emit_p = emit_p
        
    def runNaiveAlgo(self, obs, oldState):
        stateProbList = []
        if not obs:
            return ([],1)
            
        for state in self.states:         
            nextPath,nextProb = self.runNaiveAlgo(obs[1:len(obs)], state) 
            stateProbList.append(([state]+nextPath,\
                                   self.getTransProb(state, oldState)*self.emit_p[state][obs[0]]*nextProb))
        return max(stateProbList, key = lambda x:x[1])
    
    def getTransProb(self, state, oldState):
        if not oldState:
            return self.start_p[state]
        else:
            return self.trans_p[oldState][state]
        
        
        
#     The observation made by the Viterbi algorithm is that for any state at time t,
#     there is only one most likely path to that state. Therefore, if several paths converge
#     at a particular state at time t, instead of recalculating them all when calculating the
#     transitions from this state to states at time t+1, one can discard the less likely paths,
#     and only use the most likely one in one's 
#     calculations. When this is applied to each time step, it greatly reduces the number of
#     calculations to T*N^2, which is much nicer than N^T.

    def runViterbiAlgo(self, obs):
        path = dict()
        V = []
        currDict = dict()
        for state in self.states:
            currDict[state] = self.start_p[state]*self.emit_p[state][obs[0]]
            path[state] = [state]
        
        V.append(currDict)
             
        for t in range(1,len(obs)):
            currDict = dict()
            currPath = dict()
            
            for state in self.states:
                stateProbability = [(oldState, V[t-1][oldState]*self.getTransProb(state, oldState)*self.emit_p[state][obs[t]])\
                             for oldState in self.states]
                fromState,prob = max(stateProbability, key=lambda x: x[1])
                currDict[state] = prob
                currPath[state] = path[fromState]+[state]
                
            V.append(currDict)
            path = currPath
            
        stateProbability = [(state, V[-1][state]) for state in self.states]
        (state,prob) = max(stateProbability, key=lambda x: x[1])
        return prob, path[state]
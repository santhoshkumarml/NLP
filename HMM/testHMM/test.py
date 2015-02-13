'''
Created on Feb 13, 2015

@author: santhosh
'''
import algos
from algos import HMMUtil

states = ('Healthy', 'Fever')
 
observations = ('normal', 'cold', 'dizzy')
 
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
 
transition_probability = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
   }
 
emission_probability = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
   }

hmm = HMMUtil.HMM(states, start_probability, transition_probability, emission_probability)

print hmm.runNaiveAlgo(observations, '')
print hmm.runViterbiAlgo(observations)
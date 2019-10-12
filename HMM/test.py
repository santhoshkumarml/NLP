"""
Created on Feb 13, 2015

@author: santhosh
"""

from algos.HMM import HMM

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

hmm = HMM(states, start_probability, transition_probability, emission_probability)

print(hmm.run_naive_algo(observations, ''))
print(hmm.run_viterbi_algo(observations))

states = (0, 1)

observations = (4, 3, 3, 1, 5)

start_probability = {0: 0.5, 1: 0.5}
 
transition_probability = {
   0 : {1: 0.7, 0: 0.3},
   1 : {1: 0.3, 0: 0.7}
   }
 
emission_probability = {
   0 : {1: 0.166666667, 2: 0.166666667, 3:0.166666667, 4: 0.166666667, 5: 0.166666667, 6: 0.166666667},
   1 : {1: 0.166666667, 2: 0.166666667, 3:0.166666667, 4: 0.166666667, 5: 0.166666667, 6: 0.166666667}
   }

hmm = HMM(states, start_probability, transition_probability, emission_probability)

print(hmm.run_baum_welch_algo(observations))

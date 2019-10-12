"""
Created on Feb 4, 2015

@author: santhosh
"""
import numpy


class HMM(object):
    def __init__(self, states, start_p, trans_p, emit_p):
        self.states = states
        self.start_p = start_p
        self.trans_p = trans_p
        self.emit_p = emit_p

    def run_naive_algo(self, obs, old_state):
        start_prob_list = []
        if not obs:
            return [], 1

        for state in self.states:
            next_path, next_prob = self.run_naive_algo(obs[1:len(obs)], state)
            start_prob_list.append(([state] + next_path,
                                    self.get_trans_prob(state, old_state) * self.emit_p[state][obs[0]] * next_prob))
        return max(start_prob_list, key=lambda x:x[1])

    def get_trans_prob(self, state, old_state):
        if not old_state:
            return self.start_p[state]
        else:
            return self.trans_p[old_state][state]

    # The observation made by the Viterbi algorithm is that for any state at time t,
    # there is only one most likely path to that state. Therefore, if several paths converge
    # at a particular state at time t, instead of recalculating them all when calculating the
    # transitions from this state to states at time t+1, one can discard the less likely paths,
    # and only use the most likely one in one's
    # calculations. When this is applied to each time step, it greatly reduces the number of
    # calculations to T*N^2, which is much nicer than N^T.
    def run_viterbi_algo(self, obs):
        path = dict()
        V = []
        curr_dict = dict()
        for state in self.states:
            curr_dict[state] = self.start_p[state]*self.emit_p[state][obs[0]]
            path[state] = [state]

        V.append(curr_dict)

        for t in range(1,len(obs)):
            curr_dict = dict()
            curr_path = dict()

            for state in self.states:
                start_probability = [(oldState, V[t-1][oldState] *
                                      self.get_trans_prob(state, oldState) *
                                      self.emit_p[state][obs[t]]) for oldState in self.states]
                from_state, prob = max(start_probability, key=lambda x: x[1])
                curr_dict[state] = prob
                curr_path[state] = path[from_state]+[state]

            V.append(curr_dict)
            path = curr_path

        start_probability = [(state, V[-1][state]) for state in self.states]
        (state,prob) = max(start_probability, key=lambda x: x[1])
        return prob, path[state]

    def expectation(self, obs):
        # e step
        alpha = numpy.zeros((len(obs), len(self.states)))
        beta = numpy.ones((len(obs)+1, len(self.states)))

        alpha[0] = [self.start_p[i]*self.emit_p[i][obs[0]] for i in self.states]

        for d in range(1,len(obs)):
            alpha[d] = [sum([alpha[d-1][oldState]*self.get_trans_prob(state, oldState)
                             for oldState in self.states])*self.emit_p[state][obs[d]] for state in self.states]

        for d in range(len(obs)-2,-1,-1):
            beta[d] = [sum([beta[d+1][state]*self.get_trans_prob(nextState, state)
                            for state in self.states])*self.emit_p[nextState][d+1] for nextState in self.states]

        alpha_denominator = sum([alpha[len(obs)-1][i] for i in self.states])

        epsilon = numpy.zeros((len(obs),len(self.states), len(self.states)))
        gamma = numpy.zeros((len(obs), len(self.states)))

        for d in range(1,len(obs)-1):
            for state in self.states:
                for nextState in self.states:
                    epsilon[d][state][nextState] = \
                        alpha[d-1][state] * self.get_trans_prob(nextState, state) * beta[d + 1][nextState]
                    epsilon[d][state][nextState] /= alpha_denominator

        for d in range(0,len(obs)):
            for state in self.states:
                gamma[d][state] = alpha[d][state]*self.emit_p[state][obs[d]]*beta[d][state]
                gamma[d][state] /= alpha_denominator

        return epsilon, gamma

    def get_value_for_obs_dim(self, ob1, ob2):
        if ob2 == ob1:
            return 1
        return 0

    def maximization(self, epsilon, gamma, obs):

        self.start_p =  [gamma[0][i] for i in self.states]

        expected_number_of_times_for_state = numpy.array([sum([gamma[d][i]  for d in range(len(obs))])
                                                          for i in self.states])

        for old_state in self.states:
            for state in self.states:
                self.trans_p[old_state][state] = sum([epsilon[d][old_state][state] for d in range(1,len(obs))])
                self.trans_p[old_state][state] /= expected_number_of_times_for_state[old_state]

        for state in self.states:
            for ob in self.emit_p[0].keys():
                self.emit_p[state][ob] = sum([self.get_value_for_obs_dim(obs[d], ob) * gamma[d][state] for d in range(len(obs))])
                self.emit_p[state][ob] /= expected_number_of_times_for_state[state]

    def run_baum_welch_algo(self, obs, limit=5):
        it = 0
        while it < limit:
            epsilon,gamma = self.expectation(obs)
            self.maximization(epsilon, gamma, obs)
            it += 1

        print(self.start_p)
        print(self.trans_p)
        print(self.emit_p)

        return self.run_viterbi_algo(obs)

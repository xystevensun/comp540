# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        print "calling from init:\t", iterations
        values_new = util.Counter()
        for i in range(iterations):
            for state in mdp.getStates():
                max_val = float('-inf')
                for action in mdp.getPossibleActions(state):
                    print 'calling computeQValueFromValues', i, state, action
                    val = self.computeQValueFromValues(state, action)
                    if val > max_val:
                        max_val = val
                values_new[state] = max_val
            self.values = values_new
            values_new = util.Counter()
            # print mdp.isTerminal("is terminal:\t" + str(state))
        "*** YOUR CODE HERE ***"

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        print "[computeQValueFromValues] state:\t", state, "\taction:\t", action

        prob = util.Counter()
        reward = util.Counter()
        for state_next, prob_next in self.mdp.getTransitionStatesAndProbs(state, action):
            prob[state_next] = prob_next
            reward[state_next] = self.mdp.getReward(state, action, state_next)

        value = self.values.copy()
        value.divideAll(1.0 / self.discount)

        q_val = prob * (reward + value)
        print 'prob\t', prob
        print 'reward\t', reward
        print 'value\t', value
        print 'Q\t', q_val
        return q_val

        #
        #
        # ret = 0.0
        # for state_next, prob_next in self.mdp.getTransitionStatesAndProbs(state, action):
        #     ret += prob_next * (self.mdp.getReward(state, action, state_next) + self.discount * self.values[state])
        #     # print "state next\t", state_next, "\tprob next\t", prob_next, "\tret\t", ret
        # return ret

        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        print "[computeActionFromValues] state:\t", state
        val = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            val[action] = self.computeQValueFromValues(state, action)
        print 'argMax:\t', val.argMax()
        return val.argMax()

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

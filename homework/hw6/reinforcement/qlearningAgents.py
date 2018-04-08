# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # print "-----------------------------------------constructor"
        self.qs = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        ret = self.qs[state][action] if state in self.qs else 0.0
        # print "getQValue: state\t", state, "\taction\t", action, "\tret\t", ret
        return ret
        # util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0

        q_action = util.Counter()
        for action in actions:
            q_action[action] = self.getQValue(state, action)
        # print "computeValueFromQValues(return): state\t", state, "\tmax action\t", q_action.argMax(), "\tret\t", q_action[q_action.argMax()]
        return q_action[q_action.argMax()]
        # print "computeValueFromQValues"
        # max_action = self.computeActionFromQValues(state)
        # ret = self.getQValue(state, max_action) if max_action else 0.0
        # print "computeValueFromQValues(return): state\t", state, "\tmax action\t", max_action, "\tret\t", ret
        # return ret
        # util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if not actions:
            return None

        # print "computeActionFromQValues: state\t", state, "\tactions\t", actions
        q_action = util.Counter()
        for action in actions:
            q_action[action] = self.getQValue(state, action)

        action_sort = q_action.sortedKeys()
        # print "computeActionFromQValues (return): q action\t", q_action, "\taction sort\t", action_sort
        return action_sort[0]
        # util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # print "getAction", state
        if not legalActions:
            return action
        best_action = self.computeActionFromQValues(state)
        # print best_action, legalActions
        action = random.choice(legalActions) if util.flipCoin(self.epsilon) else best_action
        # util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # print "update", state, action, nextState, reward
        if state not in self.qs:
            self.qs[state] = util.Counter()
        # print reward, action, state, self.qs[state]
        self.qs[state][action] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (
            reward + self.discount * self.computeValueFromQValues(nextState))
        # print self.qs[state][action]
        # print "update (return)"
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # print self.featExtractor.getFeatures(state, action)
        return self.weights * self.featExtractor.getFeatures(state, action)
        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        diffs = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        # print diffs, reward, self.alpha, self.discount, self.computeValueFromQValues(nextState), self.getQValue(state, action)
        if diffs != 0 and self.alpha != 0:
            weights = self.featExtractor.getFeatures(state, action).copy()
            weights.divideAll(1.0 / self.alpha / diffs)
            self.weights += weights
        pass
        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

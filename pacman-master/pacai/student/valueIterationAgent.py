from pacai.agents.learning.value import ValueEstimationAgent
import sys


class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate=0.9, iters=100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        # A dictionary which holds the q-values for each state.
        self.values = {}

        # Compute the values here.
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = self.getValue(state)
        for i in range(self.iters):
            tempValues = {}
            for state in states:
                qValue = -(sys.maxsize - 1)
                for action in self.mdp.getPossibleActions(state):
                    qValue = max(qValue, self.getQValue(state, action))
                if qValue != (-(sys.maxsize - 1)):
                    tempValues[state] = qValue
            self.values = tempValues

    def getPolicy(self, state):
        if state == 'TERMINAL_STATE':
            return None
        policy = ('', float('-inf'))
        for action in self.mdp.getPossibleActions(state):
            maxVal = max(policy[1], self.getQValue(state, action))
            if maxVal > policy[1]:
                policy = (action, maxVal)
        return policy[0]

    def getQValue(self, state, action):
        transition_fn = self.mdp.getTransitionStatesAndProbs
        reward_fn = self.mdp.getReward
        transition_probs = transition_fn(state, action)
        q_val = 0.0
        for t in transition_probs:
            s_prob, prob = t
            r = reward_fn(state, action, s_prob)
            v = self.getValue(s_prob)
            q_val += (prob * (r + (self.discountRate * v)))
        return q_val

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

import random
import sys

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [
            self.evaluationFunction(
                gameState,
                action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(
                len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best.
        chosenIndex = max(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        score = successorGameState.getScore()

        succFood = successorGameState.getFood()
        currFood = currentGameState.getFood()

        succGhostStates = successorGameState.getGhostStates()

        pacPos = successorGameState.getPacmanPosition()
        # Check if there is a ghost nearby
        for ghostState in succGhostStates:
            if ghostState.isBraveGhost():
                ghostPos = ghostState.getPosition()
                if distance.manhattan(pacPos, ghostPos) < 2:
                    return -88888  # Run away if ghost is near

        # Getting distances to food pieces
        foodDistances = []
        for food in succFood:
            distToFood = distance.manhattan(pacPos, food)
            foodDistances.append(1.0 / distToFood)
        foodDistances.sort()

        # Giving pacman a score state
        countSuccFood = succFood.count()
        countCurrFood = currFood.count()

        if countCurrFood != countSuccFood:
            if countSuccFood == 0:
                return 0
            return score + foodDistances[0]

        else:
            return foodDistances[-1] - abs(score)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        BIGNUMBER = sys.maxsize - 1
        numAgents = gameState.getNumAgents()
        treeDepth = self.getTreeDepth()
        evaluationFunc = self.getEvaluationFunction()

        def maxValue(state, currentAgent, currentDepth):
            if currentDepth == treeDepth or state.isOver():
                return evaluationFunc(state)
            # Set to negative Infinity or a really big negative number.
            value = -BIGNUMBER

            if currentAgent == 0:
                actionsList = state.getLegalActions(currentAgent)
                for action in actionsList:
                    if action == 'Stop':
                        continue
                    nextState = state.generateSuccessor(
                        currentAgent, action)  # Q value? state action pair
                    value = max(
                        value,
                        minValue(
                            nextState,
                            currentAgent + 1,
                            currentDepth))
                return value

        def minValue(state, currentAgent, currentDepth):
            value = BIGNUMBER
            if currentDepth == treeDepth or state.isOver():
                return evaluationFunc(state)

            if currentAgent == (numAgents - 1):
                actionsList = state.getLegalActions(currentAgent)
                for action in actionsList:
                    if action == 'Stop':
                        continue
                    nextState = state.generateSuccessor(
                        currentAgent, action)  # Q value? state action pair
                    value = min(
                        value, maxValue(
                            nextState, 0, currentDepth + 1))
                return value

            else:
                actionsList = state.getLegalActions(currentAgent)
                for action in actionsList:
                    nextState = state.generateSuccessor(currentAgent, action)
                    value = min(
                        value,
                        minValue(
                            nextState,
                            currentAgent + 1,
                            currentDepth))
                return value

        pacActionsList = []
        actionsList = gameState.getLegalActions()
        for action in actionsList:
            if action == 'Stop':
                continue

            nextState = gameState.generateSuccessor(
                0, action)  # State action pair?
            currentAgent = nextState.getLastAgentMoved()
            minVal = minValue(nextState, currentAgent + 1, 0)
            pacActionsList.append((action, minVal))

        bestAction = pacActionsList[0]
        for action in pacActionsList:
            if action[1] > bestAction[1]:
                bestAction = action

        return bestAction[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        BIGNUMBER = sys.maxsize - 1
        numAgents = gameState.getNumAgents()
        treeDepth = self.getTreeDepth()
        evaluationFunc = self.getEvaluationFunction()

        def maxValue(state, currentAgent, currentDepth, alpha, beta):
            if currentDepth == treeDepth or state.isOver():
                return evaluationFunc(state)
            # Set to negative Infinity or a really big negative number.
            value = -BIGNUMBER

            if currentAgent == 0:
                actionsList = state.getLegalActions(currentAgent)
                for action in actionsList:
                    if action == 'Stop':
                        continue
                    nextState = state.generateSuccessor(
                        currentAgent, action)  # Q value? state action pair
                    value = max(
                        value,
                        minValue(
                            nextState,
                            currentAgent + 1,
                            currentDepth,
                            alpha,
                            beta))

                    if value > beta:
                        return value
                    alpha = max(value, alpha)
                return value

        def minValue(state, currentAgent, currentDepth, alpha, beta):
            value = BIGNUMBER
            if currentDepth == treeDepth or state.isOver():
                return evaluationFunc(state)

            if currentAgent == (numAgents - 1):
                actionsList = state.getLegalActions(currentAgent)
                for action in actionsList:
                    if action == 'Stop':
                        continue
                    nextState = state.generateSuccessor(
                        currentAgent, action)  # Q value? state action pair
                    value = min(
                        value,
                        maxValue(
                            nextState,
                            0,
                            currentDepth + 1,
                            alpha,
                            beta))

                    if value < alpha:
                        return value
                    beta = min(value, beta)

                return value

            else:
                actionsList = state.getLegalActions(currentAgent)
                for action in actionsList:
                    nextState = state.generateSuccessor(currentAgent, action)
                    value = min(
                        value,
                        minValue(
                            nextState,
                            currentAgent + 1,
                            currentDepth,
                            alpha,
                            beta))

                    if value < alpha:
                        return value
                    beta = min(value, beta)
                return value

        pacActionsList = []
        actionsList = gameState.getLegalActions()
        for action in actionsList:
            if action == 'Stop':
                continue

            nextState = gameState.generateSuccessor(
                0, action)  # State action pair?
            currentAgent = nextState.getLastAgentMoved()
            minVal = minValue(nextState, currentAgent + 1, 0, -BIGNUMBER, BIGNUMBER)
            pacActionsList.append((action, minVal))

        bestAction = pacActionsList[0]
        for action in pacActionsList:
            if action[1] > bestAction[1]:
                bestAction = action

        return bestAction[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        BIGNUMBER = sys.maxsize - 1
        numAgents = gameState.getNumAgents()
        treeDepth = self.getTreeDepth()
        evaluationFunc = self.getEvaluationFunction()

        def expectiMax(state, currentAgent, currentDepth, chance):
            value = 0
            if currentDepth == treeDepth or state.isOver():
                return evaluationFunc(state)

            # Pacman's Move
            if currentAgent == 0 and not chance:
                value = -BIGNUMBER
                actionsList = state.getLegalActions(currentAgent)
                for action in actionsList:
                    nextState = state.generateSuccessor(currentAgent, action)
                    value = max(
                        value,
                        expectiMax(
                            nextState,
                            currentAgent + 1,
                            currentDepth,
                            True))

            # We are at a chance node
            elif chance:
                value = 0

                # If we are checking last adversary
                if (currentAgent == (numAgents - 1)):
                    actionsList = state.getLegalActions(currentAgent)
                    for action in actionsList:
                        nextState = state.generateSuccessor(
                            currentAgent, action)
                        value += (expectiMax(nextState, 0,
                                  currentDepth + 1, False))

                # Chance Node for adversary
                elif (currentAgent % numAgents != 0):
                    actionsList = state.getLegalActions(currentAgent)
                    for action in actionsList:
                        nextState = state.generateSuccessor(
                            currentAgent, action)
                        value += (expectiMax(nextState,
                                  currentAgent + 1, currentDepth, True))

                value = float(value / len(actionsList))
            return value

        pacActionsList = []
        actionsList = gameState.getLegalActions()
        for action in actionsList:
            if action == 'Stop':
                continue

            nextState = gameState.generateSuccessor(
                0, action)  # State action pair?
            currentAgent = nextState.getLastAgentMoved()
            expectiVal = expectiMax(nextState, currentAgent + 1, 0, True)
            pacActionsList.append((action, expectiVal))

        bestAction = pacActionsList[0]
        for action in pacActionsList:
            if action[1] > bestAction[1]:
                bestAction = action

        return bestAction[0]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """
    pacPos = currentGameState.getPacmanPosition()

    # Food Behavior
    foodList = currentGameState.getFood().asList()
    minDistToFood = -1
    for food in foodList:
        dist = distance.manhattan(pacPos, food)
        if minDistToFood >= dist or minDistToFood == -1:
            minDistToFood = dist

    # Ghost Behavior
    distToGhost = 1
    proximityToGhost = 0
    ghostPosList = currentGameState.getGhostPositions()
    for ghostPos in ghostPosList:
        dist = distance.manhattan(pacPos, ghostPos)
        distToGhost += dist

        if dist <= 1:
            proximityToGhost += 1

    # Num of capsules
    capsuleList = currentGameState.getCapsules()
    numCapsules = len(capsuleList)

    # Evaluation function
    eval = currentGameState.getScore() + (1 / float(minDistToFood)) - \
        (1 / float(distToGhost)) - proximityToGhost - numCapsules
    return eval

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

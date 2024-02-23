from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue


"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

# Will search based on the search algorithim. The only thing that is different is the data structure they use.
# DFS will use a stack
# BFS will use a queue


def search(problem, searchAlg):
    startingNode = problem.startingState()
    currentNode = startingNode

    nodesExplored = set()
    stateHistory = {}
    cost = 0

    prevState = ""
    action = "Stop"
    prevAction = action

    # Fringe will hold currentNode, action, cost of action, prevState, prevAction
    if searchAlg == "stack":
        fringe = Stack()
    elif searchAlg == "queue":
        fringe = Queue()
    else:
        raise Exception("Invalid search algorithim")

    fringe.push((currentNode, action, cost, prevState, prevAction))

    while not fringe.isEmpty():
        currentNode, action, cost, prevState, prevAction = fringe.pop()

        if currentNode not in nodesExplored:
            nodesExplored.add(currentNode)
            stateHistory[(currentNode, action)] = (prevState, prevAction)

            # If we reached our goal
            if problem.isGoal(currentNode):
                # Where current node is our end
                completePath = createPath(
                    startingNode, currentNode, action, stateHistory)
                return completePath

            # Else we continue
            successorStates = problem.successorStates(currentNode)
            # print("Search.py: SuccessorStates: " + str(successorStates) + "|| currentNode: " + str(currentNode))
            for state in successorStates:
                # print("Search.py: state: " + str(state))
                if state[0] not in nodesExplored:
                    # print("\n\n Search.py: " +
                    #     "\n state[0]: " + str(state[0])
                    #    + "\n state[1]: " + str(state[1])
                    #    + "\n state[2]: " + str(state[2])
                    #    + "\n currentNode: " + str(currentNode)
                    #    + "\n action: " + str(action))
                    fringe.push(
                        (state[0], state[1], state[2], currentNode, action))

    return []

# Creates a path to goal


def createPath(startNode, endNode, action, stateHistory):
    path = []
    path.append(action)
    # We can work backtrack where we came from and record each step.
    prevNode, direction = stateHistory[(endNode, action)]
    path.insert(0, direction)

    # Working backwards to retrace steps
    while (prevNode, direction) != (startNode, "Stop"):
        prevNode, direction = stateHistory[(prevNode, direction)]
        path.insert(0, direction)
    # There was an extra action at the end that can just be removed.
    path.pop(0)

    return path


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    return search(problem, "stack")


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """
    # *** Your Code Here ***
    return search(problem, "queue")


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    startingNode = problem.startingState()
    currentNode = startingNode

    nodesExplored = set()
    stateHistory = {}
    cost = 0

    prevState = ""
    action = "Stop"
    prevAction = action
    priority = 0

    fringe = PriorityQueue()
    # Where the "priority" is the priority of the node
    fringe.push((currentNode, action, cost, prevState, prevAction), priority)

    while not fringe.isEmpty():
        currentNode, action, cost, prevState, prevAction = fringe.pop()

        if currentNode not in nodesExplored:
            nodesExplored.add(currentNode)
            stateHistory[(currentNode, action)] = (prevState, prevAction)

            # If we reached our goal
            if problem.isGoal(currentNode):
                # Where current node is our end
                completePath = createPath(
                    startingNode, currentNode, action, stateHistory)
                return completePath

            # Else We continue
            successorStates = problem.successorStates(currentNode)
            for state in successorStates:
                if state[0] not in nodesExplored:
                    # Where state[2] + cost becomes the prio
                    fringe.push(
                        (state[0], state[1], state[2] + cost, currentNode, action), state[2] + cost)

    return []


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    startingNode = problem.startingState()
    currentNode = startingNode

    nodesExplored = set()
    stateHistory = {}
    cost = 0

    prevState = ""
    action = "Stop"
    prevAction = action
    priority = 0

    fringe = PriorityQueue()
    # Where the "priority" is the priority of the node
    fringe.push((currentNode, action, cost, prevState, prevAction), priority)

    while not fringe.isEmpty():
        currentNode, action, cost, prevState, prevAction = fringe.pop()

        if currentNode not in nodesExplored:
            nodesExplored.add(currentNode)
            stateHistory[(currentNode, action)] = (prevState, prevAction)

            # If we reached our goal
            if problem.isGoal(currentNode):
                # Where current node is our end
                completePath = createPath(
                    startingNode, currentNode, action, stateHistory)
                return completePath

            # Else We continue
            successorStates = problem.successorStates(currentNode)
            for state in successorStates:
                if state[0] not in nodesExplored:
                    # Where hn is our heuristic function
                    hn = heuristic(state[0], problem)
                    fringe.push(
                        (state[0], state[1], state[2] + cost, currentNode, action), state[2] + cost + hn)

    return []

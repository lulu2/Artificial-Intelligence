# search.py
# ---------
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
#Author: Jian Jin

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    from util import Stack
    from game import Directions
    visited = []
    step = []
    candidateState = Stack()
    candidateStep = Stack()
    entryState = problem.getStartState()
    candidateState.push(entryState)
    candidateStep.push('Stop')
    EdgeSet = {}
    while not candidateState.isEmpty():
        currentState = candidateState.pop()
        currentStep = candidateStep.pop()
        visited.append(currentState)
        if problem.isGoalState(currentState):
            goalState = currentState
            finalStep = currentStep
            break
        else:
            tempCandidate = []
            for candidate in problem.getSuccessors(currentState):
                if not visited.count(candidate[0]):
                    tempCandidate.append(candidate)
            if len(tempCandidate) > 0:
                for candidate in tempCandidate:
                    candidateState.push(candidate[0])
                    candidateStep.push(candidate[1])
                    EdgeSet[(candidate[0], candidate[1])] = (currentState, currentStep)
    currentState = goalState
    currentStep = finalStep
    while not currentState == entryState:
        pState = EdgeSet[(currentState, currentStep)]
        step.insert(0, currentStep)
        currentState = pState[0]
        currentStep = pState[1]
    return step

def breadthFirstSearch(problem):
    from util import Queue
    from game import Directions
    step = []
    EdgeSet = {}
    process = Queue()
    entryState = problem.getStartState()
    process.push((entryState, 'Stop'))
    EdgeSet[entryState] = (entryState, 'Stop')
    while not process.isEmpty():
        current = process.pop()
        tempCandidate = []
        if problem.isGoalState(current[0]):
            goalState = current[0]
            finalStep = current[1]
            break
        else:
            for candidate in problem.getSuccessors(current[0]):
                if not EdgeSet.has_key(candidate[0]):
                    EdgeSet[candidate[0]] = current
                    process.push((candidate[0], candidate[1]))
    currentState = goalState
    currentStep = finalStep
    while not currentState == entryState:
        pState = EdgeSet[currentState]
        step.insert(0, currentStep)
        currentState = pState[0]
        currentStep = pState[1]
    return step

def iterativeDeepening(problem):
    from util import Stack
    from game import Directions
    step = []
    EdgeSet = {}
    findGoal = 0
    process = Stack()
    entryState = problem.getStartState()
    # Each element of stack is a tuple of three items: state, action, and the current depth.
    process.push((entryState, 'Stop', 0))
    depthMaximum = 0
    EdgeSet[entryState] = (entryState, 'Stop')
    # From depthMaximum = 0, in each iteration which does not find the goal, depthMaximum will be increased by 1.
    while not findGoal:
        # Perform DFS with limited depth
        while not process.isEmpty():
            current = process.pop()
            # First check if the current node is the goal. If it is, clear the stack and finish searching.
            if problem.isGoalState(current[0]):
                goalState = current[0]
                finalStep = current[1]
                findGoal = 1
                while not process.isEmpty():
                    process.pop()
                break
            # Limit the depth of each search by checking the current depth.
            elif current[2] < depthMaximum:
                tempCandidate = []
                for candidate in problem.getSuccessors(current[0]):
                    if not EdgeSet.has_key(candidate[0]):
                        tempCandidate.append(candidate)
                if len(tempCandidate) > 0:
                    for candidate in tempCandidate:
                        # Depth of the next layer is the current depth + 1 
                        process.push((candidate[0], candidate[1], current[2] + 1))
                        EdgeSet[candidate[0]] = (current[0], current[1])
        # If no goal is found by DFS with this max depth setting, initialize the setting and increase the max depth
        if not findGoal:
            depthMaximum += 1
            EdgeSet = {}
            process.push((entryState, 'Stop', 0))
    # Construct action output
    currentState = goalState
    currentStep = finalStep
    while not currentState == entryState:
        pState = EdgeSet[currentState]
        step.insert(0, currentStep)
        currentState = pState[0]
        currentStep = pState[1]
    return step

def uniformCostSearch(problem):
    from util import PriorityQueue
    step = []
    EdgeSet = {}
    costSet = {}
    findGoal = 0
    process = PriorityQueue()
    entryState = problem.getStartState()
    process.push((entryState, 'Stop'), 0)
    costSet[entryState] = []
    EdgeSet[entryState] = (entryState, 'Stop')
    minimumCost = float("inf")
    while not process.isEmpty():
        current = process.pop()
        if problem.getCostOfActions(costSet[current[0]]) >= minimumCost:
            break
        tempCandidate = []
        if problem.isGoalState(current[0]):
            goalState = current[0]
            if minimumCost > problem.getCostOfActions(costSet[goalState]):
                minimumCost = problem.getCostOfActions(costSet[goalState])
                finalStep = current[1]
        else:
            for candidate in problem.getSuccessors(current[0]):
                if not EdgeSet[current[0]][0] == candidate[0]:
                    tempCandidate.append(candidate)
            for candidate in tempCandidate:
                if not costSet.has_key(candidate[0]):
                    costSet[candidate[0]] = costSet[current[0]] + [candidate[1]]
                    process.push((candidate[0], candidate[1]), problem.getCostOfActions(costSet[candidate[0]]))
                    EdgeSet[candidate[0]] = (current[0], current[1])
                elif problem.getCostOfActions(costSet[current[0]] + [candidate[1]]) < problem.getCostOfActions(costSet[candidate[0]]):
                    costSet[candidate[0]] = costSet[current[0]] + [candidate[1]]
                    EdgeSet[candidate[0]] = (current[0], current[1])
                    process.push((candidate[0], candidate[1]), problem.getCostOfActions(costSet[candidate[0]]))
    currentState = goalState
    currentStep = finalStep
    while not currentState == entryState:
        pState = EdgeSet[currentState]
        step.insert(0, currentStep)
        currentState = pState[0]
        currentStep = pState[1]
    return step

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    from util import PriorityQueue
    step = []
    EdgeSet = {}
    costSet = {}
    findGoal = 0
    process = PriorityQueue()
    entryState = problem.getStartState()
    process.push((entryState, 'Stop'), 0)
    costSet[entryState] = []
    EdgeSet[entryState] = (entryState, 'Stop')
    minimumCost = float("inf")
    while not process.isEmpty():
        current = process.pop()
        if problem.getCostOfActions(costSet[current[0]]) + heuristic(current[0], problem) >= minimumCost:
            break
        tempCandidate = []
        if problem.isGoalState(current[0]):
            goalState = current[0]
            if minimumCost > problem.getCostOfActions(costSet[goalState]):
                minimumCost = problem.getCostOfActions(costSet[goalState])
                finalStep = current[1]
        else:
            for candidate in problem.getSuccessors(current[0]):
                if not EdgeSet[current[0]][0] == candidate[0]:
                    tempCandidate.append(candidate)
            for candidate in tempCandidate:
                if not costSet.has_key(candidate[0]):
                    costSet[candidate[0]] = costSet[current[0]] + [candidate[1]]
                    process.push((candidate[0], candidate[1]), problem.getCostOfActions(costSet[candidate[0]]) + heuristic(candidate[0], problem))
                    EdgeSet[candidate[0]] = (current[0], current[1])
                elif problem.getCostOfActions(costSet[current[0]] + [candidate[1]]) < problem.getCostOfActions(costSet[candidate[0]]):
                    costSet[candidate[0]] = costSet[current[0]] + [candidate[1]]
                    EdgeSet[candidate[0]] = (current[0], current[1])
                    process.push((candidate[0], candidate[1]), problem.getCostOfActions(costSet[candidate[0]]) + heuristic(candidate[0], problem))
    currentState = goalState
    currentStep = finalStep
    while not currentState == entryState:
        pState = EdgeSet[currentState]
        step.insert(0, currentStep)
        currentState = pState[0]
        currentStep = pState[1]
    return step

def watchLayout(problem):
    entryState = problem.getStartState()
    k = problem.getCostOfActions(['West'])
    m = problem.getSuccessors(entryState)
    print "Entry is"
    print entryState
    print "Cost is:"
    print k
    return ['Stop']


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepening
wl = watchLayout

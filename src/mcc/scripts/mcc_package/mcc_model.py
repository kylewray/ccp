#! /usr/bin/env python

""" The MIT License (MIT)

    Copyright (c) 2017 Kyle Hollins Wray, University of Massachusetts

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
import itertools as it
import random

from fsc import *


class MCC(object):
    """ A mixed collaborative-competitive (MCC) model with group dominant rewards and slack. """

    def __init__(self):
        """ The constructor for the MCC class. """

        self.agents = ["Alice", "Bob"]

        self.gridWidth = 2
        self.gridHeight = 2

        self.state_factor = list(it.product(range(self.gridWidth), range(self.gridHeight)))
        self.states = list(it.product(self.state_factor, self.state_factor))

        self.action_factor = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        self.actions = list(it.product(self.action_factor, self.action_factor))

        self.observation_factor = [False, True]
        self.observations = list(it.product(self.observation_factor, self.observation_factor))

        self.probabilityOfSuccessfulMove = 1.0
        #self.probabilityOfSuccessfulMove = 0.7
        self.probabilityOfBumpWhenMoving = 0.0
        #self.probabilityOfBumpWhenMoving = 0.2

        self.successors = dict()
        self._compute_successors()

        self.gamma = 0.95

    def _compute_successors(self):
        """ Compute all the successors of state-action pairs. """

        self.successors = dict()

        for state in self.states:
            self.successors[state] = dict()

            for action in self.actions:
                self.successors[state][action] = list()

                for successor in self.states:
                    if self.T(state, action, successor) > 0.0:
                        self.successors[state][action] += [successor]

    def T(self, state, action, statePrime):
        """ Compute the state transition probability.

            Parameters:
                state       --  The state to start.
                action      --  The action taken in state.
                statePrime  --  A potential successor state we wish to query for the probability.

            Returns:
                The probability of successor given state-action pair.
        """

        probability = 1.0

        successors = [None for i in self.agents]
        for i in range(len(self.agents)):
            successors[i] = (state[i][0] + action[i][0],
                             state[i][1] + action[i][1])

        # State transitions are deterministic for successful movement.
        for i in range(len(self.agents)):
            # Enforce bounds of grid by self-looping at the edges.
            if (successors[i][0] < 0 or successors[i][1] < 0
                    or successors[i][0] >= self.gridWidth
                    or successors[i][1] >= self.gridHeight):
                if statePrime[i] == state[i]:
                    probability *= 1.0
                else:
                    return 0.0

                continue

            # With certainty, we always succeed in a no move action.
            # Also, with some probability, we correctly take the action (move success rate).
            # Also, with some probability, we remain still (failed to move success rate).
            # All other cases of motion are impossible.
            if action[i] == (0, 0):
                if statePrime[i] == state[i]:
                    probability *= 1.0
                else:
                    return 0.0
            else:
                if statePrime[i] == successors[i]:
                    probability *= self.probabilityOfSuccessfulMove
                elif statePrime[i] == state[i]:
                    probability *= (1.0 - self.probabilityOfSuccessfulMove)
                else:
                    return 0.0

        return probability

    def O(self, action, statePrime, observation):
        """ Compute the probability of an observation given an action-successor pair.

            Parameters:
                action      --  The action taken.
                statePrime  --  The resultant successor state.
                observation --  An observation we wish to query for the probability.

            Returns:
                The probability of the observation.
        """

        probability = 1.0

        successorsPrime = [None for i in self.agents]
        for i in range(len(self.agents)):
            successorsPrime[i] = (statePrime[i][0] + action[i][0],
                                  statePrime[i][1] + action[i][1])

        for i in range(len(self.agents)):
            # A bump can happen at the edges of the grid if the agent moves outside the grid.
            # The logic here works because 'successorsPrime' is the result of applying the action
            # to the successor state. Thus, if this location is outside the grid, it definitely
            # bumped into the wall, since the state transitions also capture this fact with certainty.
            if (successorsPrime[i][0] < 0 or successorsPrime[i][1] < 0
                    or successorsPrime[i][0] >= self.gridWidth
                    or successorsPrime[i][1] >= self.gridHeight):
                if observation[i] == True:
                    probability *= 1.0
                else:
                    return 0.0

                continue

            # A bump can also happen with low probability with other actions. If this is
            # the case, than the agent did not move...
            if action[i] == (0, 0):
                if observation[i] == False:
                    probability *= 1.0
                else:
                    return 0.0
            else:
                if observation[i] == True:
                    probability *= self.probabilityOfBumpWhenMoving
                else:
                    probability *= (1.0 - self.probabilityOfBumpWhenMoving)

        return probability

    def _agent_distance(self, state):
        """ Compute the inverse distance each agent is from one another.

            Parameters:
                state   --  The state.

            Returns:
                A inverse distance between the two agents.
        """

        return 2.0 - 1.0 * (abs(state[0][0] - state[1][0]) + abs(state[0][1] - state[1][1]))

    def R0(self, state, action):
        """ Compute the *group* reward for the state-action pair.

            Parameters:
                state   --  The state.
                action  --  The action taken in the state.

            Returns:
                The *group* reward for the state-action pair.
        """

        reward = 0.0

        objectiveStates = [(0, 1), (1, 0)]

        # Reward for being as close to each other as possible, as long as
        # *both* are not in the push box states.
        if state[0] in objectiveStates or state[1] in objectiveStates:
            reward += self._agent_distance(state)

        # Movement actions subtract a reward.
        if action[0] != (0, 0):
            reward -= 0.1
        if action[1] != (0, 0):
            reward -= 0.1

        return reward

    def Ri(self, agent, state, action):
        """ Compute the *individual* reward for the state-action pair.

            Parameters:
                agent   --  The agent name.
                state   --  The state.
                action  --  The action taken in the state.

            Returns:
                The *individual* reward for the state-action pair.
        """

        # The agents still get a reward for begin as close to each other as possible, and
        # a cost for movement. These rewards are then augmented for individual objectives.
        #reward = self.R0(state, action)

        reward = 0.0

        agentIndex = self.agents.index(agent)

        objectiveStates = [(0, 0), (1, 1)]
        objectiveActions = [(-1, 0), (1, 0)]

        # Agent i is rewarded for pushing its box, but requires the other agent to be nearby.
        if (state[agentIndex] == objectiveStates[agentIndex]
                and action[agentIndex] == objectiveActions[agentIndex]):
            reward += self._agent_distance(state)

        # Movement actions subtract a reward.
        if action[agentIndex] != (0, 0):
            reward -= 0.1

        return reward

    def get_initial_belief(self):
        """ Return the initial belief state of the MCC.

            Returns:
                The initial belief state (map for each Si for each agent i),
                to the probability, of the MCC.
        """

        return {((0, 1), (1, 0)): 1.0}

        #return {((0, int(self.gridHeight / 2)),
        #         (self.gridWidth - 1, int(self.gridHeight / 2))): 1.0}

    def get_successor(self, state, action):
        """ Return a successor state following T of the MCC.

            Parameters:
                state   --  A state (in Si for each agent i) of the MCC.
                action  --  An action (in Si for each agent i) of the MCC.

            Returns:
                A randomly selected successor state.
        """

        successor = None
        current = 0.0
        target = random.random()

        for iterSuccessor in self.successors[state][action]:
            current += self.T(state, action, iterSuccessor)
            if current >= target:
                successor = iterSuccessor
                break

        return successor

    def get_observation(self, action, successor):
        """ Return an observation following O of the MCC.

            Parameters:
                action      --  An action (in Si for each agent i) of the MCC.
                successor   --  A successor state (in Si for each agent i) of the MCC.

            Returns:
                A randomly selected observation.
        """

        observation = None
        current = 0.0
        target = random.random()

        for iterObservation in self.observations:
            current += self.O(action, successor, iterObservation)
            if current >= target:
                observation = iterObservation
                break

        return observation


if __name__ == "__main__":
    mcc = MCC()

    print(mcc.states)
    print(mcc.actions)
    print(mcc.observations)

    for i, s in enumerate(mcc.states):
        for j, a in enumerate(mcc.actions):
            for k, sp in enumerate(mcc.states):
                if mcc.T(s, a, sp) > 0.0:
                    print(s, a, sp, mcc.T(s, a, sp))

    for j, a in enumerate(mcc.actions):
        for k, sp in enumerate(mcc.states):
            for i, o in enumerate(mcc.observations):
                if mcc.O(a, sp, o) > 0.0:
                    print(a, sp, o, mcc.O(a, sp, o))

    for i, s in enumerate(mcc.states):
        for j, a in enumerate(mcc.actions):
            print(s, a, mcc.R0(s, a))

    for k, agent in enumerate(mcc.agents):
        for i, s in enumerate(mcc.states):
            for j, a in enumerate(mcc.actions):
                print(agent, s, a, mcc.Ri(agent, s, a))


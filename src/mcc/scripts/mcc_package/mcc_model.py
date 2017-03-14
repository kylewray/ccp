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

    def __init__(self, gridWidth=3, gridHeight=3, objectiveStates=list()):
        """ The constructor for the MCC class.

            Parameters:
                gridWidth           --  The width of the grid world.
                gridHeight          --  The height of the grid world.
                objectiveStates     --  The objective states (factors) list for each agent.
        """

        self.agents = ["Alice", "Bob"]

        self.gridWidth = gridWidth
        self.gridHeight = gridHeight

        self.state_factor = list(it.product(range(self.gridWidth), range(self.gridHeight)))
        self.states = list(it.product(self.state_factor, self.state_factor))

        self.action_factor = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        self.actions = list(it.product(self.action_factor, self.action_factor))

        self.observation_factor = [False]
        self.observations = list(it.product(self.observation_factor, self.observation_factor))

        self.successors = dict()
        self._compute_successors()

        # Note: Like above, this must be a list of agents to list of states (not state indexes).
        self.objectiveStates = objectiveStates

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

        # State transitions are deterministic for successful movement.
        for i in range(len(self.agents)):
            successor = (state[i][0] + action[i][0],
                         state[i][1] + action[i][1])

            # Enforce bounds of grid by self-looping at the edges.
            if (successor[0] < 0 or successor[1] < 0
                    or successor[0] >= self.gridWidth
                    or successor[1] >= self.gridHeight):
                if statePrime[i] == state[i]:
                    probability *= 1.0
                    continue
                else:
                    return 0.0

            # Enforce only movement or self-loop are possible.
            if statePrime[i] != state[i] and statePrime[i] != successor:
                return 0.0

            # TODO: Currently, this is deterministic. More complex probability calculations go here.
            if statePrime[i] == successor:
                probability *= 1.0
            else:
                probability *= 0.0

        return probability

    def O(self, action, successor, observation):
        """ Compute the probability of an observation given an action-successor pair.

            Parameters:
                action      --  The action taken.
                successor   --  The resultant successor.
                observation --  An observation we wish to query for the probability.

            Returns:
                The probability of the observation.
        """

        probability = 1.0

        # TODO: Make this uncertain maybe.

        return probability

    def R0(self, state, action):
        """ Compute the *group* reward for the state-action pair.

            Parameters:
                state   --  The state.
                action  --  The action taken in the state.

            Returns:
                The *group* reward for the state-action pair.
        """

        # Reward is 1 for no motion when all robots are in the same state.
        if len(set(state)) == 1:
            return 1.0
        return 0.0

    def Ri(self, agent, state, action):
        """ Compute the *individual* reward for the state-action pair.

            Parameters:
                agent   --  The agent name.
                state   --  The state.
                action  --  The action taken in the state.

            Returns:
                The *individual* reward for the state-action pair.
        """

        agentIndex = self.agents.index(agent)

        # Reward is 1 for no motion in a state marked with an individual objective.
        if state[agentIndex] not in self.objectiveStates[agentIndex]:
            return 0.0

        return 1.0

    def get_initial_state(self):
        """ Return the initial state of the MCC.

            Returns:
                The initial state (in Si for each agent i) of the MCC.
        """

        return ((0, int(self.gridHeight / 2)), (self.gridWidth - 1, int(self.gridHeight / 2)))

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
    # "Solve" the MCC and save the policies.
    mcc = MCC()

    for agent in mcc.agents:
        fsc = FSC(mcc, agent, 5)
        fsc.save()


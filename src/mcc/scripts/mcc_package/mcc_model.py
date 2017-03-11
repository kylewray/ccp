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

from fsc import *


class MCC(object):

    def __init__(self, agents=["Alice", "Bob"], gridWidth=3, gridHeight=3, objectiveStates=dict()):
        """ The constructor for the MCC class.

            Parameters:
                agents              --  The agents in the 'Mars Meeting' domain.
                gridWidth           --  The width of the grid world.
                gridHeight          --  The height of the grid world.
                objectiveStates     --  The objective states (0-indexed, 2-tuples) dictionary for each agent.
        """

        self.agents = agents

        self.gridWidth = gridWidth
        self.gridHeight = gridHeight

        self.states = {i: list(it.product(range(self.gridWidth), range(self.gridHeight))) for i in self.agents}
        self.actions = {i: [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] for i in self.agents}
        self.observations = {i: [False] for i in self.agents}

        # Note: Like above, this must be a dictionary mapping agents to list of states (not state indexes).
        self.objectiveStates = objectiveStates

    def T(self, state, action, statePrime):
        probability = 1.0

        # State transitions are deterministic for successful movement.
        for i in self.agents:
            successor = (state[i][0] + action[i][0],
                         state[i][1] + action[i][1])

            # Enforce bounds of grid by self-looping at the edges.
            if (newLocation[0] < 0 or newLocation[1] < 0
                    or newLocation[0] >= self.gridWidth
                    or newLocation[1] >= self.gridHeight):
                if state == statePrime:
                    return 1.0
                else:
                    return 0.0

            # Enforce only movement or self-loop are possible.
            if statePrime != state and successor != statePrime:
                return 0.0

            # TODO: Currently, this is deterministic. More complex probability calculations go here.
            if statePrime == successor:
                probability *= 1.0
            else:
                probability *= 0.0

        return probability

    def O(self, action, state, observation):
        probability = 1.0

        # TODO: Make this uncertain maybe.

        return probability

    def R0(self, state, action):
        # Reward is 1 for no motion when all robots are in the same state.
        if len(set(state.values())) == 1:
            return 1.0
        return 0.0

    def Ri(self, i, state, action):
        # Reward is 1 for no motion in a state marked with an individual objective.
        for i in self.agents:
            if state[i] not in self.objectiveStates[i]:
                return 0.0
        return 1.0


if __name__ == "__main__":
    # "Solve" the MCC and save the policies.
    mcc = MCC()

    for agent in mcc.agents:
        fsc = FSC(mcc, agent, 5)
        fsc.save()


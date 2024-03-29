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

import os
import sys

thisFilePath = os.path.dirname(os.path.realpath(__file__))

import pickle
import json
import random


from mcc_model import *


class FSC(object):
    """ A stochastic finite state controller (FSC) for agents in MCC models. """

    def __init__(self, mcc, agent, n=5):
        """ The constructor for the FSC class.

            Parameters:
                mcc     --  The MCC from which this FSC is derived.
                agent   --  The name of the agent from the MCC.
                n       --  The fixed number of FSC states.
        """

        self.mcc = mcc
        self.agent = agent
        self.n = n

        #self.Q = ["%s FSC State %i" % (self.agent, s) for s in range(self.n)]
        self.Q = ["node%i" % (s + 1) for s in range(self.n)]
        self.psi = {q: {a: 1.0 / len(mcc.action_factor) for a in mcc.action_factor} \
                    for q in self.Q}
        self.eta = {q: {a: {o: {qp: 1.0 / len(self.Q) for qp in self.Q} \
                            for o in mcc.observation_factor} \
                        for a in mcc.action_factor} \
                    for q in self.Q}

    def __str__(self):
        """ Return a string of the FSC variables. """

        result = "Name: %s\n" % (self.agent)
        result += "Num Controller Nodes: %i\n" % (self.n)
        result += "Q:\n%s\n" % (str(self.Q))
        result += "psi:\n%s\n" % (str(self.psi))
        result += "eta:\n%s\n" % (str(self.eta))

        return result

    def get_initial_state(self):
        """ Return the initial state of the FSC.

            Returns:
                The initial state (in Q) of the FSC.
        """

        return self.Q[0]

    def get_action(self, state):
        """ Return a random action following the stochastic FSC.

            Parameters:
                state   --  The FSC internal state (from Q).

            Returns:
                The randomly selected action (shared by MCC).
        """

        print("FSC Action: State %s, Action:" % (state))
        for iterAction, iterProbability in self.psi[state].items():
            if iterProbability > 0.0:
                print("\tPr(%s) = %.3f" % (iterAction, iterProbability))

        action = None
        current = 0.0
        target = random.random()

        for iterAction, iterProbability in self.psi[state].items():
            current += iterProbability
            if current >= target:
                action = iterAction
                break

        if action is None:
            action = random.choice(list(self.psi[state].keys()))

        return action

    def get_successor(self, state, action, observation):
        """ Return a random successor state following the stochastic FSC.

            Parameters:
                state       --  The FSC internal state (from Q).
                action      --  The action taken (shared by MCC).
                observation --  The observation made (shared by MCC).

            Returns:
                A randomly selected successor (from Q).
        """

        print("FSC Successor: State %s, Action %s, Observation %s, Successor:" % (state, str(action), str(observation)))
        for iterSuccessor, iterProbability in self.eta[state][action][observation].items():
            if iterProbability > 0.0:
                print("\tPr(%s) = %.3f" % (iterSuccessor, iterProbability))

        successor = None
        current = 0.0
        target = random.random()

        for iterSuccessor, iterProbability in self.eta[state][action][observation].items():
            current += iterProbability
            if current >= target:
                successor = iterSuccessor
                break

        if successor is None:
            successor = random.choice(list(self.eta[state][action][observation].keys()))

        return successor

    def save(self, filePrefix=""):
        """ Convert this to a JSON-like format and save it to the policy folder.

            Parameters:
                filePrefix  --  Optionally specify a prefix for the FSC files.
        """

        data = {'agent': self.agent,
                'n': self.n,
                'Q': self.Q,
                'psi': {q: {str(a): pr for a, pr in apr.items()} for q, apr in self.psi.items()},
                'eta': {q: {str(a): oqp for a, oqp in aoqp.items()} for q, aoqp in self.eta.items()}}

        gameType = self.mcc.gameType.lower().replace(" ", "_")

        with open(os.path.join(thisFilePath, "policies", gameType, "%s_%s.json" % (filePrefix, self.agent)), 'w') as f:
            #pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            json.dump(data, f)

    def load(self, filePrefix=""):
        """ Load the policy from the policy folder.

            Parameters:
                filePrefix  --  Optionally specify a prefix for the FSC files.
        """

        data = None

        gameType = self.mcc.gameType.lower().replace(" ", "_")

        with open(os.path.join(thisFilePath, "policies", gameType, "%s_%s.json" % (filePrefix, self.agent)), 'r') as f:
            #data = pickle.load(f)
            data = json.load(f)

        self.agent = data['agent']
        self.n = data['n']
        self.Q = data['Q']
        self.psi = {str(q): {eval(a): float(pr) for a, pr in apr.items()} for q, apr in data['psi'].items()}
        self.eta = {str(q): {eval(a): {o == "true": {str(qp): float(pr) for qp, pr in qppr.items()} for o, qppr in oqppr.items()} for a, oqppr in aoqppr.items()} for q, aoqppr in data['eta'].items()}


if __name__ == "__main__":
    if len(sys.argv) == 4:
        mcc = MCC(gameType="Battle Meeting")
        fscAlice = FSC(mcc, sys.argv[1])
        fscAlice.load("%i_%i" % (int(sys.argv[2]), int(float(sys.argv[3]))))
        print(fscAlice)
    else:
        print("Format: python3 fsc.py <agent name> <num controller nodes> <slack>")


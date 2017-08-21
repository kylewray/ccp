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

import sys

from mcc_model import *
from fsc_vector import *


class MCCSolve(object):
    """ This solves an MCC by converting it to an AMPL file. """

    def __init__(self, mcc, fscVector):
        """ The constructor for the MCCSolve class.

            Parameters:
                mcc         --  The MCC object.
                fscVector   --  The object for each agent's finite state controller.
        """

        self.mcc = mcc
        self.fscVector = fscVector

    def export_R0(self, filenameIndex):
        """ Solve the MCC from the constructor by exporting a AMPL file.

            Parameters:
                filenameIndex       --  The desired filename index for the exported AMPL file.
        """

        # TODO: If you want, in the future, you can set these too in the data file.
        #b0 = self.mcc.get_initial_belief()
        #qVector0 = self.fscVector.get_initial_state_vector()
        #qVector0Index = self.fscVector.QVector.index(qVector0)

        with open("ampl/mcc_R%i.dat" % (filenameIndex), 'w') as f:
            f.write("data;\n\n\n")

            f.write("set STATES := tl tr bl br;\n")
            f.write("set ACTIONS := none north south east west;\n")
            f.write("set OBSERVATIONS := no_bump bump;\n")
            f.write("set CONTROLLER_NODES := node1 node2 node3 node4 node5;\n\n\n")

            def convertState(stateToConvert):
                mapping = {(0, 0): "tl", (1, 0): "tr", (0, 1): "bl", (1, 1): "br"}
                return mapping[stateToConvert]

            def convertAction(actionToConvert):
                mapping = {(0, 0): "none", (0, -1): "north", (0, 1): "south", (1, 0): "east", (-1, 0): "west"}
                return mapping[actionToConvert]

            def convertObservation(observationToConvert):
                mapping = {False: "no_bump", True: "bump"}
                return mapping[observationToConvert]

            for s in self.mcc.states:
                for a in self.mcc.actions:
                    for sp in self.mcc.states:
                        if self.mcc.T(s, a, sp) > 0.0:
                            f.write("let T[\"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\"] := %.3f;\n" % (convertState(s[0]), convertState(s[1]), convertAction(a[0]), convertAction(a[1]), convertState(sp[0]), convertState(sp[1]), self.mcc.T(s, a, sp)))

            f.write("\n\n")
            for a in self.mcc.actions:
                for sp in self.mcc.states:
                    for o in self.mcc.observations:
                        if self.mcc.O(a, sp, o) > 0.0:
                            f.write("let O[\"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\"] := %.3f;\n" % (convertAction(a[0]), convertAction(a[1]), convertState(sp[0]), convertState(sp[1]), convertObservation(o[0]), convertObservation(o[1]), self.mcc.O(a, sp, o)))

            f.write("\n\n")
            for s in self.mcc.states:
                for a in self.mcc.actions:
                    if self.mcc.R0(s, a) != 0.0:
                        f.write("let R0[\"%s\", \"%s\", \"%s\", \"%s\"] := %.3f;\n" % (convertState(s[0]), convertState(s[1]), convertAction(a[0]), convertAction(a[1]), self.mcc.R0(s, a)))

            for i, agent in enumerate(self.mcc.agents):
                f.write("\n\n")

                for s in self.mcc.states:
                    for a in self.mcc.actions:
                        if self.mcc.Ri(agent, s, a) != 0.0:
                            f.write("let R%i[\"%s\", \"%s\", \"%s\", \"%s\"] := %.3f;\n" % (i + 1, convertState(s[0]), convertState(s[1]), convertAction(a[0]), convertAction(a[1]), self.mcc.Ri(agent, s, a)))

            for i, agent in enumerate(self.mcc.agents):
                f.write("\n\n")

                for k, a in enumerate(self.mcc.action_factor):
                    f.write("let psi%i[\"node%i\", \"%s\"] := %.3f;\n" % (i + 1, k + 1, convertAction(a), 1.0))

    def export_Ri(self, agent, delta=5.0):
        """ Solve the MCC from the constructor by exporting a AMPL file.

            Parameters:
                delta   --  The slack variable.
        """

        agentIndex = self.mcc.agents.index(agent)
        otherAgentIndex = abs(agentIndex - 1)

        # First, we read the output text file to get the necessary information to append about V0Star.
        output = open("ampl/output_R0.txt", 'r').read().split(" iterations, objective ")[1].split("\n")
        V0Star = float(output[0])

        # Now we do the same thing but on the other output text file to get the psi_{-i} and eta_{-i} best responses.
        output = open("ampl/output_Ri_best_response.txt", 'r').read().split(" iterations, objective ")[1].split("\n")
        psiList = list()
        etaList = list()

        counter = 3
        line = ""

        while counter < 100000:
            line = list(filter(lambda x: x != "", output[counter].split(" ")))
            if line[0] == ";":
                break

            param = line[1][1:-1]
            value = max(0.0, min(1.0, float(line[2])))

            if param[0:4] == "psi%i" % (otherAgentIndex + 1):
                psiList += ["let %s := %.5f;\n" % (param, value)]
            elif param[0:4] == "eta%i" % (otherAgentIndex + 1):
                etaList += ["let %s := %.5f;\n" % (param, value)]

            counter += 1

        # Now, we export the Ri file, using R0 as a base.
        self.export_R0(agentIndex + 1)

        with open("ampl/mcc_R%i.dat" % (agentIndex + 1), 'a') as f:
            f.write("\n\n")
            f.write("let delta := %.5f;\n" % (delta))
            f.write("let V0Star := %.5f;\n" % (V0Star))

            f.write("\n\n")
            for psi in psiList:
                f.write(psi)

            f.write("\n\n")
            for eta in etaList:
                f.write(eta)


if __name__ == "__main__":
    mcc = MCC()
    fscVector = FSCVector([FSC(mcc, agent, 5) for agent in mcc.agents])
    solve = MCCSolve(mcc, fscVector)

    if len(sys.argv) == 1:
        solve.export_R0(0)
    elif len(sys.argv) == 3:
        agentIndex = int(sys.argv[1]) - 1
        delta = float(sys.argv[2])
        solve.export_Ri(mcc.agents[agentIndex], delta)



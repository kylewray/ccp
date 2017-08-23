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
import ast

thisFilePath = os.path.dirname(os.path.realpath(__file__))

import itertools as it


class FSCVector(object):
    """ An object to hold an FSC vector (one FSC for each agent). """

    def __init__(self, fscs):
        """ The constructor for the FSC vector object.

            Parameters:
                fscs    --  The vector of FSC (one FSC for each agent).
        """

        self.fscs = fscs
        self.QVector = list()

        self._compute_Q_vector()

    def _compute_Q_vector(self):
        """ Compute the QVector variable after updating the FSC vector. """

        self.QVector = list(it.product([fsc.Q for fsc in self.fscs]))

    def get_initial_state_vector(self):
        """ Get the initial state vector for all Q values.

            Returns:
                The vector of initial Q-values.
        """

        return self.QVector[0]

    def save(self, numControllerNodes, delta):
        """ Save each of the FSCs to a file.

            Parameters:
                numControllerNodes  --  The number of controller nodes for each agent.
                delta               --  The non-negative slack value.
        """

        for fsc in self.fscs:
            fsc.save("%i_%i" % (numControllerNodes, int(delta)))

    def load_from_output_file(self, mcc, fscIndex, filename):
        """ Load the FSC from this NEOS output file given the filename.

            Parameters:
                mcc         --  The MCC to get the actions and observations for exporting.
                fscIndex    --  The FSC index in the vector.
                filename    --  The name of the file.
        """

        convertAction = {"none": (0, 0), "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
        convertObservation = {"no_bump": False, "bump": True}

        output = open(filename, 'r').read().split(" iterations, objective ")[1].split("\n")

        counter = 3
        line = ""

        while counter < len(output):
            line = list(filter(lambda x: x != "", output[counter].split(" ")))
            if line[0] == ";":
                break

            param = line[1][1:-1]
            value = max(0.0, min(1.0, float(line[2])))

            if param[0:3] == "psi":
                psi = ast.literal_eval(param[4:])
                q = next(q for q in self.fscs[fscIndex].Q if q == psi[0])
                a = next(a for a in mcc.action_factor if a == convertAction[psi[1]])
                self.fscs[fscIndex].psi[q][a] = value
            elif param[0:3] == "eta":
                eta = ast.literal_eval(param[4:])
                q = next(q for q in self.fscs[fscIndex].Q if q == eta[0])
                a = next(a for a in mcc.action_factor if a == convertAction[eta[1]])
                o = next(o for o in mcc.observation_factor if o == convertObservation[eta[2]])
                qp = next(qp for qp in self.fscs[fscIndex].Q if qp == eta[3])
                self.fscs[fscIndex].eta[q][a][o][qp] = value

            counter += 1

    def load_from_data_file(self, mcc, fscIndex, filename):
        """ Load the FSC from this NEOS data file given the filename.

            Parameters:
                mcc         --  The MCC to get the actions and observations for exporting.
                fscIndex    --  The FSC index in the vector.
                filename    --  The name of the file.
        """

        convertAction = {"none": (0, 0), "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
        convertObservation = {"no_bump": False, "bump": True}

        output = open(filename, 'r').read().split("\n")

        counter = 0
        line = ""

        while counter < len(output):
            if len(output[counter]) < 4 or output[counter][0:3] != "let":
                counter += 1
                continue

            line = list(filter(lambda x: x != "", output[counter].split(" ")))

            param = line[1]

            value = 0.0
            if param[0:3] == "psi" or param[0:3] == "eta":
                value = max(0.0, min(1.0, float(line[3][:-1])))

            if param[0:3] == "psi":
                psi = ast.literal_eval(param[4:])
                q = next(q for q in self.fscs[fscIndex].Q if q == psi[0])
                a = next(a for a in mcc.action_factor if a == convertAction[psi[1]])
                self.fscs[fscIndex].psi[q][a] = value
            elif param[0:3] == "eta":
                eta = ast.literal_eval(param[4:])
                q = next(q for q in self.fscs[fscIndex].Q if q == eta[0])
                a = next(a for a in mcc.action_factor if a == convertAction[eta[1]])
                o = next(o for o in mcc.observation_factor if o == convertObservation[eta[2]])
                qp = next(qp for qp in self.fscs[fscIndex].Q if qp == eta[3])
                self.fscs[fscIndex].eta[q][a][o][qp] = value

            counter += 1



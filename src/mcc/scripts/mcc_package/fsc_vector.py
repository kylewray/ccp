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

import itertools as it


class FSCVector(object):
    """ An object to hold an FSC vector (one FSC for each agent). """

    def __init__(self, fscVector):
        """ The constructor for the FSC vector object.

            Parameters:
                fscVector   --  The vector of FSC (one FSC for each agent).
        """

        self.fscVector = fscVector
        self.QVector = list()

        self._compute_Q_vector()

    def _compute_Q_vector(self):
        """ Compute the QVector variable after updating the FSC vector. """

        self.QVector = list(it.product([fsc.Q for fsc in self.fscVector]))

    def get_initial_state_vector(self):
        """ Get the initial state vector for all Q values.

            Returns:
                The vector of initial Q-values.
        """

        return self.QVector[0]


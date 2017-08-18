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

from mcc_model import *
from fsc import *

import pylab
import numpy as np


class Experiments(object):
    """ Run experiments: solve MCC, save policies, average reward/value, and graph result. """

    def __init__(self):
        """ The constructor for the Experiments class. """

        self.numTrials = 10
        self.horizon = 30

        self.slackValues = [0.0, 1.0, 2.0, 3.0]

    def start(self):
        """ Start the experiments and graph the result at the end. """

        values = [list(), list(), list()]
        standardError = [list(), list(), list()]

        # For each slack term, create the MCC, solve it, and compute the value.
        for slack in self.slackValues:
            # Create the MCC and solve it.
            mcc = MCC(objectiveStates=[[(0, 0)], [(2, 2)]])

            # TODO: Solve...
            aliceFSC = FSC(mcc, "Alice")
            bobFSC = FSC(mcc, "Bob")

            # Compute the average value following this FSC policy.
            data = [list(), list(), list()]
            averages = np.array([0.0, 0.0, 0.0])

            for i in range(self.numTrials):
                state = mcc.get_initial_state()
                aliceState = aliceFSC.get_initial_state()
                bobState = bobFSC.get_initial_state()

                trialValues = [0.0, 0.0, 0.0]

                for t in range(self.horizon):
                    action = (aliceFSC.get_action(aliceState), bobFSC.get_action(bobState))

                    trialValues[0] += mcc.R0(state, action)
                    trialValues[1] += mcc.Ri("Alice", state, action)
                    trialValues[2] += mcc.Ri("Bob", state, action)

                    successor = mcc.get_successor(state, action)
                    observation = mcc.get_observation(action, successor)

                    state = successor
                    aliceState = aliceFSC.get_successor(aliceState, action[0], observation[0])
                    bobState = bobFSC.get_successor(bobState, action[1], observation[1])

                for j in range(len(averages)):
                    data[j] += [trialValues[j]]
                    averages[j] = float(i * averages[j] + trialValues[j]) / float(i + 1.0)

            # Record the value and compute standard error.
            for i in range(len(values)):
                values[i] += [averages[i]]
                standardError[i] += [math.sqrt(sum([pow(data[i][j] - averages[i], 2) for j in range(len(data[i]))]) / float(len(data[i]) - 1.0))]

        # Compute some final things and make adjustments.
        for i in range(len(values)):
            values[i] = np.array(values[i])

        minV = min([min(v) for v in values])
        maxV = max([max(v) for v in values])

        # Plot the result, providing beautiful paper-worthy labels.
        labels = ["V0", "V1", "V2"]
        linestyles = ["-", "--", ":"]
        markers = ["o", "s", "^"]
        colors = ["r", "g", "b"]

        minSlack = min(self.slackValues)
        maxSlack = max(self.slackValues)

        pylab.title("Average Discounted Reward vs. Slack")
        pylab.hold(True)

        pylab.xlabel("Slack")
        pylab.xticks(np.arange(minSlack, maxSlack + 1.0, 1.0))
        pylab.xlim([minSlack - 0.1, maxSlack + 0.1])

        pylab.ylabel("Average Discounted Reward")
        pylab.yticks(np.arange(-0.5, int(maxV) + 1.5, 0.5))
        pylab.ylim([-0.1, int(maxV) + 1.1])

        pylab.hlines(np.arange(int(minV) - 1.0, int(maxV) + 1.0, 1.0), minSlack - 1.0, maxSlack + 1.0, colors=[(0.7, 0.7, 0.7)])

        for i in range(len(values)):
            pylab.errorbar(self.slackValues, values[i],
                        yerr=standardError[i],
                        linestyle=linestyles[i], linewidth=1,
                        marker=markers[i], markersize=14,
                        color=colors[i])
            pylab.plot(self.slackValues, values[i],
                        label=labels[i],
                        linestyle=linestyles[i], linewidth=4,
                        marker=markers[i], markersize=14,
                        color=colors[i])

        pylab.legend(loc=4)
        pylab.show()


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

import pylab
import numpy as np

from mcc_model import *
from mcc_solve import *
from fsc import *
from fsc_vector import *


class Experiments(object):
    """ Run experiments: solve MCC, save policies, average reward/value, and graph result. """

    def __init__(self):
        """ The constructor for the Experiments class. """

        self.numTrials = 10000
        self.horizon = 100

        self.numSteps = 10

        self.numControllerNodes = [2, 4, 6]
        self.slackValues = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]

    def _compute_final_values(self):
        """ Load the 'ampl/mcc_compute_final.output' file and compute the final values.

            Returns:
                A 3-vector of the final values for V0, V1, and V2.
        """

        result = [None, None, None]

        filename = "ampl/mcc_compute_final.output"
        output = open(filename, 'r').read().split(" iterations, objective ")[1].split("\n")

        counter = 3
        line = ""

        while counter < len(output):
            line = list(filter(lambda x: x != "", output[counter].split(" ")))
            if line[0] == ";":
                break

            param = line[1][1:-1]

            if param[0] == "V" and str.startswith(param[2:], "['node1','node1','bl','tr']"):
                objectiveIndex = int(param[1])
                objectiveValue = float(line[2])
                result[objectiveIndex] = objectiveValue

            counter += 1

        return result

    def start(self, solve=True, username="", password=""):
        """ Start the experiments and graph the result at the end.

            Parameters:
                solve       --  True if we should solve the policies here. False if we should instead load them.
                username    --  The NEOS server username.
                password    --  The NEOS server password.
        """

        for numNodes in self.numControllerNodes:
            values = [list(), list(), list()]
            standardError = [list(), list(), list()]

            computeFinalValues = [list(), list(), list()]

            # For each slack term, create the MCC, solve it, and compute the value.
            for slack in self.slackValues:
                print("----- Starting Configuration [Num Nodes: %i, Slack %.1f] -----" % (numNodes, slack))

                # Create the MCC, FSCs, etc. Then solve the MCC. Then load the policies.
                mcc = MCC()

                aliceFSC = FSC(mcc, "Alice", numNodes)
                bobFSC = FSC(mcc, "Bob", numNodes)
                fscVector = FSCVector([aliceFSC, bobFSC])

                if solve:
                    # Note: This overwrites the FSCs and re-saves them for later use, if you want.
                    mccSolve = MCCSolve(mcc, fscVector, numSteps=self.numSteps, delta=slack)
                    totalTime, individualTimes = mccSolve.solve(username, password, resolve=(slack == 0.0))

                    print("Individual Times: [R0: %.2fs, R1: %.2fs, R2: %.2fs]" % (individualTimes[0], individualTimes[1], individualTimes[2]))
                    print("Total Time: %.2f seconds" % (totalTime))
                    print("")

                    # We can also use the output files to compute the actual values and make another graph!
                    computeFinalValuesResult = self._compute_final_values()
                    computeFinalValues[0] += [computeFinalValuesResult[0]]
                    computeFinalValues[1] += [computeFinalValuesResult[1]]
                    computeFinalValues[2] += [computeFinalValuesResult[2]]
                else:
                    aliceFSC.load("%i_%i" % (numNodes, int(slack)))
                    bobFSC.load("%i_%i" % (numNodes, int(slack)))

                # Compute the average value following this FSC policy.
                data = [list(), list(), list()]
                averages = np.array([0.0, 0.0, 0.0])

                for i in range(self.numTrials):
                    belief = mcc.get_initial_belief()
                    state = None

                    currentValue = 0.0
                    targetValue = random.random()
                    for s in mcc.states:
                        try:
                            currentValue += belief[s]
                            if currentValue >= targetValue:
                                state = s
                                break
                        except:
                            continue

                    aliceState = aliceFSC.get_initial_state()
                    bobState = bobFSC.get_initial_state()

                    trialValues = [0.0, 0.0, 0.0]
                    compoundedGamma = 1.0

                    for t in range(self.horizon):
                        action = (aliceFSC.get_action(aliceState), bobFSC.get_action(bobState))

                        trialValues[0] += compoundedGamma * mcc.R0(state, action)
                        trialValues[1] += compoundedGamma * mcc.Ri("Alice", state, action)
                        trialValues[2] += compoundedGamma * mcc.Ri("Bob", state, action)
                        compoundedGamma *= mcc.gamma

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

            if solve:
                for i in range(len(computeFinalValues)):
                    computeFinalValues[i] = np.array(computeFinalValues[i])

            minV = min([min(v) for v in values])
            maxV = max([max(v) for v in values])

            # Plot the result, providing beautiful paper-worthy labels.
            labels = ["V0", "V1", "V2"]
            linestyles = ["-", "--", ":"]
            markers = ["o", "s", "^"]
            colors = ["r", "g", "b"]

            minSlack = min(self.slackValues)
            maxSlack = max(self.slackValues)

            pylab.title("Average Discounted Reward vs. Slack (Num Nodes = %i)" % (numNodes))
            pylab.hold(True)

            pylab.xlabel("Slack")
            pylab.xticks(np.arange(minSlack, maxSlack + 1.0, 5.0))
            pylab.xlim([minSlack - 0.1, maxSlack + 0.1])

            pylab.ylabel("Average Discounted Reward")
            pylab.yticks(np.arange(-1.0, int(maxV) + 6, 5))
            #pylab.ylim([-0.1, int(maxV) + 1.1])

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

            # Special: If we just solved for these, then we have the actual values! Plot these results too!
            if solve:
                labels = ["V0", "V1", "V2"]
                linestyles = ["-", "--", ":"]
                markers = ["o", "s", "^"]
                colors = ["r", "g", "b"]

                minSlack = min(self.slackValues)
                maxSlack = max(self.slackValues)

                pylab.title("Computed Values vs. Slack (Num Nodes = %i)" % (numNodes))
                pylab.hold(True)

                pylab.xlabel("Slack")
                pylab.xticks(np.arange(minSlack, maxSlack + 1.0, 5.0))
                pylab.xlim([minSlack - 0.1, maxSlack + 0.1])

                pylab.ylabel("Computed Values")
                pylab.yticks(np.arange(-1.0, int(maxV) + 6, 5))
                #pylab.ylim([-0.1, int(maxV) + 1.1])

                pylab.hlines(np.arange(int(minV) - 1.0, int(maxV) + 1.0, 1.0), minSlack - 1.0, maxSlack + 1.0, colors=[(0.7, 0.7, 0.7)])

                for i in range(len(computeFinalValues)):
                    pylab.plot(self.slackValues, computeFinalValues[i],
                                label=labels[i],
                                linestyle=linestyles[i], linewidth=4,
                                marker=markers[i], markersize=14,
                                color=colors[i])

                pylab.legend(loc=4)
                pylab.show()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        experiments = Experiments()
        experiments.start(solve=True, username=sys.argv[1], password=sys.argv[2])
    else:
        print("Format: python3 experiments.py <username> <password>")



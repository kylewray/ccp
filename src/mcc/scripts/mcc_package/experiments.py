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

        self.numTrials = 1000
        self.horizon = 100

        self.maxNumSteps = 50

        self.numControllerNodes = [2, 4, 6]
        self.slackValues = [0.0, 5.0, 10.0, 15.0, 20.0]

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

    def start(self, solve=True, username="", password="", gameType="Prisoner Meeting"):
        """ Start the experiments and graph the result at the end.

            Parameters:
                solve       --  True if we should solve the policies here. False if we should instead load them.
                username    --  The NEOS server username.
                password    --  The NEOS server password.
                gameType    --  The type of game: "Prisoner Meeting" or "Battle Meeting".
        """

        for numNodes in self.numControllerNodes:
            values = [list(), list(), list()]
            standardError = [list(), list(), list()]

            computeFinalValues = [list(), list(), list()]

            # For each slack term, create the MCC, solve it, and compute the value.
            for slack in self.slackValues:
                print("----- Starting Configuration [Num Nodes: %i, Slack %.1f] -----" % (numNodes, slack))

                # Create the MCC, FSCs, etc. Then solve the MCC. Then load the policies.
                mcc = MCC(gameType)

                aliceFSC = FSC(mcc, "Alice", numNodes)
                bobFSC = FSC(mcc, "Bob", numNodes)
                fscVector = FSCVector([aliceFSC, bobFSC])

                if solve:
                    # Note: This overwrites the FSCs and re-saves them for later use, if you want.
                    mccSolve = MCCSolve(mcc, fscVector, maxNumSteps=self.maxNumSteps, delta=slack)
                    totalTime, individualTimes = mccSolve.solve(username, password, resolve=(slack == self.slackValues[0]))

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
            labels = ["V0", "V1", "V2", "Trend", "(V1+V2)/2"]
            linestyles = ["-", "--", ":", "-", "-"]
            markers = ["o", "s", "^", "", ""]
            colors = ["r", "g", "b", "k", "k"]

            minSlack = min(self.slackValues)
            maxSlack = max(self.slackValues)

            pylab.title("%s: ADR vs. Slack (Num Nodes = %i)" % (gameType, numNodes))
            pylab.hold(True)

            pylab.xlabel("Slack")
            pylab.xticks(np.arange(minSlack, maxSlack + 5.0, 5.0))
            pylab.xlim([minSlack - 0.1, maxSlack + 0.1])

            pylab.ylabel("Average Discounted Reward")
            pylab.yticks(np.arange(int(minV), int(maxV) + 5.0, 5.0))
            pylab.ylim([minV - 0.1, int(maxV) + 1.1])

            pylab.hlines(np.arange(int(minV) - 1.0, int(maxV) + 1.0, 5.0), minSlack - 1.0, maxSlack + 1.0, colors=[(0.7, 0.7, 0.7)])

            for i in range(len(values)):
                pylab.errorbar(self.slackValues, values[i],
                               yerr=standardError[i],
                               linestyle=linestyles[i], linewidth=3,
                               marker=markers[i], markersize=18,
                               color=colors[i])
                pylab.plot(self.slackValues, values[i],
                           label=labels[i],
                           linestyle=linestyles[i], linewidth=8,
                           marker=markers[i], markersize=18,
                           color=colors[i])

            # Special: Print a trend line for the individual objectives.
            trendLineZ = pylab.polyfit(self.slackValues + self.slackValues, pylab.concatenate((values[1], values[2]), axis=0), 1)
            trendLinePoly = pylab.poly1d(trendLineZ)
            trendLineValues = [trendLinePoly(slackValue) for slackValue in self.slackValues]
            pylab.plot(self.slackValues, trendLineValues,
                       label=labels[3],
                       linestyle=linestyles[3], linewidth=8,
                       marker=markers[3], markersize=18,
                       color=colors[3])

            # Special: Print the average of the individual objectives.
            #pylab.plot(self.slackValues, [(values[1][i] + values[2][i]) / 2.0 for i in range(len(self.slackValues))],
            #           label=labels[4],
            #           linestyle=linestyles[4], linewidth=8,
            #           marker=markers[4], markersize=18,
            #           color=colors[4])

            if gameType == "Prisoner Meeting":
                pylab.legend(loc=3) # Lower Left
            elif gameType == "Battle Meeting":
                pylab.legend(loc=1) # Upper Right
            pylab.show()

            # Special: If we just solved for these, then we have the actual values! Plot these results too!
            if solve:
                labels = ["V0", "V1", "V2"]
                linestyles = ["-", "--", ":"]
                markers = ["o", "s", "^"]
                colors = ["r", "g", "b"]

                minSlack = min(self.slackValues)
                maxSlack = max(self.slackValues)

                pylab.title("%s: Computed Values vs. Slack (Num Nodes = %i)" % (gameType, numNodes))
                pylab.hold(True)

                pylab.xlabel("Slack")
                pylab.xticks(np.arange(minSlack, maxSlack + 5.0, 5.0))
                pylab.xlim([minSlack - 0.1, maxSlack + 0.1])

                pylab.ylabel("Computed Values")
                pylab.yticks(np.arange(int(minV), int(maxV) + 5.0, 5.0))
                pylab.ylim([minV - 0.1, int(maxV) + 1.1])

                pylab.hlines(np.arange(int(minV) - 1.0, int(maxV) + 1.0, 5.0), minSlack - 1.0, maxSlack + 1.0, colors=[(0.7, 0.7, 0.7)])

                for i in range(len(computeFinalValues)):
                    pylab.plot(self.slackValues, computeFinalValues[i],
                                label=labels[i],
                                linestyle=linestyles[i], linewidth=8,
                                marker=markers[i], markersize=18,
                                color=colors[i])

                if gameType == "Prisoner Meeting":
                    pylab.legend(loc=3) # Lower Left
                elif gameType == "Battle Meeting":
                    pylab.legend(loc=1) # Upper Right
                pylab.show()

if __name__ == "__main__":
    error = False

    if len(sys.argv) >= 2:
        if sys.argv[1] == "1":
            gt = "Prisoner Meeting"
        elif sys.argv[1] == "2":
            gt = "Battle Meeting"
        else:
            error = True
    else:
        error = True

    if not error:
        experiments = Experiments()
        if len(sys.argv) == 2:
            experiments.start(solve=False, gameType=gt)
        elif len(sys.argv) == 4:
            experiments.start(solve=True, username=sys.argv[2], password=sys.argv[3], gameType=gt)
        else:
            error = True

    if error:
        print("Format: python3 experiments.py <game type number: {1=Prisoner Meeting, 2=Battle Meeting}> <username (optional)> <password (optional)>")



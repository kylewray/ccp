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
import time
import timeit
import os.path

import xmlrpc.server
import xmlrpc.client

from mcc_model import *
from fsc_vector import *


class NonlinearInfeasibilitiesError(Exception):
    """ For when NEOS server's SNOPT returns that it had to approximate and relax nonlinear constraints to return a solution. """

    pass


class MCCSolve(object):
    """ This solves an MCC by converting it to an AMPL file. """

    def __init__(self, mcc, fscVector, maxNumSteps=3, delta=0.0):
        """ The constructor for the MCCSolve class.

            Parameters:
                mcc         --  The MCC object.
                fscVector   --  The object for each agent's finite state controller.
                maxNumSteps    --  The number of best response iterations.
                delta       --  The slack variable, non-negative.
        """

        self.mcc = mcc
        self.fscVector = fscVector
        self.numControllerNodes = max([fsc.n for fsc in fscVector.fscs])
        self.delta = delta
        self.maxNumSteps = maxNumSteps
        self.epsilon = 0.01

        # This handles the rounding errors when reading the output text file.
        # Basically, without this, the approximation of solving NLPs tends to 
        # produce "nonlinear infeasibilities" that get relaxed away, and it
        # causes the optimal value to be far less, since the slack constraint
        # is the easiest it relaxes. The price of using approximate NLP solvers.
        self.allowableErrorForNLPSolversV0Star = 0.25
        self.checkForNonlinearInfeasibilities = True

        self.neos = None

    def _export_R0(self, filenameIndex=0, filename=None):
        """ Solve the MCC from the constructor by exporting a AMPL file.

            Parameters:
                filenameIndex       --  The desired filename index for the exported AMPL file.
                filename            --  Instead, optionally, the user can just set the filename.
        """

        # TODO: If you want, in the future, you can set these too in the data file.
        #b0 = self.mcc.get_initial_belief()
        #qVector0 = self.fscVector.get_initial_state_vector()
        #qVector0Index = self.fscVector.QVector.index(qVector0)

        if filename is None:
            filename = "ampl/mcc_R%i.dat" % (filenameIndex)

        with open(filename, 'w') as f:
            f.write("data;\n\n\n")

            controllerNodes = " ".join(["node%i" % (i + 1) for i in range(self.numControllerNodes)])

            f.write("set STATES := tl tr bl br;\n")
            f.write("set ACTIONS := none north south east west;\n")
            f.write("set OBSERVATIONS := no_bump bump;\n")
            f.write("set CONTROLLER_NODES := %s;\n\n\n" % (controllerNodes))

            convertState = {(0, 0): "tl", (1, 0): "tr", (0, 1): "bl", (1, 1): "br"}
            convertAction = {(0, 0): "none", (0, -1): "north", (0, 1): "south", (1, 0): "east", (-1, 0): "west"}
            convertObservation = {False: "no_bump", True: "bump"}

            for s in self.mcc.states:
                for a in self.mcc.actions:
                    for sp in self.mcc.states:
                        if self.mcc.T(s, a, sp) > 0.0:
                            f.write("let T[\"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\"] := %.3f;\n" % (convertState[s[0]], convertState[s[1]], convertAction[a[0]], convertAction[a[1]], convertState[sp[0]], convertState[sp[1]], self.mcc.T(s, a, sp)))

            f.write("\n\n")
            for a in self.mcc.actions:
                for sp in self.mcc.states:
                    for o in self.mcc.observations:
                        if self.mcc.O(a, sp, o) > 0.0:
                            f.write("let O[\"%s\", \"%s\", \"%s\", \"%s\", \"%s\", \"%s\"] := %.3f;\n" % (convertAction[a[0]], convertAction[a[1]], convertState[sp[0]], convertState[sp[1]], convertObservation[o[0]], convertObservation[o[1]], self.mcc.O(a, sp, o)))

            f.write("\n\n")
            for s in self.mcc.states:
                for a in self.mcc.actions:
                    if self.mcc.R0(s, a) != 0.0:
                        f.write("let R0[\"%s\", \"%s\", \"%s\", \"%s\"] := %.3f;\n" % (convertState[s[0]], convertState[s[1]], convertAction[a[0]], convertAction[a[1]], self.mcc.R0(s, a)))

            for i, agent in enumerate(self.mcc.agents):
                f.write("\n\n")

                for s in self.mcc.states:
                    for a in self.mcc.actions:
                        if self.mcc.Ri(agent, s, a) != 0.0:
                            f.write("let R%i[\"%s\", \"%s\", \"%s\", \"%s\"] := %.3f;\n" % (i + 1, convertState[s[0]], convertState[s[1]], convertAction[a[0]], convertAction[a[1]], self.mcc.Ri(agent, s, a)))

    def _export_Ri(self, agentIndex):
        """ Solve the MCC from the constructor by exporting a AMPL file.

            Parameters:
                agentIndex      --  The index of the agent from mcc.agents to export.
        """

        otherAgentIndex = abs(agentIndex - 1)

        # First, we read the output text file to get the necessary information to append about V0Star.
        output = open("ampl/mcc_R0.output", 'r').read().split(" iterations, objective ")[1].split("\n")
        V0Star = float(output[0]) - self.allowableErrorForNLPSolversV0Star

        # Now we do the same thing but on the other output text file to get the psi_{-i} and eta_{-i} best responses.
        output = open("ampl/mcc_R%i.output" % (otherAgentIndex + 1), 'r').read().split(" iterations, objective ")[1].split("\n")
        psiList = list()
        etaList = list()

        counter = 3
        line = ""

        while counter < len(output):
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
        self._export_R0(agentIndex + 1)

        with open("ampl/mcc_R%i.dat" % (agentIndex + 1), 'a') as f:
            f.write("\n\n")
            f.write("let delta := %.5f;\n" % (self.delta))
            f.write("let V0Star := %.5f;\n" % (V0Star))

            f.write("\n\n")
            for psi in psiList:
                f.write(psi)

            f.write("\n\n")
            for eta in etaList:
                f.write(eta)

    def _export_compute_final(self):
        """ Export an AMPL file for computing V0, V1, and V2. """

        # First, we export the final data file using R0 as a base.
        outputFilename = "ampl/mcc_compute_final.dat"
        self._export_R0(filename=outputFilename)

        # Now, we open each of the agents' output files and load their policies.
        for agentIndex in range(len(self.mcc.agents)):
            output = open("ampl/mcc_R%i.output" % (agentIndex + 1), 'r').read().split(" iterations, objective ")[1].split("\n")
            psiList = list()
            etaList = list()

            counter = 3
            line = ""

            while counter < len(output):
                line = list(filter(lambda x: x != "", output[counter].split(" ")))
                if line[0] == ";":
                    break

                param = line[1][1:-1]
                value = max(0.0, min(1.0, float(line[2])))

                if param[0:4] == "psi%i" % (agentIndex + 1):
                    psiList += ["let %s := %.5f;\n" % (param, value)]
                elif param[0:4] == "eta%i" % (agentIndex + 1):
                    etaList += ["let %s := %.5f;\n" % (param, value)]

                counter += 1

            with open(outputFilename, 'a') as f:
                f.write("\n\n")
                for psi in psiList:
                    f.write(psi)

                f.write("\n\n")
                for eta in etaList:
                    f.write(eta)

    def _connect_to_neos(self):
        """ Connect to NEOS and verify that things are working. """

        NEOS_HOST="neos-server.org"
        NEOS_PORT=3333

        self.neos = xmlrpc.client.ServerProxy("https://%s:%d" % (NEOS_HOST, NEOS_PORT))

        if self.neos.ping() != "NeosServer is alive\n":
            print("Failed to make a connection to the NEOS server...")
            sys.exit(1)

    def _submit(self, objectiveIndex, username, password):
        """ Submit a job to NEOS server to solves the group or individual objective once.

            Parameters:
                objectiveIndex      --  The desired objective to solve. If None, then compute final values.
                username            --  The username for NEOS server.
                password            --  The password for NEOS server.

            Returns:
                Returns the resultant (raw) output from the NEOS server.
        """

        # Export the data file if necessary.
        if objectiveIndex is None:
            self._export_compute_final()
        elif objectiveIndex == 0:
            self._export_R0(0)
        else:
            self._export_Ri(objectiveIndex - 1)

        # Load the model, data, and commands files.
        if objectiveIndex is None:
            model = open("ampl/mcc_compute_final.mod").read()
            data = open("ampl/mcc_compute_final.dat").read()
            commands = open("ampl/mcc.cmd").read()
        else:
            model = open("ampl/mcc_R%i.mod" % (objectiveIndex)).read()
            data = open("ampl/mcc_R%i.dat" % (objectiveIndex)).read()
            commands = open("ampl/mcc.cmd").read()

        # Construct the XML string for the job submission.
        xmlString = "<document>\n"
        xmlString += "<category>nco</category><solver>SNOPT</solver><inputMethod>AMPL</inputMethod>\n"
        xmlString += "<model><![CDATA[%s]]></model>\n" % (model)
        xmlString += "<data><![CDATA[%s]]></data>\n" % (data)
        xmlString += "<commands><![CDATA[%s]]></commands>\n" % (commands)
        #xmlString += "<comments><![CDATA[%s]]></comments>\n" % ()
        xmlString += "</document>"

        # Submit the job. If an error happens, handle it. Otherwise, wait for the job to complete.
        (jobNumber, jobPassword) = self.neos.authenticatedSubmitJob(xmlString, username, password)

        if jobNumber == 0:
            print("Failed to submit job, probably because there have too many.")
            raise Exception()

        if objectiveIndex is None:
            print("Submitted job %i. Solving for the values of the final objective function." % (jobNumber), end='')
        else:
            print("Submitted job %i. Solving for objective function %i." % (jobNumber, objectiveIndex), end='')
        sys.stdout.flush()

        # Continuously check if the job is done. Note: The getIntermediateResults function will
        # intentionally hang until a new packet is received from NEOS server.
        offset = 0
        status = ""

        while status != "Done":
            status = self.neos.getJobStatus(jobNumber, jobPassword)

            time.sleep(1)

            print('.', end='')
            sys.stdout.flush()

        time.sleep(3)

        msg = self.neos.getFinalResults(jobNumber, jobPassword)
        result = msg.data.decode()

        time.sleep(1)

        print("Done!")

        return result

    def solve(self, username, password, resolve=True):
        """ Solve the MCC by submitting to NEOS server and doing best response a few times.

            Parameters:
                username    --  The username for NEOS server.
                password    --  The password for NEOS server.
                resolve     --  If the group objective should be resolved or not.

            Returns:
                totalTime    --  The time it took to compute the entire result.
                individualTimes --  A 3-vector (R0, R1, R2) of times to compute each result.
        """

        totalTic = timeit.default_timer()
        totalToc = 0.0
        individualTic = [0.0, list(), list()]
        individualToc = [0.0, list(), list()]

        self._connect_to_neos()

        # First, if desired, submit a job to solve R0. Save the result to an output file.
        result = ""
        if resolve:
            print("Solving Group Objective.")

            individualTic[0] = timeit.default_timer()
            result = self._submit(0, username, password)
            individualToc[0] = timeit.default_timer()

            with open("ampl/mcc_R0.output", "w") as f:
                f.write(result)
        # Otherwise, we need to reset the best response outputs, so load the R0 output.
        else:
            with open("ampl/mcc_R0.output", "r") as f:
                result = f.read()

        with open("ampl/mcc_R1.output", "w") as f:
            f.write(result)
        with open("ampl/mcc_R2.output", "w") as f:
            f.write(result)

        objectiveValues = [0.0, 0.0, 0.0]
        objectiveValues[0] = float(result.split(" iterations, objective ")[1].split("\n")[0])
        print("Group Objective Value: %.5f" % (objectiveValues[0]))

        # Next, iterate over each agent a few times and submit jobs to solve them,
        # each time updating the best response output file.
        print("Initiating Best Response Dynamics.")

        done = 0
        finalPolicyAgentIndex = 0

        lastObjectiveValue = [-1000000.0, -1000000.0]

        for i in range(self.maxNumSteps):
            for agentIndex in range(len(self.mcc.agents)):
                print("Step %i of %i - Agent %i of %i" % (i + 1, self.maxNumSteps, agentIndex + 1, len(self.mcc.agents)))

                individualTic[agentIndex + 1] += [timeit.default_timer()]
                result = self._submit(agentIndex + 1, username, password)
                individualToc[agentIndex + 1] += [timeit.default_timer()]

                with open("ampl/mcc_R%i.output" % (agentIndex + 1), "w") as f:
                    f.write(result)

                # Two Things: First, output the objective value. Second, check if it even computed one!
                try:
                    # Break if we failed to solve the problem this step...
                    if self.checkForNonlinearInfeasibilities and "Nonlinear infeasibilities minimized." in result:
                        raise NonlinearInfeasibilitiesError()

                    objectiveValues[agentIndex + 1] = float(result.split(" iterations, objective ")[1].split("\n")[0])
                    finalPolicyAgentIndex = agentIndex

                    print("Individual Objective %i Value: %.5f" % (agentIndex + 1, objectiveValues[agentIndex + 1]))
                except NonlinearInfeasibilitiesError:
                    print("Failed to solve the problem exactly; had to relax nonlinear constraints... Terminated.")
                    done = -1
                    break
                except IndexError:
                    print("Failed to solve the problem at this step... Terminated.")
                    done = -1
                    break

            # Check for termination. Note: objectiveValues is offset by 1, whereas lastObjectiveValue is not.
            if (abs(lastObjectiveValue[0] - objectiveValues[1]) <= self.epsilon
                    and abs(lastObjectiveValue[1] - objectiveValues[2]) <= self.epsilon):
                done = 1
            elif done != -1:
                lastObjectiveValue[0] = objectiveValues[1]
                lastObjectiveValue[1] = objectiveValues[2]

            if done != 0:
                break

        # Compute and record timings.
        totalToc = timeit.default_timer()
        totalTime = totalToc - totalTic

        individualTimes = [individualToc[0] - individualTic[0], list(), list()]
        for i in range(1, 3):
            individualTime = 0.0
            for j in range(len(individualToc[i])):
                individualTime = (float(j) * individualTime + individualToc[i][j] - individualTic[i][j]) / float(j + 1)
            individualTimes[i] = individualTime

        # Load the FSC policies from the output and data files. Then, we save the FSCs!
        self.fscVector.load_from_output_file(self.mcc, finalPolicyAgentIndex, "ampl/mcc_R%i.output" % (finalPolicyAgentIndex + 1))
        self.fscVector.load_from_data_file(self.mcc, abs(finalPolicyAgentIndex - 1), "ampl/mcc_R%i.dat" % (finalPolicyAgentIndex + 1))
        self.fscVector.save(self.numControllerNodes, self.delta)

        # One last step! Compute the values of these policies.
        if done != -1:
            print("Computing the actual values of each objective under this policy.")
            result = self._submit(None, username, password)
            with open("ampl/mcc_compute_final.output", "w") as f:
                f.write(result)
        else:
            print("Note: Did not compute actual values because it failed at some point above.")

        print("Completed. Final Objective Values: ", objectiveValues)

        return totalTime, individualTimes


if __name__ == "__main__":
    if len(sys.argv) == 6:
        mcc = MCC()
        fscVector = FSCVector([FSC(mcc, agent, int(sys.argv[3])) for agent in mcc.agents])
        mccSolve = MCCSolve(mcc, fscVector, maxNumSteps=int(sys.argv[4]), delta=float(sys.argv[5]))
        totalTime, individualTimes = mccSolve.solve(sys.argv[1], sys.argv[2])
        print("Individual Times: [R0: %.2fs, R1: %.2fs, R2: %.2fs]" % (individualTimes[0], individualTimes[1], individualTimes[2]))
        print("Total Time: %.2f seconds" % (totalTime))
    else:
        print("Format: python3 mcc_solve.py <username> <password> <num controller nodes> <num BR iterations> <slack>")



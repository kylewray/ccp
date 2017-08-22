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
import os.path

import xmlrpc.server
import xmlrpc.client

from mcc_model import *
from fsc_vector import *


class MCCSolve(object):
    """ This solves an MCC by converting it to an AMPL file. """

    def __init__(self, mcc, fscVector, numSteps=3, delta=0.0):
        """ The constructor for the MCCSolve class.

            Parameters:
                mcc         --  The MCC object.
                fscVector   --  The object for each agent's finite state controller.
                numSteps    --  The number of best response iterations.
                delta       --  The slack variable, non-negative.
        """

        self.mcc = mcc
        self.fscVector = fscVector
        self.numControllerNodes = max([fsc.n for fsc in fscVector.fscs])
        self.delta = delta
        self.numSteps = numSteps

        self.neos = None

    def _export_R0(self, filenameIndex=0):
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
        V0Star = float(output[0])

        # Now we do the same thing but on the other output text file to get the psi_{-i} and eta_{-i} best responses.
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
                objectiveIndex      --  The desired objective to solve.
                username            --  The username for NEOS server.
                password            --  The password for NEOS server.

            Returns:
                Returns the resultant (raw) output from the NEOS server.
        """

        # Export the data file if necessary.
        if objectiveIndex == 0:
            self._export_R0(0)
        else:
            self._export_Ri(objectiveIndex - 1)

        # Load the model, data, and commands files.
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

        print("Submitted job %i. Solving for objective function %i." % (jobNumber, objectiveIndex), end='')
        sys.stdout.flush()

        # Continuously check if the job is done. Note: The getIntermediateResults function will
        # intentionally hang until a new packet is received from NEOS server.
        offset = 0
        status = ""

        while status != "Done":
            status = self.neos.getJobStatus(jobNumber, jobPassword)

            time.sleep(5)

            print('.', end='')
            sys.stdout.flush()

        msg = self.neos.getFinalResults(jobNumber, jobPassword)
        result = msg.data.decode()

        print("Done!")

        return result

    def solve(self, username, password):
        """ Solve the MCC by submitting to NEOS server and doing best response a few times.

            Parameters:
                username    --  The username for NEOS server.
                password    --  The password for NEOS server.
        """

        self._connect_to_neos()

        solveGroupObjective = True

        # Special: Do not waste time if we already solved the group objective.
        if os.path.isfile("ampl/mcc_R0.output"):
            answer = " "
            while answer not in ['', 'y', 'n']:
                print("Group objective solution has already been computed! Recompute it (y/N)?")
                answer = input()
            solveGroupObjective = (answer == 'y')

        # First, if desired, submit a job to solve R0. Save the result to an output file.
        result = ""

        if solveGroupObjective:
            print("Solving Group Objective.")
            result = self._submit(0, username, password)
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

        done = False
        finalPolicyAgentIndex = 0

        for i in range(self.numSteps):
            for agentIndex in range(len(self.mcc.agents)):
                print("Step %i of %i - Agent %i of %i" % (i + 1, self.numSteps, agentIndex + 1, len(self.mcc.agents)))
                result = self._submit(agentIndex + 1, username, password)

                with open("ampl/mcc_R%i.output" % (agentIndex + 1), "w") as f:
                    f.write(result)

                # Two Things: First, output the objective value. Second, check if it even computed one!
                try:
                    objectiveValues[agentIndex + 1] = float(result.split(" iterations, objective ")[1].split("\n")[0])
                    print("Individual Objective %i Value: %.5f" % (agentIndex + 1, objectiveValues[agentIndex + 1]))
                except IndexError:
                    print("Failed to solve the problem at this step... Terminated.")
                    done = True
                    break

                finalPolicyAgentIndex = agentIndex
            if done:
                break

        # Load the FSC policies from the output and data files. Then, we save the FSCs!
        fscVector.load_from_output_file(self.mcc, finalPolicyAgentIndex, "ampl/mcc_R%i.output" % (finalPolicyAgentIndex + 1))
        fscVector.load_from_data_file(self.mcc, abs(finalPolicyAgentIndex - 1), "ampl/mcc_R%i.dat" % (abs(finalPolicyAgentIndex - 1) + 1))
        fscVector.save(self.numControllerNodes, self.delta)

        print("Completed. Final Objective Values: ", objectiveValues)


if __name__ == "__main__":
    if len(sys.argv) == 6:
        mcc = MCC()
        fscVector = FSCVector([FSC(mcc, agent, int(sys.argv[3])) for agent in mcc.agents])
        mccSolve = MCCSolve(mcc, fscVector, numSteps=int(sys.argv[4]), delta=float(sys.argv[5]))
        mccSolve.solve(sys.argv[1], sys.argv[2])
    else:
        print("Format: python3 mcc_solve.py <username> <password> <num controller nodes> <num BR iterations> <slack>")



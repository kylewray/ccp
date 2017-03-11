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


import rospy

from tf.transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from mcc.msg import *
from mcc.srv import *

import math
import random
import itertools as it
import ctypes as ct
import numpy as np

import time


MCC_OCCUPANCY_GRID_OBSTACLE_THRESHOLD = 50
MCC_GOAL_THRESHOLD = 0.001


class MCCModel(object):
    """ The code which controls the robot following an MCC model's FSC. """

    def __init__(self):
        """ The constructor for the MCCModel class. """

        # The POMDP and other related information. Note: ctypes pointer (policy) is 'False' if null. Otherwise,
        # we can call policy.contents to dereference the pointer.
        self.pomdp = POMDP()
        self.policy = None
        self.belief = None

        self.initialBeliefIsSet = False
        self.goalIsSet = False
        self.algorithmIsInitialized = False

        self.stateObstaclePercentage = None

        self.baseProbabilityOfActionSuccess = rospy.get_param("~base_probability_of_action_success", 0.8)
        self.penaltyForFreespace = rospy.get_param("~penalty_for_freespace", 0.1)
        #self.numberOfBeliefsToAdd = rospy.get_param("~number_of_beliefs_to_add", 10000)
        self.numberOfBeliefExpansions = rospy.get_param("~number_of_belief_expansions", 2)

        # Information about the map for use by a path follower once our paths are published.
        self.mapWidth = 0
        self.mapHeight = 0
        self.mapOriginX = 0.0
        self.mapOriginY = 0.0
        self.mapResolution = 1.0

        # This is the number of x and y states that will be created using the map. Obstacle states
        # will, of course, be omitted.
        self.gridWidth = rospy.get_param("~grid_width", 10)
        self.gridHeight = rospy.get_param("~grid_height", 10)

        # There is a pre-determined number of updates before get_action will return something.
        self.numberOfUpdates = 0
        self.numberOfUpdatesBeforeProvidingActions = rospy.get_param("~number_of_updates_before_providing_actions",
                                                                     self.gridWidth * self.gridHeight)

        # Store if we performed the initial theta adjustment and the final goal theta adjustment
        self.performedInitialPoseAdjustment = False
        self.initialPoseAdjustmentX = 0.0
        self.initialPoseAdjustmentY = 0.0
        self.initialPoseAdjustmentTheta = 0.0

        self.performedGoalAdjustment = False
        self.goalAdjustmentX = 0.0
        self.goalAdjustmentY = 0.0
        self.goalAdjustmentTheta = 0.0

        # Subscribers, publishers, services, etc. for ROS messages.
        self.subOccupancyGrid = None
        self.subMapPoseEstimate = None
        self.subMapNavGoal = None

        self.occupancyGridMsg = None
        self.mapPoseEstimateMsg = None
        self.mapNavGoalMsg = None

        self.pubModelUpdate = None
        self.srvGetAction = None
        self.srvGetFSCState = None
        self.srvUpdateFSC = None

        # Finally, this optionally can publish markers to a topic, allowing for a visualization of the FSC state.
        self.visualizeFSCState = rospy.get_param("~visualize_fsc_state", False)
        self.pubMarker = None
        self.pubMarkerArray = None

    def __del__(self):
        """ The deconstructor for the MCCModel class. """

        if self.algorithmIsInitialized:
            self.uninitializeAlgorithm()

    def initialize(self):
        """ Initialize the MCCModel class, mainly registering subscribers and services. """

        subOccupancyGridTopic = rospy.get_param("~sub_occupancy_grid", "/map")
        self.subOccupancyGrid = rospy.Subscriber(subOccupancyGridTopic,
                                                 OccupancyGrid,
                                                 self.sub_occupancy_grid)

        subMapPoseEstimateTopic = rospy.get_param("~sub_map_pose_estimate", "/initialpose")
        self.subMapPoseEstimate = rospy.Subscriber(subMapPoseEstimateTopic,
                                                   PoseWithCovarianceStamped,
                                                   self.sub_map_pose_estimate)

        subMapNavGoalTopic = rospy.get_param("~sub_map_nav_goal", "/move_base_simple/goal")
        self.subMapNavGoal = rospy.Subscriber(subMapNavGoalTopic,
                                              PoseStamped,
                                              self.sub_map_nav_goal)

        pubModelUpdateTopic = rospy.get_param("~model_update", "~model_update")
        self.pubModelUpdate = rospy.Publisher(pubModelUpdateTopic, ModelUpdate, queue_size=10)

        srvGetActionTopic = rospy.get_param("~get_action", "~get_action")
        self.srvGetAction = rospy.Service(srvGetActionTopic,
                                          GetAction,
                                          self.srv_get_action)

        srvGetFSCStateTopic = rospy.get_param("~get_fsc_state", "~get_fsc_state")
        self.srvGetFSCState = rospy.Service(srvGetFSCStateTopic,
                                            GetFSCState,
                                            self.srv_get_fsc_state)

        srvUpdateFSCTopic = rospy.get_param("~update_fsc", "~update_fsc")
        self.srvUpdateFSC = rospy.Service(srvUpdateFSCTopic,
                                             UpdateFSC,
                                             self.srv_update_fsc)

        pubVisualizationMarkerTopic = rospy.get_param("~visualization_marker",
                                                      "~visualization_marker")
        self.pubMarker = rospy.Publisher(pubVisualizationMarkerTopic, Marker, queue_size=1000)

        pubVisualizationMarkerArrayTopic = rospy.get_param("~visualization_marker_array",
                                                           "~visualization_marker_array")
        self.pubMarkerArray = rospy.Publisher(pubVisualizationMarkerArrayTopic, MarkerArray, queue_size=32)

    def update(self):
        """ Update the MCCModel object. """

        # These methods deal with the threading issue. Basically, the update below could be called
        # while the POMDP itself is being modified in a different thread. This can easily be reproduced
        # by continually assigning new initial pose estimates and goals. Instead, however, we have
        # any subscriber callbacks assign a variable with the message. This message is then handled
        # as part of the main node's thread update call (here).
        self.handle_occupancy_grid_message()
        self.handle_map_pose_estimate_msg()
        self.handle_map_nav_goal_msg()

        # We only update once we have a valid POMDP.
        if self.pomdp is None or not self.initialBeliefIsSet or not self.goalIsSet:
            return

        # If this is the first time the POMDP has been ready to be updated, then
        # initialize necessary variables.
        if not self.algorithmIsInitialized:
            self.initialize_algorithm()

        #rospy.loginfo("Info[MCCModel.update]: Updating the policy.")

        # This can be NOVA_SUCCESS or NOVA_CONVERGED.
        result = self.solver.update()

        self.numberOfUpdates += 1

    def initialize_algorithm(self):
        """ Initialize the POMDP algorithm. """

        if self.algorithmIsInitialized:
            rospy.logwarn("Warn[MCCModel.initialize_algorithm]: Algorithm is already initialized.")
            return

        if not self.initialBeliefIsSet or not self.goalIsSet:
            rospy.logwarn("Warn[MCCModel.initialize_algorithm]: Initial belief or goal is not set yet.")
            return

        initialGamma = np.array([[float(self.pomdp.Rmin / (1.0 - self.pomdp.gamma)) \
                                        for s in range(self.pomdp.n)] \
                                    for i in range(self.pomdp.r)])

        array_type_rn_float = ct.c_float * (self.pomdp.r * self.pomdp.n)
        initialGamma = array_type_rn_float(*initialGamma.flatten())

        rospy.loginfo("Info[MCCModel.initialize_algorithm]: Initializing the algorithm.")

        try:
            self.solver = solver.POMDPPBVICPU(self.pomdp, initialGamma)
        except:
            rospy.logerr("Error[MCCModel.initialize_algorithm]: Failed to initialize algorithm.")
            return

        self.numberOfUpdates = 0

        self.algorithmIsInitialized = True

    def uninitialize_algorithm(self):
        """ Uninitialize the POMDP algorithm. """

        if not self.algorithmIsInitialized:
            rospy.logwarn("Warn[MCCModel.uninitialize_algorithm]: Algorithm has not been initialized.")
            return

        rospy.loginfo("Info[MCCModel.uninitialize_algorithm]: Uninitializing the algorithm.")

        self.policy = None

        try:
            del self.solver
        except:
            rospy.logwarn("Warn[MCCModel.uninitialize_algorithm]: Failed to uninitialize algorithm.")

        self.algorithmIsInitialized = False

    def visualize_fsc_state(self):
        """ Visualize the FSC state, if desired, by publishing markers. """

        if self.visualizeFSCState != True or self.pomdp is None or self.belief is None:
            return

        #rospy.loginfo("Info[MCCModel.visualize_fsc_state]: Starting to place markers.")

        marker = Marker()
        marker.header.frame_id = "map" #subOccupancyGridTopic
        marker.header.stamp = rospy.get_rostime()
        marker.ns = "MCC Model FSC State"
        marker.id = 0
        marker.action = 3 #Marker.DELETEALL

        self.pubMarker.publish(marker)

        # Put a 50ms pause here because publishing the delete all and update below sometimes
        # happen out of order without a small pause...
        time.sleep(0.05)

        # Quickly compute the top k belief states; these will be colored blue.
        k = 3
        topStates = sorted(range(len(self.belief)), key=lambda s: self.belief[s], reverse=True)
        topStates = topStates[0:k]

        markerArray = list()

        for s, state in enumerate(self.pomdp.states):
            marker = Marker()

            marker.header.frame_id = "map" #subOccupancyGridTopic
            marker.header.stamp = rospy.get_rostime()
            marker.ns = "MCC Model FSC State"
            marker.id = s
            marker.type = 1 #Marker.CUBE
            marker.action = 0 #Marker.ADD

            x, y = self.map_to_world(state[0] + 0.5, state[1] + 0.5)
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.015

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = self.mapResolution * self.mapWidth / self.gridWidth
            marker.scale.y = self.mapResolution * self.mapHeight / self.gridHeight
            marker.scale.z = 0.1

            # Increase to show more probabilities.
            alpha = self.belief[s]
            alphaLog = 0.0
            alphaWeight = 5.0
            if alpha > 0.001 and math.log(alpha) >= -alphaWeight:
                alphaLog = (math.log(alpha) + alphaWeight) / alphaWeight

            marker.color.a = alphaLog

            # Color the top k states a different color.
            if s in topStates:
                marker.color.r = 0.2
                marker.color.g = 0.2
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0

            markerArray += [marker]

        self.pubMarkerArray.publish(markerArray)

        #rospy.loginfo("Info[MCCModel.visualize_belief]: Completed placing markers.")

    def map_to_world(self, mx, my):
        """ Convert map coordinates (integers) to world coordinates (offset, resolution, floats).

            Parameters:
                mx      --  The map x coordinate.
                my      --  The map y coordinate.

            Returns:
                wx      --  The resultant world x coordinate. This is the center of the grid cell.
                wy      --  The resultant world y coordinate. This is the center of the grid cell.
        """

        xSize = self.mapWidth / self.gridWidth
        ySize = self.mapHeight / self.gridHeight

        mx *= xSize
        my *= ySize

        wx = self.mapOriginX + mx * self.mapResolution
        wy = self.mapOriginY + my * self.mapResolution

        return wx, wy

    def world_to_map(self, wx, wy):
        """ Convert world coordinates (offset, resolution, floats) to map coordinates (integers).

            Parameters:
                wx      --  The world x coordinate.
                wy      --  The world y coordinate.
                self    --  A reference to the relevant MCCModel object.

            Returns:
                mx      --  The resultant map x coordinate.
                my      --  The resultant map y coordinate.

            Exceptions:
                The world coordinate is outside of the map.
        """

        if wx < self.mapOriginX or wy < self.mapOriginY or \
                wx >= self.mapOriginX + self.mapWidth * self.mapResolution or \
                wy >= self.mapOriginY + self.mapHeight * self.mapResolution:
            raise Exception()

        mx = (wx - self.mapOriginX) / self.mapResolution
        my = (wy - self.mapOriginY) / self.mapResolution

        xSize = self.mapWidth / self.gridWidth
        ySize = self.mapHeight / self.gridHeight

        mx = int(mx / xSize)
        my = int(my / ySize)

        return mx, my

    def sub_occupancy_grid(self, msg):
        """ A subscriber for OccupancyGrid messages. This converges any 2d map
            into a set of POMDP states. This is a static method to work as a ROS callback.

            Parameters:
                msg     --  The OccupancyGrid message data.
        """

        if self.occupancyGridMsg is None:
            self.occupancyGridMsg = msg

    def handle_occupancy_grid_message(self):
        """ A handler for OccupancyGrid messages. This converges any 2d map
            into a set of POMDP states. This is a static method to work as a ROS callback.
        """

        if self.occupancyGridMsg is None:
            return
        msg = self.occupancyGridMsg

        rospy.loginfo("Info[MCCModel.sub_occupancy_grid]: Received map. Creating a new POMDP.")

        # Remember map information.
        self.mapWidth = msg.info.width
        self.mapHeight = msg.info.height
        self.mapOriginX = msg.info.origin.position.x
        self.mapOriginY = msg.info.origin.position.y
        self.mapResolution = msg.info.resolution

        xStep = int(self.mapWidth / self.gridWidth)
        yStep = int(self.mapHeight / self.gridHeight)

        # Create a new POMDP every time the map is updated. This 'resets' the goal of course.
        self.pomdp = POMDP()

        self.initialBeliefIsSet = False
        self.goalIsSet = False
        if self.algorithmIsInitialized:
            self.uninitialize_algorithm()

        self.pomdp.states = list(it.product(range(self.gridWidth), range(self.gridHeight)))
        self.pomdp.n = len(self.pomdp.states)

        self.pomdp.actions = list(it.product([-1, 0, 1], [-1, 0, 1]))
        #self.pomdp.actions.remove((0, 0))
        self.pomdp.m = len(self.pomdp.actions)

        self.pomdp.observations = list([False, True])
        self.pomdp.z = len(self.pomdp.observations)

        # Compute the left and right actions for later use computing probability of drifting.
        driftActions = dict()
        for ax, ay in self.pomdp.actions:
            # Three cases: zero, diagonal, or axis-aligned.
            if ax == 0 and ay == 0:
                driftActions[(ax, ay)] = None
            elif ax != 0 and ay != 0:
                driftActions[(ax, ay)] = [(ax, 0), (0, ay)]
            else:
                if ax != 0:
                    driftActions[(ax, ay)] = [(ax, -1), (ax, 1)]
                elif ay != 0:
                    driftActions[(ax, ay)] = [(-1, ay), (1, ay)]

        # Compute the percentage of obstacles within each state.
        self.stateObstaclePercentage = dict()
        for x, y in self.pomdp.states:
            self.stateObstaclePercentage[(x, y)] = 0

            pixelOffsets = list(it.product(range(xStep), range(yStep)))
            for i, j in pixelOffsets:
                if msg.data[(y * yStep + j) * self.mapWidth + (x * xStep + i)] >= MCC_OCCUPANCY_GRID_OBSTACLE_THRESHOLD:
                    self.stateObstaclePercentage[(x, y)] += 1

            self.stateObstaclePercentage[(x, y)] = float(self.stateObstaclePercentage[(x, y)]) / float(len(pixelOffsets))

        # The number of successors is always only uncertain about left/right, or self-looping at center.
        self.pomdp.ns = 4

        # Compute the state transitions!
        S = [[[int(-1) for sp in range(self.pomdp.ns)] for a in range(self.pomdp.m)] for s in range(self.pomdp.n)]
        T = [[[float(0.0) for sp in range(self.pomdp.ns)] for a in range(self.pomdp.m)] for s in range(self.pomdp.n)]

        for s, state in enumerate(self.pomdp.states):
            for a, action in enumerate(self.pomdp.actions):
                # First, if the action is (0, 0) (i.e., no movement), then it automatically succeeds.
                if action == (0, 0):
                    S[s][a][0] = s
                    T[s][a][0] = 1.0
                    continue

                # Second, if the agent is *in* an obstacle, then self-loop.
                if self.stateObstaclePercentage[state] == 1.0:
                    S[s][a][0] = s
                    T[s][a][0] = 1.0
                    continue

                # Third, if any drifting is guaranteed to hit an obstacle, then self-loop.
                diagonalActionDriftStateEntirelyObstacle = False
                if action[0] != 0 and action[1] != 0:
                    for i in [0, 1]:
                        try:
                            statePrime = (state[0] + driftActions[action][i][0], state[1] + driftActions[action][i][1])
                            sp = self.pomdp.states.index(statePrime)

                            if self.stateObstaclePercentage[statePrime] == 1.0:
                                diagonalActionDriftStateEntirelyObstacle = True

                        except ValueError:
                            continue

                if diagonalActionDriftStateEntirelyObstacle:
                    S[s][a][0] = s
                    T[s][a][0] = 1.0
                    continue

                # Begin with the pure probability of success and failure for each action.
                prSuccess = self.baseProbabilityOfActionSuccess
                prFailDrift = [(1.0 - self.baseProbabilityOfActionSuccess) / 2.0,
                               (1.0 - self.baseProbabilityOfActionSuccess) / 2.0]
                prFailNoMove = 0.0

                cur = 0

                # Now adjust the success probability based on the obstacle percentage at that state.
                try:
                    statePrime = (state[0] + action[0], state[1] + action[1])
                    sp = self.pomdp.states.index(statePrime)

                    S[s][a][cur] = sp
                    T[s][a][cur] = prSuccess * (1.0 - self.stateObstaclePercentage[statePrime])
                    cur += 1

                    prFailNoMove += prSuccess * self.stateObstaclePercentage[statePrime]
                except ValueError:
                    prFailNoMove += prSuccess
                    prSuccess = 0.0

                # Do the same for drifting side-to-side.
                for i in [0, 1]:
                    try:
                        statePrime = (state[0] + driftActions[action][i][0], state[1] + driftActions[action][i][1])
                        sp = self.pomdp.states.index(statePrime)

                        S[s][a][cur] = sp
                        T[s][a][cur] = prFailDrift[i] * (1.0 - self.stateObstaclePercentage[statePrime])
                        cur += 1

                        prFailNoMove += prFailDrift[i] * self.stateObstaclePercentage[statePrime]
                    except ValueError:
                        prFailNoMove += prFailDrift[i]
                        prFailDrift[i] = 0.0

                # Finally, if there was a non-zero value for the probability of failure, then assign this, too.
                if prFailNoMove > 0.0:
                    S[s][a][cur] = s
                    T[s][a][cur] = prFailNoMove
                    cur += 1

        array_type_nmns_int = ct.c_int * (self.pomdp.n * self.pomdp.m * self.pomdp.ns)
        array_type_nmns_float = ct.c_float * (self.pomdp.n * self.pomdp.m * self.pomdp.ns)

        self.pomdp.S = array_type_nmns_int(*np.array(S).flatten())
        self.pomdp.T = array_type_nmns_float(*np.array(T).flatten())

        # Compute the observation function!
        O = [[[0.0 for z in range(self.pomdp.z)] for sp in range(self.pomdp.n)] for a in range(self.pomdp.m)]

        for a, action in enumerate(self.pomdp.actions):
            for sp, statePrime in enumerate(self.pomdp.states):
                spx, spy = statePrime

                probability = self.stateObstaclePercentage[statePrime]

                # For movement actions, add the probability contribution of the
                # action's and drift-action's states, then normalize.
                if action != (0, 0):
                    for ax, ay in [action] + driftActions[action]:
                        x = max(0, min(self.gridWidth - 1, spx + ax))
                        y = max(0, min(self.gridHeight - 1, spy + ay))
                        actionState = (x, y)

                        probability += self.stateObstaclePercentage[actionState]

                    probability /= 4.0

                # The probability of observing a hit at a state is (assumed to be) equal to the percentage of obstacles in the state.
                O[a][sp][0] = 1.0 - probability
                O[a][sp][1] = probability

        array_type_mnz_float = ct.c_float * (self.pomdp.m * self.pomdp.n * self.pomdp.z)
        self.pomdp.O = array_type_mnz_float(*np.array(O).flatten())

        self.pomdp.gamma = 0.99
        self.pomdp.horizon = self.gridWidth * self.gridHeight

        #print(np.array(S))
        #print(np.array(T))
        #print(np.array(O))
        #for a, action in enumerate(self.pomdp.actions):
        #    for sp, statePrime in enumerate(self.pomdp.states):
        #        print("O(%i, %i, [0, 1]) = [%.2f, %.2f]" % (a, sp, O[a][sp][0], O[a][sp][1]))
        ##        #print("O(%s, %s, %s) = [%.1f, %.1f]" % (str(self.pomdp.actions[a]), str(self.pomdp.states[sp]), str(self.pomdp.observations), O[a][sp][0], O[a][sp][1]))

        self.occupancyGridMsg = None

        self.pubModelUpdate.publish(ModelUpdate())

    def sub_map_pose_estimate(self, msg):
        """ A subscriber for PoseWithCovarianceStamped messages. This is when an initial
            pose is assigned, inducing an initial FSC State. This is a static method to work as a
            ROS callback.

            Parameters:
                msg     --  The PoseWithCovarianceStamped message data.
        """

        if self.mapPoseEstimateMsg is None:
            self.mapPoseEstimateMsg = msg

    def handle_map_pose_estimate_msg(self):
        """ A handler for PoseWithCovarianceStamped messages. This is when an initial
            pose is assigned, inducing an initial belief. This is a static method to work as a
            ROS callback.
        """

        if self.mapPoseEstimateMsg is None:
            return
        msg = self.mapPoseEstimateMsg

        if self.pomdp is None:
            rospy.logwarn("Warn[MCCModel.sub_map_pose_estimate]: POMDP has not yet been defined.")
            return

        try:
            gridPoseEstimateX, gridPoseEstimateY = self.world_to_map(msg.pose.pose.position.x, msg.pose.pose.position.y)
            sInitialBelief = self.pomdp.states.index((gridPoseEstimateX, gridPoseEstimateY))
        except Exception:
            rospy.logwarn("Warn[MCCModel.sub_map_pose_estimate]: Pose estimate position is outside of map bounds.")
            return

        rospy.loginfo("Info[MCCModel.sub_map_pose_estimate]: Received pose estimate. Assigning POMDP initial beliefs.")

        # Setup the initial pose adjustment.
        worldPoseEstimateOfGridX, worldPoseEstimateOfGridY = self.map_to_world(gridPoseEstimateX, gridPoseEstimateY)
        self.initialPoseAdjustmentX = worldPoseEstimateOfGridX - gridPoseEstimateX
        self.initialPoseAdjustmentY = worldPoseEstimateOfGridY - gridPoseEstimateY

        roll, pitch, yaw = euler_from_quaternion([msg.pose.pose.orientation.x,
                                                  msg.pose.pose.orientation.y,
                                                  msg.pose.pose.orientation.z,
                                                  msg.pose.pose.orientation.w])
        self.initialPoseAdjustmentTheta = -yaw

        self.performedInitialPoseAdjustment = False

        # Random Version: The seed belief is simply a collapsed belief on this initial pose.
        #Z = [[sInitialBelief]]
        #B = [[1.0]]

        #self.pomdp.r = len(B)
        #self.pomdp.rz = 1

        #array_type_rrz_int = ct.c_int * (self.pomdp.r * self.pomdp.rz)
        #array_type_rrz_float = ct.c_float * (self.pomdp.r * self.pomdp.rz)

        #self.pomdp.Z = array_type_rrz_int(*np.array(Z).flatten())
        #self.pomdp.B = array_type_rrz_float(*np.array(B).flatten())

        #self.pomdp.expand(method='random', numBeliefsToAdd=self.numberOfBeliefsToAdd)

        # Intelligent Version: Assign the belief to be intelligently spread over each state.
        Z = [[s] for s in range(self.pomdp.n)]
        B = [[1.0] for s in range(self.pomdp.n)]

        self.pomdp.r = len(B)
        self.pomdp.rz = 1

        array_type_rrz_int = ct.c_int * (self.pomdp.r * self.pomdp.rz)
        array_type_rrz_float = ct.c_float * (self.pomdp.r * self.pomdp.rz)

        self.pomdp.Z = array_type_rrz_int(*np.array(Z).flatten())
        self.pomdp.B = array_type_rrz_float(*np.array(B).flatten())

        #for i in range(self.numberOfBeliefExpansions):
        #    rospy.loginfo("Info[MCCModel.sub_map_pose_estimate]: Starting expansion %i." % (i + 1))
        #    self.pomdp.expand(method='distinct_beliefs')

        #rospy.loginfo("Info[MCCModel.sub_map_pose_estimate]: Completed expansions.")

        # Enable/Disable Sigma Approximation!
        # Enable/Disable Sigma Approximation!
        # Enable/Disable Sigma Approximation!
        #sigmaValue = self.pomdp.sigma_approximate(4)
        #rospy.loginfo("Info[MCCModel.sub_map_pose_estimate]: Sigma approximation applied! Sigma is %.3f." % (sigmaValue))
        # Enable/Disable Sigma Approximation!
        # Enable/Disable Sigma Approximation!
        # Enable/Disable Sigma Approximation!

        self.uninitialize_algorithm()
        self.initialize_algorithm()

        # Set the initial belief to be collapsed at the correct location.

        # Belief Initialization Method 1/3 -- Exact On Point
        self.belief = np.array([float(s == sInitialBelief) for s in range(self.pomdp.n)])

        # Belief Initialization Method 2/3 -- Uniform
        #self.belief = np.array([1.0 / float(self.pomdp.n) for s in range(self.pomdp.n)])

        # Belief Initialization Method 3/3 -- Randomly Around Point
        #self.belief = np.array([random.random() / (math.sqrt(pow(gridPoseEstimateX - state[0], 2) \
        #                                                        + pow(gridPoseEstimateY - state[1], 2)) + 1.0) \
        #                            * float(math.sqrt(pow(gridPoseEstimateX - state[0], 2) \
        #                                                        + pow(gridPoseEstimateY - state[1], 2)) < 5.0 \
        #                            and self.stateObstaclePercentage[state] < 1.0) \
        #                        for state in it.product(range(self.gridWidth), range(self.gridHeight))])
        #self.belief /= self.belief.sum()

        # Optionally, visualize the belief.
        self.visualize_belief()

        self.initialBeliefIsSet = True

        self.mapPoseEstimateMsg = None

        self.pubModelUpdate.publish(ModelUpdate())

    def sub_map_nav_goal(self, msg):
        """ A subscriber for PoseStamped messages. This is called when a goal is provided,
            assigning the rewards for the POMDP. This is a static method to work as a ROS callback.

            Parameters:
                msg     --  The OccupancyGrid message data.
        """

        if self.mapNavGoalMsg is None:
            self.mapNavGoalMsg = msg

    def handle_map_nav_goal_msg(self):
        """ A handler for PoseStamped messages. This is called when a goal is provided,
            assigning the rewards for the POMDP. This is a static method to work as a ROS callback.
        """

        if self.mapNavGoalMsg is None:
            return
        msg = self.mapNavGoalMsg

        if self.pomdp is None:
            rospy.logwarn("Warn[MCCModel.sub_map_nav_goal]: POMDP has not yet been defined.")
            return

        try:
            gridGoalX, gridGoalY = self.world_to_map(msg.pose.position.x, msg.pose.position.y)
            sGoal = self.pomdp.states.index((gridGoalX, gridGoalY))
        except Exception:
            rospy.logwarn("Warn[MCCModel.sub_map_nav_goal]: Goal position is outside of map bounds.")
            return

        rospy.loginfo("Info[MCCModel.sub_map_nav_goal]: Received goal. Assigning POMDP rewards.")

        # Setup the goal theta adjustment.
        worldGoalOfGridX, worldGoalOfGridY = self.map_to_world(gridGoalX, gridGoalY)
        self.goalAdjustmentX = worldGoalOfGridX - gridGoalX
        self.goalAdjustmentY = worldGoalOfGridY - gridGoalY

        roll, pitch, yaw = euler_from_quaternion([msg.pose.orientation.x,
                                                  msg.pose.orientation.y,
                                                  msg.pose.orientation.z,
                                                  msg.pose.orientation.w])
        self.goalAdjustmentTheta = yaw

        self.performedGoalAdjustment = False

        R = [[0.0 for a in range(self.pomdp.m)] for s in range(self.pomdp.n)]

        for s, state in enumerate(self.pomdp.states):
            for a, action in enumerate(self.pomdp.actions):
                if gridGoalX == state[0] and gridGoalY == state[1] and action == (0, 0):
                    R[s][a] = 0.0
                #elif self.stateObstaclePercentage[state] == 1.0:
                #    R[s][a] = -100.0
                else:
                    #R[s][a] = -1.0
                    R[s][a] = min(-self.penaltyForFreespace, -self.stateObstaclePercentage[state])

        self.pomdp.Rmax = np.array(R).max()
        self.pomdp.Rmin = np.array(R).min()

        array_type_nm_float = ct.c_float * (self.pomdp.n * self.pomdp.m)

        self.pomdp.R = array_type_nm_float(*np.array(R).flatten())

        self.uninitialize_algorithm()
        self.initialize_algorithm()

        self.goalIsSet = True

        self.mapNavGoalMsg = None

        self.pubModelUpdate.publish(ModelUpdate())

        #print(np.array(R))
        #print(self.pomdp)

    def srv_get_action(self, req):
        """ This service returns an action based on the current FSC state, provided enough updates were done.

            Parameters:
                req     --  The service request as part of GetAction.

            Returns:
                The service response as part of GetAction.
        """

        if self.pomdp is None or not self.initialBeliefIsSet or not self.goalIsSet or self.belief is None:
            rospy.logerr("Error[MCCModel.srv_get_action]: POMDP or belief are undefined.")
            return GetActionResponse(False, 0.0, 0.0, 0.0)

        # Do nothing until the number of updates is sufficient.
        if self.numberOfUpdates < self.numberOfUpdatesBeforeProvidingActions:
            return GetActionResponse(False, 0.0, 0.0, 0.0)

        # Reset the policy so we can get the newest one.
        self.policy = None

        try:
            self.policy = self.solver.get_policy()
        except:
            rospy.logerr("Error[MCCModel.srv_get_action]: Could not get policy.")
            return GetActionResponse(False, 0.0, 0.0, 0.0)

        value, actionIndex = self.policy.value_and_action(self.belief)
        actionIndex = int(actionIndex)

        # The relative goal is simply the relative location based on the "grid-ize-ation"
        # and resolution of the map. The goal theta is a bit harder to compute (estimate).
        goalX, goalY = self.pomdp.actions[actionIndex]

        xSize = self.mapWidth / self.gridWidth
        ySize = self.mapHeight / self.gridHeight

        goalX *= xSize * self.mapResolution
        goalY *= ySize * self.mapResolution

        # Visualize the belief, if desired.
        self.visualize_belief()

        # If this is the first action we take, then we need to offset the goalX and goalY
        # as well as assign a goalTheta to properly setup the initial motion. Also, if the
        # goal is reached, then we need to perform the final theta rotation. Otherwise,
        # the adjustment required is simply 0; the path (action) follower will handle this.
        if not self.performedInitialPoseAdjustment:
            #goalX += self.initialPoseAdjustmentX
            #goalY += self.initialPoseAdjustmentY
            goalTheta = self.initialPoseAdjustmentTheta
            self.performedInitialPoseAdjustment = True
        elif not self.performedGoalAdjustment and self.pomdp.actions[actionIndex] == (0, 0):
            #goalX += self.goalAdjustmentX
            #goalY += self.goalAdjustmentY
            goalTheta = self.goalAdjustmentTheta
            self.performedGoalAdjustment = True
        else:
            goalTheta = 0.0

        return GetActionResponse(True, goalX, goalY, goalTheta)

    def srv_get_fsc_state(self, req):
        """ This service returns the current FSC state.

            Parameters:
                req     --  The service request as part of GetFSCState.

            Returns:
                The service response as part of GetFSCState.
        """

        if self.pomdp is None or not self.initialBeliefIsSet or not self.goalIsSet or self.belief is None:
            rospy.logerr("Error[MCCModel.srv_get_belief]: POMDP or belief are undefined.")
            return GetFSCStateResponse(list())

        self.visualize_fsc_state()

        return GetFSCStateResponse(self.belief.tolist())

    def srv_update_fsc(self, req):
        """ This service updates the FSC based on a given action and observation.

            Parameters:
                req     --  The service request as part of UpdateFSC.

            Returns:
                The service response as part of UpdateFSC.
        """

        if self.pomdp is None or not self.initialBeliefIsSet or not self.goalIsSet or self.belief is None:
            rospy.logerr("Error[MCCModel.srv_update_fsc]: POMDP or belief are undefined.")
            return UpdateFSCResponse(False)

        # Do nothing until the number of updates is sufficient.
        if self.numberOfUpdates < self.numberOfUpdatesBeforeProvidingActions:
            return UpdateFSCResponse(False)

        # Determine which action corresponds to this goal. Do the same for the observation.
        actionX = int(np.sign(req.goal_x) * float(abs(req.goal_x) > MCC_GOAL_THRESHOLD))
        actionY = int(np.sign(req.goal_y) * float(abs(req.goal_y) > MCC_GOAL_THRESHOLD))
        action = (actionX, actionY)

        try:
            actionIndex = self.pomdp.actions.index(action)
        except ValueError:
            rospy.logerr("Error[MCCModel.srv_update_fsc]: Invalid action given: [%i, %i]." % (actionX, actionY))
            return UpdateFSCResponse(False)

        try:
            observationIndex = self.pomdp.observations.index(req.bump_observed)
        except ValueError:
            rospy.logerr("Error[MCCModel.srv_update_fsc]: Invalid observation given: %s." % (str(req.bump_observed)))
            return UpdateFSCResponse(False)

        # Attempt to update the FSC state by randomly selecting an action.
        try:
            self.belief = self.pomdp.belief_update(self.belief, actionIndex, observationIndex)
        except:
            rospy.logerr("Error[MCCModel.srv_update_fsc]: Failed to update belief.")
            return UpdateFSCResponse(False)

        # Finally, visualize the FSC state if desired.
        self.visualize_fsc_state()

        return UpdateFSCResponse(True)



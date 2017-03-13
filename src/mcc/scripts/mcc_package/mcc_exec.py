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

from mcc_model import *
from fsc import *


MCC_OCCUPANCY_GRID_OBSTACLE_THRESHOLD = 50
MCC_GOAL_THRESHOLD = 0.001


class MCCExec(object):
    """ The code which controls the robot following an MCC model's FSC. """

    def __init__(self, agent):
        """ The constructor for the MCCExec class.

            Parameters:
                agent   --  The unique agent name from the MCC.
        """

        # The MCC and other related information. Note: These are for *this* agent.
        self.mcc = MCC()
        self.fsc = None
        self.fscState = None
        self.agent = agent

        self.initialFSCStateIsSet = False
        self.goalIsSet = False
        self.algorithmIsInitialized = False

        # Information about the map for use by a path follower once our paths are published.
        self.mapWidth = 0
        self.mapHeight = 0
        self.mapOriginX = 0.0
        self.mapOriginY = 0.0
        self.mapResolution = 1.0

        # This is the number of x and y states that will be created using the map. Obstacle states
        # will, of course, be omitted.
        self.gridWidth = rospy.get_param(rospy.search_param('grid_width'))
        self.gridHeight = rospy.get_param(rospy.search_param('grid_height'))

        # Store if we performed the initial theta adjustment and the final goal theta adjustment
        self.performedInitialPoseAdjustment = False
        self.initialPoseAdjustmentTheta = 0.0

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

    def __del__(self):
        """ The deconstructor for the MCCExec class. """

        if self.algorithmIsInitialized:
            self.uninitializeAlgorithm()

    def initialize(self):
        """ Initialize the MCCExec class, mainly registering subscribers and services. """

        subOccupancyGridTopic = rospy.get_param(rospy.search_param('sub_occupancy_grid'))
        self.subOccupancyGrid = rospy.Subscriber(subOccupancyGridTopic,
                                                 OccupancyGrid,
                                                 self.sub_occupancy_grid)

        subMapPoseEstimateTopic = rospy.get_param(rospy.search_param('sub_map_pose_estimate'))
        self.subMapPoseEstimate = rospy.Subscriber(subMapPoseEstimateTopic,
                                                   PoseWithCovarianceStamped,
                                                   self.sub_map_pose_estimate)

        subMapNavGoalTopic = rospy.get_param(rospy.search_param('sub_map_nav_goal'))
        self.subMapNavGoal = rospy.Subscriber(subMapNavGoalTopic,
                                              PoseStamped,
                                              self.sub_map_nav_goal)

        pubModelUpdateTopic = rospy.get_param(rospy.search_param('model_update'))
        self.pubModelUpdate = rospy.Publisher(pubModelUpdateTopic, ModelUpdate, queue_size=10)

        srvGetActionTopic = rospy.get_param(rospy.search_param('get_action'))
        self.srvGetAction = rospy.Service(srvGetActionTopic,
                                          GetAction,
                                          self.srv_get_action)

        srvGetFSCStateTopic = rospy.get_param(rospy.search_param('get_fsc_state'))
        self.srvGetFSCState = rospy.Service(srvGetFSCStateTopic,
                                            GetFSCState,
                                            self.srv_get_fsc_state)

        srvUpdateFSCTopic = rospy.get_param(rospy.search_param('update_fsc'))
        self.srvUpdateFSC = rospy.Service(srvUpdateFSCTopic,
                                          UpdateFSC,
                                          self.srv_update_fsc)

        print("*///** %s ***" % (pubModelUpdateTopic))
        print("*///** %s ***" % (srvGetActionTopic))
        print("*///** %s ***" % (srvGetFSCStateTopic))
        print("*///** %s ***" % (srvUpdateFSCTopic))

    def update(self):
        """ Update the MCCExec object. """

        # These methods deal with the threading issue. Basically, the update below could be called
        # while the MCC itself is being modified in a different thread. This can easily be reproduced
        # by continually assigning new initial pose estimates and goals. Instead, however, we have
        # any subscriber callbacks assign a variable with the message. This message is then handled
        # as part of the main node's thread update call (here).
        self.handle_occupancy_grid_message()
        self.handle_map_pose_estimate_msg()
        self.handle_map_nav_goal_msg()

        # We only update once we have a valid MCC.
        if self.mcc is None or not self.initialFSCStateIsSet or not self.goalIsSet:
            return

        # If this is the first time the MCC has been ready to be updated, then
        # initialize necessary variables.
        if not self.algorithmIsInitialized:
            self.initialize_algorithm()

        #rospy.loginfo("Info[MCCExec.update]: Updating the policy.")

        # Note: There is no update anymore. It is solved offline...
        #result = self.mcc.update()

    def initialize_algorithm(self):
        """ Initialize the MCC algorithm. """

        if self.algorithmIsInitialized:
            rospy.logwarn("Warn[MCCExec.initialize_algorithm]: Algorithm is already initialized.")
            return

        # Note: This is kind of just to prevent execution, not actually using initial FSC and goal from clicking.
        if not self.initialFSCStateIsSet or not self.goalIsSet:
            rospy.logwarn("Warn[MCCExec.initialize_algorithm]: Initial FSC state or goal is not set yet.")
            return

        rospy.loginfo("Info[MCCExec.initialize_algorithm]: Initializing the algorithm.")

        # Load the policy for the initial and goal state selected.
        self.fsc = FSC(self.mcc, self.agent)
        self.fsc.load()

        # Setup the initial FSC state for this agent.
        self.fscState = self.fsc.Q[0]

        self.algorithmIsInitialized = True

    def uninitialize_algorithm(self):
        """ Uninitialize the MCC algorithm. """

        if not self.algorithmIsInitialized:
            rospy.logwarn("Warn[MCCExec.uninitialize_algorithm]: Algorithm has not been initialized.")
            return

        rospy.loginfo("Info[MCCExec.uninitialize_algorithm]: Uninitializing the algorithm.")

        self.fsc = None
        self.fscState = None

        self.algorithmIsInitialized = False

    def sub_occupancy_grid(self, msg):
        """ A subscriber for OccupancyGrid messages. This converges any 2d map
            into a set of MCC states. This is a static method to work as a ROS callback.

            Parameters:
                msg     --  The OccupancyGrid message data.
        """

        if self.occupancyGridMsg is None:
            self.occupancyGridMsg = msg

    def handle_occupancy_grid_message(self):
        """ A handler for OccupancyGrid messages. This converges any 2d map
            into a set of MCC states. This is a static method to work as a ROS callback.
        """

        if self.occupancyGridMsg is None:
            return
        msg = self.occupancyGridMsg

        rospy.loginfo("Info[MCCExec.sub_occupancy_grid]: Received map. Creating a new MCC.")

        # Remember map information.
        self.mapWidth = msg.info.width
        self.mapHeight = msg.info.height
        self.mapOriginX = msg.info.origin.position.x
        self.mapOriginY = msg.info.origin.position.y
        self.mapResolution = msg.info.resolution

        xStep = int(self.mapWidth / self.gridWidth)
        yStep = int(self.mapHeight / self.gridHeight)

        # TODO: Perhaps define the MCC here, and solve it? We'll leave that for future work.
        #self.mcc = MCC()

        # Un-/Re-initialize other helpful variables.
        self.initialFSCStateIsSet = False
        self.goalIsSet = False
        if self.algorithmIsInitialized:
            self.uninitialize_algorithm()

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
            pose is assigned, inducing an initial FSC state. This is a static method to work as a
            ROS callback.
        """

        if self.mapPoseEstimateMsg is None:
            return
        msg = self.mapPoseEstimateMsg

        if self.mcc is None:
            rospy.logwarn("Warn[MCCExec.sub_map_pose_estimate]: MCC has not yet been defined.")
            return

        rospy.loginfo("Info[MCCExec.sub_map_pose_estimate]: Received pose estimate. Assigning MCC initial FSC state.")

        # Setup the initial (theta) pose adjustment.
        roll, pitch, yaw = euler_from_quaternion([msg.pose.pose.orientation.x,
                                                  msg.pose.pose.orientation.y,
                                                  msg.pose.pose.orientation.z,
                                                  msg.pose.pose.orientation.w])
        self.initialPoseAdjustmentTheta = -yaw

        self.performedInitialPoseAdjustment = False

        self.uninitialize_algorithm()
        self.initialize_algorithm()

        # NOTE: Just set both...
        self.initialFSCStateIsSet = True
        self.goalIsSet = True

        self.mapPoseEstimateMsg = None

        self.pubModelUpdate.publish(ModelUpdate())

    def sub_map_nav_goal(self, msg):
        """ A subscriber for PoseStamped messages. This is called when a goal is provided,
            assigning the rewards for the MCC. This is a static method to work as a ROS callback.

            Parameters:
                msg     --  The OccupancyGrid message data.
        """

        if self.mapNavGoalMsg is None:
            self.mapNavGoalMsg = msg

    def handle_map_nav_goal_msg(self):
        """ A handler for PoseStamped messages. This is called when a goal is provided,
            assigning the rewards for the MCC. This is a static method to work as a ROS callback.
        """

        if self.mapNavGoalMsg is None:
            return
        msg = self.mapNavGoalMsg

        if self.mcc is None:
            rospy.logwarn("Warn[MCCExec.sub_map_nav_goal]: MCC has not yet been defined.")
            return

        self.uninitialize_algorithm()
        self.initialize_algorithm()

        # NOTE: Just set both...
        self.initialFSCStateIsSet = True
        self.goalIsSet = True

        self.mapNavGoalMsg = None

        self.pubModelUpdate.publish(ModelUpdate())

    def srv_get_action(self, req):
        """ This service returns an action based on the current FSC state, provided enough updates were done.

            Parameters:
                req     --  The service request as part of GetAction.

            Returns:
                The service response as part of GetAction.
        """

        if self.mcc is None or not self.initialFSCStateIsSet or not self.goalIsSet or self.fscState is None:
            rospy.logerr("Error[MCCExec.srv_get_action]: MCC or FSC state are undefined.")
            return GetActionResponse(False, 0.0, 0.0, 0.0)

        # Randomly select an action following the stochastic FSC.
        action = list(self.fsc.psi[self.fscState].values())[0]
        current = 0.0
        target = random.random()

        for iterAction, iterProbability in self.fsc.psi[self.fscState].items():
            current += iterProbability
            if current >= target:
                action = iterAction
                break

        rospy.loginfo("Info[MCCExec.srv_get_action]: Agent '%s' has selected action '%s'." % (self.agent, str(action)))

        # The relative goal is simply the relative location based on the "grid-ize-ation"
        # and resolution of the map. The goal theta is a bit harder to compute (estimate).
        goalX, goalY = action

        xSize = self.mapWidth / self.gridWidth
        ySize = self.mapHeight / self.gridHeight

        goalX *= xSize * self.mapResolution
        goalY *= ySize * self.mapResolution

        # If this is the first action we take, then we need to offset the goalX and goalY
        # as well as assign a goalTheta to properly setup the initial motion. Otherwise,
        # the adjustment required is simply 0; the path (action) follower will handle this.
        if not self.performedInitialPoseAdjustment:
            #goalX += self.initialPoseAdjustmentX
            #goalY += self.initialPoseAdjustmentY
            goalTheta = self.initialPoseAdjustmentTheta
            self.performedInitialPoseAdjustment = True
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

        if self.mcc is None or not self.initialFSCStateIsSet or not self.goalIsSet or self.fscState is None:
            rospy.logerr("Error[MCCExec.srv_get_fsc_state]: MCC or FSC state are undefined.")
            return GetFSCStateResponse("")

        # Print the FSC states for debug purposes.
        #rospy.loginfo("Info[MCCExec.srv_get_fsc_state]: Agent '%s' has FSC state '%s'." % (self.agent, self.fscState))

        return GetFSCStateResponse(str(self.fscState))

    def srv_update_fsc(self, req):
        """ This service updates the FSC based on a given action and observation.

            Parameters:
                req     --  The service request as part of UpdateFSC.

            Returns:
                The service response as part of UpdateFSC.
        """

        if self.mcc is None or not self.initialFSCStateIsSet or not self.goalIsSet or self.fscState is None:
            rospy.logerr("Error[MCCExec.srv_update_fsc]: MCC or FSC state are undefined.")
            return UpdateFSCResponse(False)

        # Determine which action corresponds to this goal. Do the same for the observation.
        actionX = int(np.sign(req.goal_x) * float(abs(req.goal_x) > MCC_GOAL_THRESHOLD))
        actionY = int(np.sign(req.goal_y) * float(abs(req.goal_y) > MCC_GOAL_THRESHOLD))
        action = (actionX, actionY)

        try:
            actionIndex = self.mcc.actions[self.agent].index(action)
        except ValueError:
            rospy.logerr("Error[MCCExec.srv_update_fsc]: Invalid action given: [%i, %i]." % (actionX, actionY))
            return UpdateFSCResponse(False)

        # Determine which observation corresponds to the request data.
        observation = req.bump_observed

        try:
            observationIndex = self.mcc.observations[self.agent].index(observation)
        except ValueError:
            rospy.logerr("Error[MCCExec.srv_update_fsc]: Invalid observation given: %s." % (str(req.bump_observed)))
            return UpdateFSCResponse(False)

        # Update the FSC state by randomly selecting a successor FSC state.
        successor = list(self.fsc.eta[self.fscState][action][observation].values())[0]
        current = 0.0
        target = random.random()

        for iterSuccessor, iterProbability in self.fsc.eta[self.fscState][action][observation].items():
            current += iterProbability
            if current >= target:
                successor = iterSuccessor
                break

        self.fscState = successor

        rospy.loginfo("Info[MCCExec.srv_update_fsc]: Agent '%s' has selected successor FSC state '%s'." % (self.agent, str(self.fscState)))

        return UpdateFSCResponse(True)



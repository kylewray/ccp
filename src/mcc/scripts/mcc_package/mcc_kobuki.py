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

from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from kobuki_msgs.msg import BumperEvent

from mcc.msg import *
from mcc.srv import *

import math
import numpy as np


class MCCKobuki(object):
    """ A class to control a Kobuki following an MCC policy. """

    def __init__(self):
        """ The constructor for the MCCKobuki class. """

        # These are the world-frame x, y, and theta values of the Kobuki. They
        # are updated as it moves toward the goal.
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.previousX = 0.0
        self.previousY = 0.0
        self.previousTheta = 0.0

        # Again, these are world-frame goal values. Once it arrives at the goal, it will
        # get a new action, which assigns a new goal.
        self.atGoal = False
        self.goalX = 0.0
        self.goalY = 0.0
        self.goalTheta = 0.0

        # These are relative-frame goal values received from a 'get_action' call. This node
        # passes this information back to the MCC model, which resolves what action it was.
        self.relGoalX = 0.0
        self.relGoalY = 0.0
        self.relGoalTheta = 0.0
        self.noActionStartTime = rospy.get_rostime()

        # Over time, the MCC's messages will do single-action relGoalTheta assignments. These
        # need to be tracked and accounted for over time as a permanent theta adjustment.
        self.permanentThetaAdjustment = 0.0

        # This bump/edge variable is assigned in the callback for the sensor. It is used in
        # the odometry callback to control behavior. Also, setup recovery variables.
        self.recovery = False
        self.recoveryX = 0.0
        self.recoveryY = 0.0

        # Setup the topics for the important services.
        mccModelNamespace = rospy.get_param("~mcc_exec_namespace", "/mcc_exec_node")
        self.subModelUpdateTopic = mccModelNamespace + "/model_update"
        self.srvGetActionTopic = mccModelNamespace + "/get_action"
        self.srvGetFSCStateTopic = mccModelNamespace + "/get_fsc_state"
        self.srvUpdateFSCTopic = mccModelNamespace + "/update_fsc"

        # The distance at which we terminate saying that we're at the goal,
        # in meters and radians, respectively.
        self.atPositionGoalThreshold = rospy.get_param("~at_position_goal_threshold", 0.05)
        self.atThetaGoalThreshold = rospy.get_param("~at_theta_goal_threshold", 0.05)
        self.recoveryDistanceThreshold = rospy.get_param("~recovery_distance_threshold", 0.1)

        # PID control variables.
        self.pidDerivator = 0.0
        self.pidIntegrator = 0.0
        self.pidIntegratorBounds = rospy.get_param("~pid_integrator_bounds", 0.05)

        # Load the gains for PID control.
        self.pidThetaKp = rospy.get_param("~pid_theta_Kp", 1.0)
        self.pidThetaKi = rospy.get_param("~pid_theta_Ki", 0.2)
        self.pidThetaKd = rospy.get_param("~pid_theta_Kd", 0.2)

        self.desiredVelocity = rospy.get_param("~desired_velocity", 0.2)

        # Remember the path.
        self.rawPath = list()
        self.lastPathPublishTime = rospy.get_rostime()

        # Finally, we create variables for the messages.
        self.started = False
        self.resetRequired = False
        self.modelReady = False

        self.subKobukiOdom = None
        self.subKobukiBump = None
        self.pubKobukiVel = None
        self.pubKobukiResetOdom = None
        self.pubPath = None

    def start(self):
        """ Start the necessary messages to operate the Kobuki. """

        if self.started:
            rospy.logwarn("Warn[MCCKobuki.start]: Already started.")
            return

        #rospy.sleep(15)

        self.subModelUpdate = rospy.Subscriber(self.subModelUpdateTopic,
                                               ModelUpdate,
                                               self.sub_model_update)

        subKobukiOdomTopic = rospy.get_param("~sub_kobuki_odom", "/odom")
        self.subKobukiOdom = rospy.Subscriber(subKobukiOdomTopic,
                                              Odometry,
                                              self.sub_kobuki_odom)

        subKobukiBumpTopic = rospy.get_param("~sub_kobuki_bump", "/evt_bump")
        self.subKobukiBump = rospy.Subscriber(subKobukiBumpTopic,
                                              BumperEvent,
                                              self.sub_kobuki_bump)

        pubKobukiVelTopic = rospy.get_param("~pub_kobuki_vel", "/cmd_vel")
        self.pubKobukiVel = rospy.Publisher(pubKobukiVelTopic, Twist, queue_size=32)

        pubKobukiResetOdomTopic = rospy.get_param("~pub_kobuki_reset_odom", "/cmd_reset_odom")
        self.pubKobukiResetOdom = rospy.Publisher(pubKobukiResetOdomTopic, Empty, queue_size=32)

        pubPathTopic = rospy.get_param("~pub_path", "/path")
        self.pubPath = rospy.Publisher(pubPathTopic, Path, queue_size=32)

        self.started = True

    def reset(self):
        """ Reset all of the variables that change as the robot moves. """

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Note: We do *not* reset these on a 'reset'. The odometers do not get reset, and these
        # are used to compute the delta and update (x, y, theta) above according to this difference.
        # Thus, we just leave these at whatever they are, all the time.
        #self.previousX = 0.0
        #self.previousY = 0.0
        #self.previousTheta = 0.0

        self.atGoal = False
        self.goalX = 0.0
        self.goalY = 0.0
        self.goalTheta = 0.0

        self.relGoalX = 0.0
        self.relGoalY = 0.0
        self.relGoalTheta = 0.0

        self.permanentThetaAdjustment = 0.0

        self.recovery = False
        self.recoveryX = 0.0
        self.recoveryY = 0.0

        self.pidDerivator = 0.0
        self.pidIntegrator = 0.0

        self.rawPath = list()
        self.lastPathPublishTime = rospy.get_rostime()

        self.started = False

        # Reset the robot's odometry.
        #if self.pubKobukiResetOdom is not None:
        #    self.pubKobukiResetOdom.publish(Empty())

        # Stop the robot's motion.
        if self.pubKobukiVel is not None: 
            control = Twist()
            self.pubKobukiVel.publish(control)

        self.resetRequired = False

    def update_state_from_odometry(self, msg):
        """ Update the state (location and orientation) from the odometry message.

            Parameters:
                msg     --  The Odometry message data.

            Returns:
                True if successful; False otherwise.
        """

        # Get the updated location and orientation from the odometry message.
        self.x += msg.pose.pose.position.x - self.previousX
        self.y += msg.pose.pose.position.y - self.previousY
        roll, pitch, yaw = euler_from_quaternion([msg.pose.pose.orientation.x,
                                                  msg.pose.pose.orientation.y,
                                                  msg.pose.pose.orientation.z,
                                                  msg.pose.pose.orientation.w])
        self.theta += yaw - self.previousTheta

        self.previousX = msg.pose.pose.position.x
        self.previousY = msg.pose.pose.position.y
        self.previousTheta = yaw

        #rospy.loginfo("ODOMETERS: %.3f %.3f %.3f" % (self.x, self.y, self.theta))

        #rospy.logwarn("[x, y, theta]: [%.4f, %.4f, %.4f]" % (self.x, self.y, self.theta))
        #rospy.logwarn("[goalX, goalY]: [%.4f, %.4f]" % (self.goalX, self.goalY))
        #rospy.logwarn("[relGoalX, relGoalY]: [%.4f, %.4f]" % (self.relGoalX, self.relGoalY))

    def sub_model_update(self, msg):
        """ The MCC model has changed. We need to reset everything.

            Parameters:
                msg     --  The ModelUpdate message data.
        """

        rospy.loginfo("Info[MCCKobuki.sub_model_update]: The model has been updated. Resetting.")

        self.resetRequired = True
        self.modelReady = True

    def sub_kobuki_odom(self, msg):
        """ Move the robot based on the MCCModel node's action.

            This gets the current action, moves the robot, calls the update FSC state service,
            gets the next action upon arriving at the next location, etc. It does not
            handle interrupts via a bump. It updates FSC state when not observing an obstacle.

            Parameters:
                msg     --  The Odometry message data.
        """

        if self.resetRequired:
            self.reset()

        if not self.modelReady:
            return

        if self.check_recovery(msg):
            self.move_recovery(msg)
        elif self.check_reached_goal(msg):
            self.move_to_goal(msg)

        self.publish_path(msg)

    def publish_path(self, msg):
        """ Record the path taken, but only at a certain rate.

            Parameters:
                msg     --  The Odometry message data.
        """

        publishRate = 0.2
        currentTime = rospy.get_rostime()

        if self.lastPathPublishTime.to_sec() + publishRate <= currentTime.to_sec():
            # Add to raw path with a timestamped pose from odometers.
            poseStamped = PoseStamped()
            poseStamped.header.frame_id = rospy.get_param("~sub_kobuki_odom", "/odom")
            poseStamped.header.stamp = currentTime
            poseStamped.pose = msg.pose.pose

            self.rawPath += [poseStamped]

            # Create and publish the path.
            path = Path()
            path.header.frame_id = rospy.get_param("~sub_kobuki_odom", "/odom")
            path.header.stamp = currentTime
            path.poses = self.rawPath

            self.pubPath.publish(path)

            self.lastPathPublishTime = currentTime

    def check_recovery(self, msg):
        """ Handle checking and recovering from a bump or edge detection.

            This moves the Kobuki back a small number of centimeters to prevent issues
            when rotating during the next action, or move it away from a cliff.

            Parameters:
                msg     --  The Odometry message data.

            Returns:
                True if a bump or edge is detected so movement away should be performed; False otherwise.
        """

        # If the robot is not in bump/edge recovery mode, then do not do recovery movement.
        if not self.recovery:
            return False

        # Compute the distance from when the bump was detected to the current location.
        errorX = self.recoveryX - self.x
        errorY = self.recoveryY - self.y
        distanceFromRecoveryLocation = math.sqrt(pow(errorX, 2) + pow(errorY, 2))

        # If the robot is not far enough away from the bump location, then we
        # still need to move backwards.
        if distanceFromRecoveryLocation < self.recoveryDistanceThreshold:
            return True

        # Otherwise, we are far enough away, or we simply did not detect a bump.
        # We are done recovery. Reset recovery variables.
        self.recovery = False
        self.recoveryX = 0.0
        self.recoveryY = 0.0

        # Perform an update to the MCC model; return on error. Note that if we got
        # here, then there *was* a bump or an edge.
        # TODO: Uncomment if you decide that bump is actually an observation in the MCC model.
        #if not self.update_mcc_model(msg, True):
        #    return False

        return True

    def check_reached_goal(self, msg):
        """ Handle checking and reaching a goal.

            This means getting a new action from the MCCModel and setting variables,
            as well as doing distance calculations.

            Parameters:
                msg     --  The Odometry message data.

            Returns:
                True if successful and movement should be performed; False otherwise.
        """

        # Allow for a no-action to actually wait for a second.
        noActionDuration = 5.0
        currentTime = rospy.get_rostime()

        if self.noActionStartTime.to_sec() + noActionDuration > currentTime.to_sec():
            return True

        # Compute the distance to the goal given the positions, as well as the theta goal.
        errorX = self.goalX - self.x
        errorY = self.goalY - self.y
        distanceToPositionGoal = math.sqrt(pow(errorX, 2) + pow(errorY, 2))
        distanceToThetaGoal = abs(self.goalTheta - self.theta)

        #rospy.loginfo("ERROR ---- %.4f %.4f" % (errorX, errorY))

        # If the robot is far from the goal, with no bump detected either, then do nothing.
        if ((distanceToPositionGoal >= self.atPositionGoalThreshold
                or (abs(self.relGoalX) < 0.01 and abs(self.relGoalY) < 0.01
                    and distanceToThetaGoal >= self.atThetaGoalThreshold))):
            return True

        # Perform an update to the MCC model; return on error. Note that if we got
        # here, then there *was not* a bump or an edge.
        if not self.update_mcc_model(msg, False, errorX, errorY):
            return False

        return True

    def update_mcc_model(self, msg, observedBumpOrEdge, adjustX=0.0, adjustY=0.0):
        """ Update the MCC Model given the observation.

            Parameters:
                msg                 --  The Odometry message data.
                observedBumpOrEdge  --  If a bump or edge was observed.
                adjustX             --  Optionally, adjust the x goal location. Default is 0.0.
                adjustY             --  Optionally, adjust the y goal location. Default is 0.0.

            Returns:
                True if successful; False otherwise.
        """

        # However, if it is close enough to the goal, then update the FSC state with
        # observing a bump or not. This may fail if not enough updates have been performed.
        rospy.wait_for_service(self.srvUpdateFSCTopic)
        try:
            srvUpdateFSC = rospy.ServiceProxy(self.srvUpdateFSCTopic, UpdateFSC)
            res = srvUpdateFSC(self.relGoalX, self.relGoalY, observedBumpOrEdge)
            if not res.success:
                rospy.logwarn("Error[MCCKobuki.update_mcc_model]: Failed to update FSC.")
                return False
        except rospy.ServiceException:
            rospy.logerr("Error[MCCKobuki.update_mcc_model]: Service exception when updating FSC.")
            return False

        # Now do a service request for the MCCModel to send the current action.
        rospy.wait_for_service(self.srvGetActionTopic)
        try:
            srvGetAction = rospy.ServiceProxy(self.srvGetActionTopic, GetAction)
            res = srvGetAction()
        except rospy.ServiceException:
            rospy.logerr("Error[MCCKobuki.update_mcc_model]: Service exception when getting action.")
            return False

        # This may fail if not enough updates have been performed.
        if not res.success:
            rospy.loginfo("Error[MCCKobuki.update_mcc_model]: No action was returned.")
            return False

        self.update_state_from_odometry(msg)

        #rospy.logwarn("Action: [%.1f, %.1f, %.3f]" % (res.goal_x, res.goal_y, res.goal_theta))
        self.relGoalX = res.goal_x
        self.relGoalY = res.goal_y
        self.relGoalTheta = res.goal_theta

        if abs(res.goal_x) < 0.01 and abs(res.goal_y) < 0.01:
            self.noActionStartTime = rospy.get_rostime()

        self.permanentThetaAdjustment += self.relGoalTheta

        # Importantly, we rotate the relative goal by the relative theta provided!
        xyLength = math.sqrt(pow(self.relGoalX, 2) + pow(self.relGoalY, 2))
        xyTheta = math.atan2(self.relGoalY, self.relGoalX)
        relGoalAdjustedX = xyLength * math.cos(xyTheta + self.permanentThetaAdjustment)
        relGoalAdjustedY = xyLength * math.sin(xyTheta + self.permanentThetaAdjustment)

        # We need to translate the goal location given by srvGetAction to the world-frame.
        # They are provided as a relative goal. Theta, however, is given in 'world-frame'
        # kinda, basically because it is not in the MCCModel's state space.
        self.goalX = self.x + relGoalAdjustedX + adjustX
        self.goalY = self.y + relGoalAdjustedY + adjustY
        self.goalTheta = np.arctan2(self.goalY - self.y, self.goalX - self.x)

        return True

    def move_recovery(self, msg):
        """ Move away from a wall or edge backwards using the relevant Kobuki messages.

            Parameters:
                msg     --  The Odometry message data.

            Returns:
                True if successful; False otherwise.
        """

        self.update_state_from_odometry(msg)

        control = Twist()
        control.linear.x = -self.desiredVelocity

        self.pubKobukiVel.publish(control)

        return True

    def move_to_goal(self, msg):
        """ Move toward the goal using the relevant Kobuki messages.

            Parameters:
                msg     --  The Odometry message data.

            Returns:
                True if successful; False otherwise.
        """

        self.update_state_from_odometry(msg)

        control = Twist()

        # If close to the goal, then do nothing. Otherwise, drive based on normal control. However,
        # we only update the distance if there is no more relative theta adjustment required.
        distanceToPositionGoal = math.sqrt(pow(self.x - self.goalX, 2)
                                         + pow(self.y - self.goalY, 2))
        if distanceToPositionGoal < self.atPositionGoalThreshold:
            control.linear.x = 0.0
        else:
            # This assigns the desired set-point for speed in meters per second.
            control.linear.x = self.desiredVelocity

        #rospy.logwarn("Distance to Goal: %.4f" % (distanceToPositionGoal))

        # Compute the new goal theta based on the updated (noisy) location of the robot.
        self.goalTheta = np.arctan2(self.goalY - self.y, self.goalX - self.x)

        #rospy.logwarn("Goal Theta: %.4f" % (self.goalTheta))

        error = self.goalTheta - self.theta
        if error > math.pi:
            self.goalTheta -= 2.0 * math.pi
            error -= 2.0 * math.pi
        if error < -math.pi:
            self.goalTheta += 2.0 * math.pi
            error += 2.0 * math.pi

        #rospy.logwarn("Theta Error: %.4f" % (abs(error)))

        if abs(error) < self.atThetaGoalThreshold:
            control.angular.z = 0.0
        else:
            valP = error * self.pidThetaKp

            self.pidIntegrator += error
            self.pidIntegrator = np.clip(self.pidIntegrator,
                                         -self.pidIntegratorBounds,
                                         self.pidIntegratorBounds)
            valI = self.pidIntegrator * self.pidThetaKi

            self.pidDerivator = error - self.pidDerivator
            self.pidDerivator = error
            valD = self.pidDerivator * self.pidThetaKd

            # This assigns the desired set-point for relative angle.
            control.angular.z = valP + valI + valD

        self.pubKobukiVel.publish(control)

        return True

    def sub_kobuki_bump(self, msg):
        """ This method checks for sensing a bump.

            Parameters:
                msg     --  The BumperEvent message data.
        """

        if msg.state == BumperEvent.PRESSED:
            self.recovery = True
            self.recoveryX = self.x
            self.recoveryY = self.y


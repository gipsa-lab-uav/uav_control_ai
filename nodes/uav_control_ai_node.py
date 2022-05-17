#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from mavros_msgs.msg import AttitudeTarget, State
from mavros_msgs.srv import CommandBool, SetMode, ParamPush
from geometry_msgs.msg import TwistStamped, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Odometry

class UavControlAiNode:
    def __init__(self, is_simu, node_name='uav_control_ai_node'):
        # Node initialization
        rospy.init_node(node_name, anonymous=True)

        # Subscriber definition
        rospy.Subscriber("/mavros/desired_trajectory", JointTrajectory, self.callback_trajectory)
        rospy.Subscriber("/mavros/state", State, self.callback_state_acquire)

        if is_simu:
            # Simulation (TO DO : replace odom_gt --> vision/pose)
            # rospy.Subscriber("/odom_gt", Odometry, self.callback_vicon);
            rospy.Subscriber("/mavros/vision_pose/pose", PoseStamped, self.callback_vicon)
        else:
            # Manipulation
            rospy.Subscriber("/mavros/local_position/odom", Odometry, self.callback_vicon)

        # Publisher definition
        # self.cmd_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
        self.ref_pub = rospy.Publisher("mavros/statesReference", JointTrajectoryPoint, queue_size=10)

        self.state = State()
        self.jtrj = JointTrajectory()
        self.drone_state = State()
        self.is_mes_ok = False
        self.is_trj_started = False

    ## Callback Functions
    def callback_trajectory(self, msg):
        self.jtrj = msg

    def callback_state_acquire(self, msg):
        self.drone_state = msg

    def callback_vicon(self, msg):
        if not self.is_mes_ok :
            self.is_mes_ok = True

    def getNextTrajectoryPoint(self, time):
        # Return trajectory point at givent rostime time
        i = 0
        trajectoryTime = self.jtrj.header.stamp.to_sec()
        # print("Traj time : ",trajectoryTime)
        # Find the next trajectory point with respect to time
        for point in self.jtrj.points:
            if (trajectoryTime + point.time_from_start.to_sec()  > time):
                break
            i = i+1
        if self.jtrj.points ==[]:
            out = []
        else:
            out = self.jtrj.points[i]

        if not self.is_trj_started and (i > 1) :
            self.is_trj_started = True

        return out

    def set_arm(self, arm, timeout):
        """arm: True to arm or False to disarm, timeout(int): seconds"""
        rospy.loginfo("setting FCU arm: {0}".format(arm))
        old_arm = self.state.armed
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        arm_set = False
        for i in range(timeout * loop_freq):
            if self.state.armed == arm:
                arm_set = True
                rospy.loginfo("set arm success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            elif arm_set == False:
                try:
                    res = self.set_arming_srv(arm)
                    # arm_set = True
                    # rospy.loginfo("Success to send arm command")
                    if not res.success:
                        arm_set = False
                        rospy.logerr("failed to send arm command")
                    else:
                        arm_set = True
                        rospy.loginfo("Success to send arm command")
                        rospy.loginfo("[uav_control_ai] Drone Armed.")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

    def set_mode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        old_mode = self.state.mode
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        mode_set = False
        for i in range(timeout * loop_freq):
            if self.state.mode == mode:
                mode_set = True
                rospy.loginfo("set mode success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    rospy.loginfo("[uav_control_ai] Offboard mode.")
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)
            try:
                rate.sleep()
            except rospy.ROSException as e:
                self.fail(e)

if __name__ == '__main__':
    #***************************** Initialization *****************************#

    # Get parameters
    param = rospy.get_param('uav_control_ai_node')
    if param.get("name") == "simu" :
        is_simu = True
    else:
        is_simu = False

    # Create and init node
    UavCtrAiN = UavControlAiNode(is_simu)

    rate = rospy.Rate(100)

    # Wait for the drone to connect
    rospy.loginfo("[uav_control_ai] Waiting for drone connection ...")
    while not rospy.is_shutdown() and UavCtrAiN.drone_state.connected:
        rate.sleep()
    rospy.loginfo("[uav_control_ai] Drone connected.")

    # Wait for the first measurement callback
    rospy.loginfo("[uav_control_ai] Waiting for position measurement callback ...")
    while not rospy.is_shutdown() and not UavCtrAiN.is_mes_ok:
        rate.sleep()
    rospy.loginfo("[uav_control_ai] Position measurement callback ok.")

    # Log info
    rospy.loginfo("[uav_control_ai] Real test: enable the offboard control and arm the drone with the remote controller.")
    rospy.loginfo("[uav_control_ai] Estimators are reset as long as drone is not armed, offboard control is disabled & no trajectory is being broadcasted.")
    rospy.loginfo("[uav_control_ai] Have a safe flight.")

    # Start ROS services
    service_timeout = 1
    rospy.loginfo("[uav_control_ai] Waiting for ROS services.")
    try:
        rospy.wait_for_service('mavros/cmd/arming', service_timeout)
        rospy.wait_for_service('mavros/set_mode', service_timeout)
    except rospy.ROSException:
        rospy.fail("[uav_control_ai] Failed to connect to services")

    UavCtrAiN.set_arming_srv = rospy.ServiceProxy('mavros/cmd/arming',
                                                 CommandBool)
    UavCtrAiN.set_mode_srv = rospy.ServiceProxy('mavros/set_mode', SetMode)


    #************************** Publishing References *************************#

    while not rospy.is_shutdown():
        # UavCtrAiN.ref_pub.publish(UavCtrAiN.getNextTrajectoryPoint(rospy.get_time()))
        if is_simu:

            # Start offboard
            if UavCtrAiN.drone_state.mode != "OFFBOARD":
                UavCtrAiN.set_mode("OFFBOARD", 1)

            elif UavCtrAiN.drone_state.armed != True :
                # Drone arming
                UavCtrAiN.set_arm(True, 1)

        if UavCtrAiN.drone_state.mode == "OFFBOARD" and UavCtrAiN.drone_state.armed == True:
            # print('Cest bon')
            nextpoint = UavCtrAiN.getNextTrajectoryPoint(rospy.get_time())
            if not nextpoint == []:
                UavCtrAiN.ref_pub.publish(nextpoint)


        rate.sleep()

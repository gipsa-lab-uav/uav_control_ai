#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from tensorflow.keras.models  import load_model
from mavros_msgs.msg import AttitudeTarget, Thrust
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TwistStamped, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Odometry
from uav_control_ai.msg import StateCommandData
from sensor_msgs.msg import Imu, BatteryState
import rotations_tools as rt
import os

class AiNode:
    def __init__(self, param, node_name='ai_node'):
        # Node initialization
        rospy.init_node(node_name, anonymous=True)

        # Get Flight Parameters
        if param.get("name") == "simu" :
            self.is_simu = True
        else:
            self.is_simu = False

        self.mass = param.get("mass")
        self.hoverComp = param.get("hoverCompensation")
        self.ctrl_rate_freq = param.get("ctrl_rate_freq")
        self.ctrl_freq = param.get("ctrl_freq")
        self.ctr_type = param.get("ctr_type")
        self.ctr_gains = self.get_gains(param)

        # Get DNN model
        if "ai" in self.ctr_type:
            # print(os.path.dirname(os.path.abspath(__file__)))
            nn_mdl = os.path.dirname(os.path.abspath(__file__))+param.get("nn_model_path")
            self.network_acc_lin_mdl = load_model(nn_mdl)

        self.mes = 0
        self.trj_started = False

        # Subscriber definition
        rospy.Subscriber("/mavros/desired_trajectory", JointTrajectory, self.callback_trajectory)
        rospy.Subscriber("/mavros/statesReference", JointTrajectoryPoint, self.callback_ref)
        rospy.Subscriber("/mavros/imu/data", Imu, self.callback_imu)
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self.callback_odom)
        rospy.Subscriber("/mavros/battery", BatteryState, self.callback_battery)

        if self.is_simu:
            # Simulation
            # rospy.Subscriber("/odom_gt", Odometry, self.callback_vicon)
            rospy.Subscriber("/mavros/vision_pose/pose", PoseStamped, self.callback_vicon)
        else:
            # Manipulations
            rospy.Subscriber("/mavros/vision_pose/pose", PoseStamped, self.callback_vicon)

        # Publisher definition
        self.pub_cmd = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
        self.pub_error = rospy.Publisher("mavros/acceleration_error", Vector3, queue_size=10)
        self.pub_data = rospy.Publisher("uav_control_ai/statecommanddata", StateCommandData, queue_size=10)

        # Attitude target
        self.attitude_trgt = AttitudeTarget()
        self.attitude_trgt.type_mask = 0b10000000

        self.attitude_trgt_ai = AttitudeTarget()
        self.attitude_trgt_ai.type_mask = 0b10000000

        # Init StateCommandData msg
        self.state_command = StateCommandData()
        self.state_command.positions = [0, 0, 0]
        self.state_command.positions_vicon = [0, 0, 0]
        self.state_command.positions_linear_des = [0, 0, 0]
        self.state_command.positions_ref = [0, 0, 0, 0]
        self.state_command.velocities_linear = [0, 0, 0]
        self.state_command.velocities_linear_ref = [0, 0, 0]
        self.state_command.euler_angles = [0, 0, 0]
        self.state_command.battery = 0
        self.state_command.corr = [0, 0, 0]

        # Discrete Model
        self.discrete_x = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
        self.discrete_u = np.array([[0, 0, 0, 0]]).T
        self.Ad_PX4 = np.array([[1, 0, 0, 0, 0.0005, 0, 0.0100, 0, 0],
                                [0, 1, 0, -0.0005,         0,         0,         0,    0.0100,         0],
                                [0, 0, 1,       0,        0,         0,         0,         0,    0.0100],
                                [0, 0, 0,  1.0000,         0,         0,         0,         0,         0],
                                [0, 0, 0,       0,    1.0000,         0,         0,         0,         0],
                                [0, 0, 0,       0,         0,    1.0000,         0,         0,         0],
                                [0, 0, 0,       0,    0.0981,         0 ,   1.0000,         0 ,        0],
                                [0, 0, 0, -0.0981,         0,         0,         0,    1.0000,         0],
                                [0, 0, 0,       0,         0,         0,         0,         0,    1.0000]])
        self.Bd_PX4 = np.array([[ 0,    0.0000,         0,         0],
                                [-0.0000,         0,         0,         0],
                                [0,         0,         0,    0.0001],
                                [0.0100,         0,         0,         0],
                                [0,    0.0100,         0,         0],
                                [0,         0,   0.0100,         0],
                                [0,    0.0005,         0,         0],
                                [-0.0005,         0,        0,         0],
                                [0,         0,         0,    0.0181]])
        self.error = np.array([0,0,0])
        self.error_k = np.array([0,0,0])

    @staticmethod
    def get_gains(param):
        if param.get("ctr_type") in ['clin', 'clin_ai']:
            ctr_gains = [param.get("clin_kx"), param.get("clin_kvx"),
                        param.get("clin_ky"), param.get("clin_kvy"),
                        param.get("clin_kz"), param.get("clin_kvz"),
                        param.get("clin_kangle"), param.get("clin_ff")]
        else:
            ctr_gains = []

        return ctr_gains

    #*************************** Callback Functions ***************************#
    def callback_trajectory(self, msg):
        # Callback tto tell if trajectory is started
        self.trj_started = True

    def callback_battery(self, msg):
        # Store battery voltage
        self.state_command.battery = msg.voltage

    def callback_imu(self, msg):
        # Acceleratiosn from IMU
        self.state_command.accelerations_linear = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

    def callback_vicon(self, msg):
        # Store positions from Vicon
        if self.is_simu:
            # Simulation
            # self.state_command.positions_vicon = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
            self.state_command.positions_vicon = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        else:
            self.state_command.positions_vicon = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

    def callback_odom(self, msg):
        # Store positions
        self.state_command.positions = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]

        # Store linear velocities
        self.state_command.velocities_linear = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]

        # Store Euler angles
        # Quaternion to euler angles
        q = msg.pose.pose.orientation;

        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        self.state_command.euler_angles[0] = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        if (np.abs(sinp) >= 1):
            self.state_command.euler_angles[1] = np.copysign(np.pi/ 2, sinp)
        else:
            self.state_command.euler_angles[1] = np.arcsin(sinp)

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)

        self.state_command.euler_angles[2] = np.arctan2(siny_cosp, cosy_cosp)

        # Store linear velocities
        self.state_command.velocities_angular = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]

    def callback_ref(self, msg):
        # Store positions & velocities references
        self.state_command.positions_ref = msg.positions
        self.state_command.velocities_linear_ref = msg.velocities

    #*************************** Controller Functions ***************************#
    @staticmethod
    def clamp(value, lowBound, upperBound):
        # Saturation function
        if value < lowBound:
            value = lowBound
        elif value > upperBound:
            value = upperBound
        return value

    @staticmethod
    def wrap(x, low, high):
        # Wrap function
        if (low <= x and x < high):
            value = x
        else:
            range = high - low
            inv_range = 1.0 / range
            num_wraps = np.floor((x - low) * inv_range)
            value = x - range * num_wraps
        return value

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        # Euler angles to Quaternion
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

    def wrap_error_yaw(self, angles, angles_ref):
        def wrap(x, low, high):
            if (low <= x and x < high):
                value = x
            else:
                range = high - low
                inv_range = 1.0 / range
                num_wraps = np.floor((x - low) * inv_range)
                value = x - range * num_wraps
            return value
        error = angles - angles_ref
        angles[2] = self.wrap(angles[2], -np.pi, np.pi)
        error[2] = self.wrap(angles[2] - angles_ref[2], -np.pi, np.pi)
        return error

    def start_controller_cascaded(self):
        if self.ctr_type in ['clin', 'clin_ai']:
            rospy.Timer(rospy.Duration(1.0 / self.ctrl_freq), self.cascaded_linear_state_feedback)

        rospy.spin()


    ############################################################################
    ######################## Cascaded Linear Controller ########################
    ############################################################################

    def cascaded_linear_state_feedback(self, event=None):

        #************************ Position Speed Loop *************************#
        # Get positions & speed
        # x = self.state_command.positions[0]
        # y = self.state_command.positions[1]
        # z = self.state_command.positions[2]

        x = self.state_command.positions_vicon[0]
        y = self.state_command.positions_vicon[1]
        z = self.state_command.positions_vicon[2]

        x_dot = self.state_command.velocities_linear[0]
        y_dot = self.state_command.velocities_linear[1]
        z_dot = self.state_command.velocities_linear[2]

        phi = self.state_command.euler_angles[0]
        theta = self.state_command.euler_angles[1]
        psi = self.state_command.euler_angles[2]

        speed_rot = np.matmul(rt.rot_m(psi, theta, phi,'ZYX'), np.array([[x_dot, y_dot, z_dot]]).T)

        x_dot = speed_rot[0][0]
        y_dot = speed_rot[1][0]
        z_dot = speed_rot[2][0]

        with_ff = self.ctr_gains[7]
        state_ref = np.append(np.asarray(self.state_command.positions_ref[0:3]), with_ff*np.asarray(self.state_command.velocities_linear_ref[0:3]))

        state_pos_speed = np.array([x, y, z, x_dot, y_dot, z_dot])

        error_pos_speed = state_pos_speed - state_ref

        kx = self.ctr_gains[0]
        kvx = self.ctr_gains[1]
        ky = self.ctr_gains[2]
        kvy = self.ctr_gains[3]
        kz = self.ctr_gains[4]
        kvz = self.ctr_gains[5]
        kangle = self.ctr_gains[6]

        phi_ref = -ky*error_pos_speed[1]-kvy*error_pos_speed[4]
        theta_ref = -kx*error_pos_speed[0]-kvx*error_pos_speed[3]
        psi_ref = self.state_command.positions_ref[3]

        # cmd =  phi_ref, theta_ref, psi_ref, thrust_ref
        angles_ref = np.array([phi_ref, theta_ref, psi_ref])

        thrust_ref = -kz*error_pos_speed[2]-kvz*error_pos_speed[5]

        #*************************** Attitude Loop ****************************#

        # Use AI Correction Term or not
        if self.ctr_type == 'clin_ai':
            thrust_ref, angles_ref = self.correction_cascaded(thrust_ref, angles_ref)
            self.state_command.corr[0] = thrust_ref
            self.state_command.corr[1] = angles_ref[0]
            self.state_command.corr[2] = angles_ref[1]


        Rpsi = np.array([[np.cos(psi), np.sin(psi)],
                         [-np.sin(psi), np.cos(psi)]])

        angles_ref_rot = np.matmul(Rpsi, np.array([angles_ref[0:2]]).T)


        # Rpsi = np.array([[np.cos(psi), -np.sin(psi)],
        #                  [np.sin(psi), np.cos(psi)]])
        # angles_ref_rot = np.matmul(np.linalg.inv(Rpsi), np.array([angles_ref[0:2]]).T)
        angles_ref[0] = angles_ref_rot[0][0]
        angles_ref[1] = angles_ref_rot[1][0]
        angles_ref[2] = psi_ref

        state_attitude = np.array([phi, theta, psi])
        erreur_attitude = -self.wrap_error_yaw(state_attitude, angles_ref)

        command_attitude = np.array([kangle*erreur_attitude[0], kangle*erreur_attitude[1], kangle*erreur_attitude[2]])

        # Compute Desired Linear Behavior
        if self.trj_started == 0:
            self.discrete_xk = np.array([[x, y, z, phi, theta, psi, x_dot, y_dot, z_dot]]).T
        else:
            xk = np.array([self.discrete_xk[0,0], self.discrete_xk[1,0], self.discrete_xk[2,0],
            self.discrete_xk[6,0], self.discrete_xk[7,0], self.discrete_xk[8,0]])

            error_pos_speed_k = xk - state_ref

            phi_ref_k = -ky*error_pos_speed_k[1]-kvy*error_pos_speed_k[4]
            theta_ref_k = -kx*error_pos_speed_k[0]-kvx*error_pos_speed_k[3]
            psi_ref_k = self.state_command.positions_ref[3]

            angles_ref_k = np.array([phi_ref_k, theta_ref_k, psi_ref_k])
            thrust_ref_k = -kz*error_pos_speed_k[2]-kvz*error_pos_speed_k[5]
            erreur_attitude_k = angles_ref_k - self.discrete_xk[3:6].reshape(-1)
            command_attitude_k = np.array([kangle*erreur_attitude_k[0], kangle*erreur_attitude_k[1], kangle*erreur_attitude_k[2] ])

            # Linear expected behavior
            # self.discrete_xk = np.array([[x, y, z, phi, theta, psi, x_dot, y_dot, z_dot]]).T
            self.discrete_uk = np.array([[command_attitude_k[0], command_attitude_k[1], command_attitude_k[2], thrust_ref_k]]).T

            # xk+1 = Ad * xk + Bd * uk
            self.discrete_xkp1 = np.matmul(self.Ad_PX4, self.discrete_xk) + np.matmul(self.Bd_PX4,self.discrete_uk)

            # Store in msg
            self.state_command.positions_linear_des[0] = self.discrete_xkp1[0, 0]
            self.state_command.positions_linear_des[1] = self.discrete_xkp1[1, 0]
            self.state_command.positions_linear_des[2] = self.discrete_xkp1[2, 0]

            # Update for next loop
            self.discrete_xk = self.discrete_xkp1

        # Euler to Quaternion
        quat = Quaternion()
        qx, qy, qz, qw = self.euler_to_quaternion(phi_ref, theta_ref, psi_ref)
        quat.x = qx
        quat.y = qy
        quat.z = qz
        quat.w = qw

        # Create Vector3 for publishing
        body_rate = Vector3()
        body_rate.x = command_attitude[0]
        body_rate.y = command_attitude[1]
        body_rate.z = command_attitude[2]

        # Ensure to disable attitude control
        self.attitude_trgt.type_mask = 0b10000000

        # Fill msg for publisher
        self.attitude_trgt.body_rate = body_rate
        # self.attitude_trgt.orientation = quat
        self.attitude_trgt.thrust = self.clamp(thrust_ref * self.hoverComp/ (self.mass * 9.81)+self.hoverComp,0.0,1.0)

        # Publishing cmd =  p_ref, q_ref, r_ref, thrust_ref
        self.pub_cmd.publish(self.attitude_trgt)

        # Publishing data at max freq
        self.state_command.time = rospy.get_time()
        self.state_command.thrust_cmd = self.attitude_trgt.thrust
        self.state_command.angular_velocities_cmd = [body_rate.x, body_rate.y, body_rate.z]
        self.pub_data.publish(self.state_command)


    #************************* AI Correction Functions *************************#
    def correction_cascaded(self, thrust_ref, angles_ref, batt=1):
        # Compute delta(x, u)
        delta_acc_lin = self.network_acc_lin_mdl(np.array([self.network_input(with_batt=batt), ]), training=False).numpy()

        errx = delta_acc_lin[0][0]
        erry = delta_acc_lin[0][1]
        errz = delta_acc_lin[0][2]

        corr_thrust = -errz*self.mass
        corr_phi = erry/9.81
        corr_theta = -errx/9.81

        new_thrust = thrust_ref + corr_thrust
        new_angles_ref = angles_ref + np.array([corr_phi, corr_theta, 0])
        return new_thrust, new_angles_ref

    def network_input(self, with_batt=0):
        # Create linear accelerations network inputs
        if with_batt==0:
            net_input = [None]*11
        else:
            net_input = [None]*12
            net_input[11] = self.state_command.battery
        net_input[0] = np.cos(self.state_command.euler_angles[0])
        net_input[1] = np.cos(self.state_command.euler_angles[1])
        net_input[2] = np.cos(self.state_command.euler_angles[2])
        net_input[3] = np.sin(self.state_command.euler_angles[0])
        net_input[4] = np.sin(self.state_command.euler_angles[1])
        net_input[5] = np.sin(self.state_command.euler_angles[2])
        net_input[6] = self.state_command.velocities_linear[0]
        net_input[7] = self.state_command.velocities_linear[1]
        net_input[8] = self.state_command.velocities_linear[2]
        net_input[9] = self.attitude_trgt.thrust
        net_input[10] = self.state_command.positions_vicon[2]

        return net_input

# Main ai_node
if __name__ == '__main__':
    try:
        param = rospy.get_param('uav_control_ai_node')
        ai_object = AiNode(param)

        if ai_object.ctr_type in ['clin', 'clin_ai',] :
            if ai_object.ctr_type == 'clin':
                rospy.loginfo("[ai_node] STARTING LINEAR CASCADED STATE FEEDBACK")
            elif ai_object.ctr_type == 'clin_ai':
                rospy.loginfo("[ai_node] STARTING LINEAR CASCADED CONTROL WITH AI CORRECTION")
            ai_object.start_controller_cascaded()
        else:
            rospy.loginfo("[ai_node] INEXISTING CONTROL TYPE")

    except rospy.ROSInterruptException:
        rospy.loginfo("[ai_node] FAIL STARTING CONTROLLER")

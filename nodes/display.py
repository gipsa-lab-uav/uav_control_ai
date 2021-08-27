#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import matplotlib.pyplot as plt
from Tkinter import TclError

import tf.transformations
from nav_msgs.msg import Odometry
from mavros_msgs.msg import AttitudeTarget
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class Display:
    def __init__(self):
        self.node_name = 'display_node'
        self.command_sub = 'mavros/setpoint_raw/attitude'
        self.position_sub = 'mavros/local_position/odom'
        self.reference_sub = 'mavros/statesReference'
        self.estimated_sub = 'mavros/statesEstimated'

        self.time_window = 15  # [s]
        self.window_10 = self.time_window * 10
        self.window_30 = self.time_window * 30
        self.window_100 = self.time_window * 100
        self.x_10 = np.linspace(0, self.time_window, self.window_10)
        self.x_30 = np.linspace(0, self.time_window, self.window_30)
        self.x_100 = np.linspace(0, self.time_window, self.window_100)

        # Attitude commands published at 100Hz
        self.rol = np.zeros(self.x_100.size)
        self.pit = np.zeros(self.x_100.size)
        self.yaw = np.zeros(self.x_100.size)
        self.thr = np.zeros(self.x_100.size)

        # EKF fusion position published at 100Hz
        self.x_measured = np.zeros(self.x_30.size)
        self.y_measured = np.zeros(self.x_30.size)
        self.z_measured = np.zeros(self.x_30.size)

        # Reference points published at 100Hz
        self.x_reference = np.zeros(self.x_100.size)
        self.y_reference = np.zeros(self.x_100.size)
        self.z_reference = np.zeros(self.x_100.size)

        self.vx_reference = np.zeros(self.x_100.size)
        self.vy_reference = np.zeros(self.x_100.size)
        self.vz_reference = np.zeros(self.x_100.size)

        self.ax_reference = np.zeros(self.x_100.size)
        self.ay_reference = np.zeros(self.x_100.size)
        self.az_reference = np.zeros(self.x_100.size)

        # Estimated points published at 100Hz
        self.x_estimated = np.zeros(self.x_100.size)
        self.y_estimated = np.zeros(self.x_100.size)
        self.z_estimated = np.zeros(self.x_100.size)

        self.ya_estimated = np.zeros(self.x_100.size)

        self.vx_estimated = np.zeros(self.x_100.size)
        self.vy_estimated = np.zeros(self.x_100.size)
        self.vz_estimated = np.zeros(self.x_100.size)

    def attitudeTargetCallback(self, attitude):
        q = (attitude.orientation.x, attitude.orientation.y, attitude.orientation.z, attitude.orientation.w)
        euler = tf.transformations.euler_from_quaternion(q)

        self.rol = np.append(self.rol, euler[0])[-self.window_100:]
        self.pit = np.append(self.pit, euler[1])[-self.window_100:]
        self.yaw = np.append(self.yaw, euler[2])[-self.window_100:]
        self.thr = np.append(self.thr, attitude.thrust)[-self.window_100:]

    def positionCallback(self, odometry):
        self.x_measured = np.append(self.x_measured, odometry.pose.pose.position.x)[-self.window_30:]
        self.y_measured = np.append(self.y_measured, odometry.pose.pose.position.y)[-self.window_30:]
        self.z_measured = np.append(self.z_measured, odometry.pose.pose.position.z)[-self.window_30:]

    def referenceCallback(self, trajectoryPoint):
        positions = trajectoryPoint.positions
        velocities = trajectoryPoint.velocities
        accelerations = trajectoryPoint.accelerations

        self.x_reference = np.append(self.x_reference, positions[0])[-self.window_100:]
        self.y_reference = np.append(self.y_reference, positions[1])[-self.window_100:]
        self.z_reference = np.append(self.z_reference, positions[2])[-self.window_100:]

        self.vx_reference = np.append(self.vx_reference, velocities[0])[-self.window_100:]
        self.vy_reference = np.append(self.vy_reference, velocities[1])[-self.window_100:]
        self.vz_reference = np.append(self.vz_reference, velocities[2])[-self.window_100:]

        self.ax_reference = np.append(self.ax_reference, accelerations[0])[-self.window_100:]
        self.ay_reference = np.append(self.ay_reference, accelerations[1])[-self.window_100:]
        self.az_reference = np.append(self.az_reference, accelerations[2])[-self.window_100:]

    def estimatedCallback(self, trajectoryPoint):
        positions = trajectoryPoint.positions
        velocities = trajectoryPoint.velocities

        self.x_estimated = np.append(self.x_estimated, positions[0])[-self.window_100:]
        self.y_estimated = np.append(self.y_estimated, positions[1])[-self.window_100:]
        self.z_estimated = np.append(self.z_estimated, positions[2])[-self.window_100:]

        self.ya_estimated = np.append(self.ya_estimated, positions[3])[-self.window_100:]

        self.vx_estimated = np.append(self.vx_estimated, velocities[0])[-self.window_100:]
        self.vy_estimated = np.append(self.vy_estimated, velocities[1])[-self.window_100:]
        self.vz_estimated = np.append(self.vz_estimated, velocities[2])[-self.window_100:]

    def start(self):
        rospy.init_node(self.node_name, anonymous=True)

        rospy.Subscriber(self.command_sub, AttitudeTarget, self.attitudeTargetCallback)
        rospy.Subscriber(self.position_sub, Odometry, self.positionCallback)
        rospy.Subscriber(self.reference_sub, JointTrajectoryPoint, self.referenceCallback)
        rospy.Subscriber(self.estimated_sub, JointTrajectoryPoint, self.estimatedCallback)

        rospy.Rate(10)

        plt.ion()

        fig = plt.figure(figsize=(14, 12))

        ax_thr = fig.add_subplot(321)
        ax_thr.set_title('Thrust command')
        ax_thr.legend('thrust')
        ax_thr.set_xlim(0, self.time_window)
        ax_thr.set_ylim(0, 1)
        line_thr, = ax_thr.plot(self.x_100, self.thr, 'r-')

        ax_att = fig.add_subplot(323)
        ax_att.set_title('Attitude command')
        ax_att.legend(['roll', 'pitch', 'yaw'])
        ax_att.set_xlim(0, self.time_window)
        ax_att.set_ylim(-3.1416, 3.1416)
        line_rol, = ax_att.plot(self.x_100, self.rol, 'r-')
        line_pit, = ax_att.plot(self.x_100, self.pit, 'g-')
        # line_yaw, = ax_att.plot(self.x_100, self.yaw, 'b-')

        ax_pos = fig.add_subplot(322)
        ax_pos.set_title('Measured (-) & reference (--) ')
        # ax_pos.set_title('Measured (-) & reference (--) & estimated (:) position')
        ax_pos.legend(['x_measured', 'y_measured', 'z_measured', 'x_reference', 'y_reference', 'z_reference', 'x_estimated', 'y_estimated', 'z_estimated'])
        ax_pos.set_xlim(0, self.time_window)
        line_xmea, = ax_pos.plot(self.x_30, self.x_measured, 'r-')
        line_ymea, = ax_pos.plot(self.x_30, self.y_measured, 'g-')
        line_zmea, = ax_pos.plot(self.x_30, self.z_measured, 'b-')
        line_xref, = ax_pos.plot(self.x_100, self.x_reference, 'r--')
        line_yref, = ax_pos.plot(self.x_100, self.y_reference, 'g--')
        line_zref, = ax_pos.plot(self.x_100, self.z_reference, 'b--')
        # line_xest, = ax_pos.plot(self.x_100, self.x_estimated, 'r:')
        # line_yest, = ax_pos.plot(self.x_100, self.y_estimated, 'g:')
        # line_zest, = ax_pos.plot(self.x_100, self.z_estimated, 'b:')

        ax_vel = fig.add_subplot(324)
        ax_vel.set_title('Reference (--) & estimated (:) velocities')
        ax_vel.legend(['vx_reference', 'vy_reference', 'vz_reference', 'vx_estimated', 'vy_estimated', 'vz_estimated'])
        ax_vel.set_xlim(0, self.time_window)
        line_vxref, = ax_vel.plot(self.x_100, self.vx_reference, 'r--')
        line_vyref, = ax_vel.plot(self.x_100, self.vy_reference, 'g--')
        line_vzref, = ax_vel.plot(self.x_100, self.vz_reference, 'b--')
        line_vxest, = ax_vel.plot(self.x_100, self.vx_estimated, 'r:')
        line_vyest, = ax_vel.plot(self.x_100, self.vy_estimated, 'g:')
        line_vzest, = ax_vel.plot(self.x_100, self.vz_estimated, 'b:')

        ax_acc = fig.add_subplot(326)
        ax_acc.set_title('Reference accelerations')
        ax_acc.legend(['vx_reference', 'vy_reference', 'vz_reference'])
        ax_acc.set_xlim(0, self.time_window)
        line_axref, = ax_acc.plot(self.x_100, self.ax_reference, 'r--')
        line_ayref, = ax_acc.plot(self.x_100, self.ay_reference, 'g--')
        line_azref, = ax_acc.plot(self.x_100, self.az_reference, 'b--')

        ax_yaw = fig.add_subplot(325)
        ax_yaw.set_title('Yaw measured (-)')
        ax_yaw.legend(['ya_estimated'])
        ax_yaw.set_xlim(0, self.time_window)
        ax_yaw.set_ylim(-3.1416, 3.1416)
        line_yaest, = ax_yaw.plot(self.x_100, self.ya_estimated, 'r-')

        while not rospy.is_shutdown():
            try:
                line_thr.set_ydata(self.thr)
                line_rol.set_ydata(self.rol)
                line_pit.set_ydata(self.pit)
                # line_yaw.set_ydata(self.yaw)
                line_xmea.set_ydata(self.x_measured)
                line_ymea.set_ydata(self.y_measured)
                line_zmea.set_ydata(self.z_measured)
                line_xref.set_ydata(self.x_reference)
                line_yref.set_ydata(self.y_reference)
                line_zref.set_ydata(self.z_reference)
                line_vxref.set_ydata(self.vx_reference)
                line_vyref.set_ydata(self.vy_reference)
                line_vzref.set_ydata(self.vz_reference)
                line_axref.set_ydata(self.ax_reference)
                line_ayref.set_ydata(self.ay_reference)
                line_azref.set_ydata(self.az_reference)
                # line_xest.set_ydata(self.x_estimated)
                # line_yest.set_ydata(self.y_estimated)
                # line_zest.set_ydata(self.z_estimated)
                line_yaest.set_ydata(self.ya_estimated)
                line_vxest.set_ydata(self.vx_estimated)
                line_vyest.set_ydata(self.vy_estimated)
                line_vzest.set_ydata(self.vz_estimated)

                ax_att.relim()
                ax_pos.relim()
                ax_vel.relim()
                ax_acc.relim()

                ax_att.autoscale()
                ax_pos.autoscale()
                ax_vel.autoscale()
                ax_acc.autoscale()

                fig.canvas.draw()
                fig.canvas.flush_events()

            except TclError:
                break


if __name__ == '__main__':

    d = Display()
    d.start()

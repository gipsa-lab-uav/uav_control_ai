#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module to plot different results

"""

import numpy as np
import rosbag
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import axes3d, Axes3D
from cycler import cycler
from sklearn.metrics import mean_squared_error as mse

def display_states(time_axis, states, time_ref=None, states_ref=None):
    """ Plot evolutions of states
        If time_trj and trj are given position plot is updated with it
    """
    grid_opt = {"color":"lightgray", "linestyle":"--"}
    cycle = cycler(color=['r', 'g', 'b'])

    fig, axs = plt.subplots(2, 2, figsize=((12, 9)))
    fig.suptitle('Evolution of states', fontsize=16)

    legend = ['x', 'y', 'z']
    axs[0, 0].set_prop_cycle(cycle)
    axs[0, 0].plot(time_axis, states[:, 0:3])

    if time_ref and states_ref is not None:
        axs[0, 0].plot(time_ref, states_ref[:, 0:3], '--')
        legend += ['$x_{ref}$', '$y_{ref}$', '$z_{ref}$']

    axs[0, 0].legend(legend)
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude (m)')
    axs[0, 0].set_title('Evolution of positions')
    axs[0, 0].grid(**grid_opt)

    legend = ['vx', 'vy', 'vz']
    axs[0, 1].set_prop_cycle(cycle)
    axs[0, 1].plot(time_axis, states[:, 6:9])
    axs[0, 1].legend(legend)
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Speed (m/s)')
    axs[0, 1].set_title('Evolution of linear velocities')
    axs[0, 1].grid(**grid_opt)

    legend = ['roll $\\phi$', 'pitch $\\theta$', 'yaw $\\psi$']
    axs[1, 0].set_prop_cycle(cycle)
    axs[1, 0].plot(time_axis, states[:, 3:6])
    axs[1, 0].legend(legend)
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Angle (radians)')
    axs[1, 0].set_title('Evolution of angles')
    axs[1, 0].grid(**grid_opt)

    legend = ['p', 'q', 'r']
    axs[1, 1].set_prop_cycle(cycle)
    axs[1, 1].plot(time_axis, states[:, 9:12])
    axs[1, 1].legend(legend)
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Angular velocity (rad/s)')
    axs[1, 1].set_title('Evolution of angular velocities')
    axs[1, 1].grid(**grid_opt)

def display_states_derivatives(time_axis, states_derivatives):
    """ Plot evolutions of states derivatives
    """
    grid_opt = {"color":"lightgray", "linestyle":"--"}
    cycle = cycler(color=['r', 'g', 'b'])

    fig, axs = plt.subplots(2, 2, figsize=((12, 9)))
    fig.suptitle('Evolution of states', fontsize=16)

    legend = ['vx', 'vy', 'vz']
    axs[0, 0].set_prop_cycle(cycle)
    axs[0, 0].plot(time_axis, states_derivatives[:, 0:3])
    axs[0, 0].legend(legend)
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Speed (m/s)')
    axs[0, 0].set_title('Evolution of linear velocities')
    axs[0, 0].grid(**grid_opt)

    legend = ['ax', 'ay', 'az']
    axs[0, 1].set_prop_cycle(cycle)
    axs[0, 1].plot(time_axis, states_derivatives[:, 6:9])
    axs[0, 1].legend(legend)
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Acceleration [m/s2]')
    axs[0, 1].set_title('Evolution of linear accelerations')
    axs[0, 1].grid(**grid_opt)

    legend = ['p', 'q', 'r']
    axs[1, 0].set_prop_cycle(cycle)
    axs[1, 0].plot(time_axis, states_derivatives[:, 3:6])
    axs[1, 0].legend(legend)
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Angular velocity (rad/s)')
    axs[1, 0].set_title('Evolution of angular velocities')
    axs[1, 0].grid(**grid_opt)

    legend = ['ap', 'aq', 'ar']
    axs[1, 1].set_prop_cycle(cycle)
    axs[1, 1].plot(time_axis, states_derivatives[:, 9:12])
    axs[1, 1].legend(legend)
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Angular acceleration [rad/s2]')
    axs[1, 1].set_title('Evolution of angular accelerations')
    axs[1, 1].grid(**grid_opt)

def display_command(time_axis, commmands, type_command="classic", cmd_min=None, cmd_max=None):
    """ Plot evolutions of commands
        Choose type_command = "classic" if:
            command = [T Mp Mq Mr]
        Else:
            command = [wr1 wr2 wr3 wr4]
    """
    grid_opt = {"color":"lightgray", "linestyle":"--"}

    fig, axs = plt.subplots(2, 2, figsize=((12, 8)))
    fig.suptitle('Evolution of commands', fontsize=16)

    if type_command == "classic":
        label1 = 'Thrust'
        label2 = 'Moment p'
        label3 = 'Moment q'
        label4 = 'Moment r'
        y_label1 = 'Thrust [N]'
        y_label2 = 'Moment [Nm]'
    else:
        label1 = "Rotor speed 1"
        label2 = "Rotor speed 2"
        label3 = "Rotor speed 3"
        label4 = "Rotor speed 4"
        y_label1 = "Rotor speed (rad/s)"
        y_label2 = "Rotor speed (rad/s)"

    axs[0, 0].plot(time_axis, commmands[:, 0], 'r')
    if cmd_min is not None and cmd_max is not None:
        axs[0, 0].plot(time_axis, cmd_min[0]*np.ones(len(time_axis)), '--r')
        axs[0, 0].plot(time_axis, cmd_max[0]*np.ones(len(time_axis)), '--r')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel(y_label1)
    axs[0, 0].legend([label1])
    axs[0, 0].grid(**grid_opt)

    axs[1, 0].plot(time_axis, commmands[:, 1], 'g')
    if cmd_min is not None and cmd_max is not None:
        axs[1, 0].plot(time_axis, cmd_min[1]*np.ones(len(time_axis)), '--g')
        axs[1, 0].plot(time_axis, cmd_max[1]*np.ones(len(time_axis)), '--g')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel(y_label2)
    axs[1, 0].legend([label2])
    axs[1, 0].grid(**grid_opt)

    axs[0, 1].plot(time_axis, commmands[:, 2], 'b')
    if cmd_min is not None and cmd_max is not None:
        axs[0, 1].plot(time_axis, cmd_min[2]*np.ones(len(time_axis)), '--b')
        axs[0, 1].plot(time_axis, cmd_max[2]*np.ones(len(time_axis)), '--b')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel(y_label2)
    axs[0, 1].legend([label3])
    axs[0, 1].grid(**grid_opt)

    axs[1, 1].plot(time_axis, commmands[:, 3], 'c')
    if cmd_min is not None and cmd_max is not None:
        axs[1, 1].plot(time_axis, cmd_min[3]*np.ones(len(time_axis)), '--c')
        axs[1, 1].plot(time_axis, cmd_max[3]*np.ones(len(time_axis)), '--c')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel(y_label2)
    axs[1, 1].legend([label4])
    axs[1, 1].grid(**grid_opt)
    plt.show()

def compare_results(time_axis, states1, states2):
    """ Plot a comparison of two trajectories """
    grid_opt = {"color":"lightgray", "linestyle":"--"}
    cycle = cycler(color=['r', 'g', 'b'])

    fig, axs = plt.subplots(2, 2, figsize=((12, 9)))
    fig.suptitle('Evolution of states', fontsize=16)

    legend = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2']
    axs[0, 0].set_prop_cycle(cycle)
    axs[0, 0].plot(time_axis, states1[:, 0:3])
    axs[0, 0].plot(time_axis, states2[:, 0:3], '--')
    axs[0, 0].legend(legend)
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude (m)')
    axs[0, 0].set_title('Evolution of positions')
    axs[0, 0].grid(**grid_opt)

    legend = ['vx1', 'vy1', 'vz1', 'vx2', 'vy2', 'vz2']
    axs[0, 1].set_prop_cycle(cycle)
    axs[0, 1].plot(time_axis, states1[:, 6:9])
    axs[0, 1].plot(time_axis, states2[:, 6:9], '--')
    axs[0, 1].legend(legend)
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Speed (m/s)')
    axs[0, 1].set_title('Evolution of linear velocities')
    axs[0, 1].grid(**grid_opt)

    legend = ['roll1 $\\phi_1$', 'pitch1 $\\theta_1$', 'yaw1 $\\psi_1$',\
              'roll2 $\\phi_2$', 'pitch2 $\\theta_2$', 'yaw2 $\\psi_2$']
    axs[1, 0].set_prop_cycle(cycle)
    axs[1, 0].plot(time_axis, states1[:, 3:6])
    axs[1, 0].plot(time_axis, states2[:, 3:6], '--')
    axs[1, 0].legend(legend)
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Angle (radians)')
    axs[1, 0].set_title('Evolution of angles')
    axs[1, 0].grid(**grid_opt)

    legend = ['p1', 'q1', 'r1', 'p2', 'q2', 'r2']
    axs[1, 1].set_prop_cycle(cycle)
    axs[1, 1].plot(time_axis, states1[:, 9:12])
    axs[1, 1].plot(time_axis, states2[:, 9:12], '--')
    axs[1, 1].legend(legend)
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Angular velocity (rad/s)')
    axs[1, 1].set_title('Evolution of angular velocities')
    axs[1, 1].grid(**grid_opt)

def display_3d(states_ref, state=None, y_lim=None):
    """ Plot 3D view of a trajectory """
    fig = plt.figure(figsize=((12, 8)))
    # fig.suptitle('Evolution of positions', fontsize=16)
    axis = fig.add_subplot(111, projection='3d')
    axis.plot(states_ref[:, 0], states_ref[:, 1], states_ref[:, 2], 'blue',\
    linestyle='--', linewidth=1.5)
    legend = ['reference']
    if state is not None:
        axis.plot(state[:, 0], state[:, 1], state[:, 2], 'blue', linewidth=1.5)
        legend += ['trajectory']
        axis.legend(legend)
    axis.set_xlabel('x axis (m)')
    axis.set_ylabel('y axis (m)')
    axis.set_zlabel('z axis (m)')

    if y_lim is not None:
        axis.set_ylim(y_lim)

def display_ref(time_ref, states_ref):
    """ Plot evolutions of states references """
    grid_opt = {"color":"lightgray", "linestyle":"--"}
    plt.figure(figsize=(12, 9))
    legend = ['$x_{ref}$', '$y_{ref}$', '$z_{ref}$']
    plt.plot(time_ref, states_ref[:, 0:3])
    plt.legend(legend)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (m)')
    plt.title('Evolution of positions')
    plt.grid(**grid_opt)
    if states_ref.shape[1] == 13:
        plt.plot(time_ref, states_ref[:, -1])

def display_position_paper(time_cor, states_corr, time_nl, states_nl, time_lin, states_lin, opt=2):
    """ Plot evolutions of states
        If time_trj and trj are given position plot is updated with it
    """
    grid_opt = {"color":"lightgray", "linestyle":"--"}
    if opt not in [1, 2, 3]:
        fig, axs = plt.subplots(1, 3, figsize=((12, 9)))
        fig.suptitle('Evolution of positions', fontsize=16)
        legend = ['Expected linear','Obtained complete', 'Corrected']
        plt.legend(legend)
        axs[0].plot(time_lin, states_lin[:, 0], 'g')
        axs[0].plot(time_nl, states_nl[:, 0], 'r--')
        axs[0].plot(time_cor, states_corr[:, 0], 'b-')

        axs[0].set_ylabel('Amplitude (m)')
        axs[0].set_title('Evolution of x')
        axs[0].grid(**grid_opt)

        axs[1].plot(time_lin, states_lin[:, 1], 'g')
        axs[1].plot(time_nl, states_nl[:, 1], 'r--')
        axs[1].plot(time_cor, states_corr[:, 1], 'b-')

        axs[1].set_ylabel('Amplitude (m)')
        axs[1].set_title('Evolution of y')
        axs[1].grid(**grid_opt)

        axs[2].plot(time_lin, states_lin[:, 2], 'g')
        axs[2].plot(time_nl, states_nl[:, 2], 'r--')
        axs[2].plot(time_cor, states_corr[:, 2], 'b-')

        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Amplitude (m)')
        axs[2].set_title('Evolution of z')
        axs[2].grid(**grid_opt)
    else:
        plt.figure()
        # plt.title('Evolution of altitude (z-axis)')
        plt.plot(time_lin, states_lin[:, opt], 'g:')
        plt.plot(time_nl, states_nl[:, opt], 'r--')
        plt.plot(time_cor, states_corr[:, opt], 'b-.')
        legend = ['Expected linear behavior','Feedback law', 'Feedback law + Correction']
        # plt.legend(legend)
        plt.xlabel('Time (s)')
        plt.ylabel('Altitude (m)')
        plt.grid(**grid_opt)

        ax2 = plt.subplot(223)
        ax2.plot(time_lin, states_lin[:, opt], 'g:')
        ax2.plot(time_nl, states_nl[:, opt], 'r--')
        ax2.plot(time_cor, states_corr[:, opt], 'b-.')
        ax2.set_xlabel('Time (s)', size=13)
        ax2.set_ylabel('Altitude (m)', size=13)



        # ax2.grid(**grid_opt)
        # ax2.set_xlim([3,5])
        # ax2.set_ylim([1.4,2.])
        # ax2.set_title('Zoom on (A)', size=13)
        # ax3 = plt.subplot(224)
        # ax3.plot(time_lin, states_lin[:, opt], 'g:')
        # ax3.plot(time_nl, states_nl[:, opt], 'r--')
        # ax3.plot(time_cor, states_corr[:, opt], 'b-.')
        # ax3.set_xlabel('Time (s)', size=13)
        # ax3.grid(**grid_opt)
        # ax3.set_xlim([15,17.9])
        # ax3.set_ylim([1.41,1.53])
        # ax3.set_title('Zoom on (B)', size=13)


        # ax1 = plt.subplot(211)
        # ax1.set_title('Evolution of altitude', size=13)
        # ax1.plot(time_lin, states_lin[:, opt], 'g:')
        # ax1.plot(time_nl, states_nl[:, opt], 'r--')
        # ax1.plot(time_cor, states_corr[:, opt], 'b-.')
        # ax1.set_xlabel('Time (s)', size=13)
        # ax1.set_ylabel('Altitude (m)', size=13)
        # ax1.grid(**grid_opt)
        # # ax1.vlines(22, 0, 2.6, colors='k', linestyles=':', linewidth=1)
        # rect = patches.Rectangle((3,1.4), 1.9, 0.6, linewidth=2, edgecolor='none',facecolor='lightcoral', alpha=0.5)
        # ax1.add_patch(rect)
        # rect = patches.Rectangle((14,1.41), 3, 0.2, linewidth=2, edgecolor='none',facecolor='lightcoral', alpha=0.5)
        # ax1.add_patch(rect)
        # ax1.set_xlim([0,40])
        # ax3.legend(legend, loc='lower center', bbox_to_anchor=(-0.6, -0.3, 1, .08),
        #                       fancybox=True, shadow=False, ncol=3, prop={'size': 12})
        # # ax1.text(15,2.5,'With correction', color='k')
        # # ax1.text(22.2,2.5,'Without correction', color='k')


        # ax1.text(5.8,1.7,'(A)', color='lightcoral')
        # ax1.text(15,1,'(B)', color='lightcoral')
        # ax2 = plt.subplot(223)
        # ax2.plot(time_lin, states_lin[:, opt], 'g:')
        # ax2.plot(time_nl, states_nl[:, opt], 'r--')
        # ax2.plot(time_cor, states_corr[:, opt], 'b-.')
        # ax2.set_xlabel('Time (s)', size=13)
        # ax2.set_ylabel('Altitude (m)', size=13)
        # ax2.grid(**grid_opt)
        # ax2.set_xlim([2.5,12])
        # ax2.set_ylim([1.97,2.01])
        # ax2.set_title('Zoom on (A)', size=13)
        # ax3 = plt.subplot(224)
        # ax3.plot(time_lin, states_lin[:, opt], 'g:')
        # ax3.plot(time_nl, states_nl[:, opt], 'r--')
        # ax3.plot(time_cor, states_corr[:, opt], 'b-.')
        # ax3.set_xlabel('Time (s)', size=13)
        # ax3.grid(**grid_opt)
        # ax3.set_xlim([14,17.5])
        # ax3.set_ylim([0.9,1.6])
        # ax3.set_title('Zoom on (B)', size=13)

        # ax1 = plt.subplot(211)
        # ax1.set_title('Evolution of altitude', size=13)
        # ax1.plot(time_lin, states_lin[:, opt], 'g:')
        # ax1.plot(time_nl, states_nl[:, opt], 'r--')
        # ax1.plot(time_cor, states_corr[:, opt], 'b-.')
        # ax1.set_xlabel('Time (s)', size=13)
        # ax1.set_ylabel('Altitude (m)', size=13)
        # ax1.grid(**grid_opt)
        # rect = patches.Rectangle((2.5,1.9), 9, 0.17, linewidth=2, edgecolor='none',facecolor='lightcoral', alpha=0.5)
        # ax1.add_patch(rect)
        # rect = patches.Rectangle((14,0.9), 3.5, 0.7, linewidth=2, edgecolor='none',facecolor='lightcoral', alpha=0.5)
        # ax1.add_patch(rect)
        # ax1.set_xlim([0,32])
        # ax3.legend(legend, loc='lower center', bbox_to_anchor=(-0.6, -0.3, 1, .08),
        #                       fancybox=True, shadow=False, ncol=3, prop={'size': 12})
        # ax1.text(7,1.7,'(A)', color='lightcoral')
        # ax1.text(15,0.6,'(B)', color='lightcoral')

        ax2.grid(**grid_opt)
        ax2.set_xlim([21.7,25])
        ax2.set_ylim([1.4,1.63])
        ax2.set_title('Zoom on (A)', size=13)
        ax3 = plt.subplot(224)
        ax3.plot(time_lin, states_lin[:, opt], 'g:')
        ax3.plot(time_nl, states_nl[:, opt], 'r--')
        ax3.plot(time_cor, states_corr[:, opt], 'b-.')
        ax3.set_xlabel('Time (s)', size=13)
        ax3.grid(**grid_opt)
        ax3.set_xlim([31,34.4])
        ax3.set_ylim([2.3,2.55])
        ax3.set_title('Zoom on (B)', size=13)


        ax1 = plt.subplot(211)
        ax1.set_title('Evolution of altitude', size=13)
        ax1.plot(time_lin, states_lin[:, opt], 'g:')
        ax1.plot(time_nl, states_nl[:, opt], 'r--')
        ax1.plot(time_cor, states_corr[:, opt], 'b-.')
        ax1.set_xlabel('Time (s)', size=13)
        ax1.set_ylabel('Altitude (m)', size=13)
        ax1.grid(**grid_opt)
        # ax1.vlines(22, 0, 2.6, colors='k', linestyles=':', linewidth=1)
        rect = patches.Rectangle((21.7,1.4), 4, 0.6, linewidth=2, edgecolor='none',facecolor='lightcoral', alpha=0.5)
        ax1.add_patch(rect)
        rect = patches.Rectangle((31,2.3), 3, 0.4, linewidth=2, edgecolor='none',facecolor='lightcoral', alpha=0.5)
        ax1.add_patch(rect)
        ax1.set_xlim([0,40])
        ax3.legend(legend, loc='lower center', bbox_to_anchor=(-0.6, -0.3, 1, .08),
                              fancybox=True, shadow=False, ncol=3, prop={'size': 12})
        # ax1.text(15,2.5,'With correction', color='k')
        # ax1.text(22.2,2.5,'Without correction', color='k')


        ax1.text(20,1.7,'(A)', color='lightcoral')
        ax1.text(32,1.9,'(B)', color='lightcoral')

def display_error(time_cor, states_corr, time_nl, states_nl, time_lin, states_lin, opt=2):
        grid_opt = {"color":"lightgray", "linestyle":"--"}
        legend = ['Feedback law', 'Feedback law with correction']
        if opt in [0, 1, 2]:
            fig, ax = plt.subplots()
            # plt.title('Error between obtained and expected behavior on z-axis')
            plt.plot(time_lin, states_nl[:, opt] - states_lin[:, opt], 'r--')
            plt.plot(time_nl, states_corr[:, opt] - states_lin[:, opt], 'b--')

            plt.legend(legend, loc='lower center', bbox_to_anchor=(0, -0.15, 1, .08),
                              fancybox=True, shadow=False, ncol=2,prop={'size': 12})
            plt.xlabel('Time (s)',size=12)
            plt.ylabel('Error amplitude (m)',size=15)
            plt.grid(**grid_opt)
            ax.tick_params(axis='both', which='major', labelsize=11)
        elif opt == 3:
            fig, axs = plt.subplots(2, 1, figsize=((12, 9)))
            # plt.title('Evolution of altitude (z-axis)')
            axs[0].plot(time_lin, states_lin[:, 2], 'g:')
            axs[0].plot(time_nl, states_nl[:, 2], 'r--')
            axs[0].plot(time_cor, states_corr[:, 2], 'b-.')
            legend = ['Expected linear behavior','Feedback law', 'Feedback law with correction']
            axs[0].legend(legend,prop={'size': 12})
            axs[0].set_xlabel('Time (s)',size=12)
            axs[0].set_ylabel('Error amplitude (m)',size=12)
            axs[0].grid(**grid_opt)

            axs[1].plot(time_lin, states_nl[:, 2] - states_lin[:, 2], 'r--')
            axs[1].plot(time_nl, states_corr[:, 2] - states_lin[:, 2], 'b--')

            axs[1].legend(legend)
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('Amplitude (m)')
            axs[1].grid(**grid_opt)
        else:
            fig, axs = plt.subplots(1, 3, figsize=((12, 9)))
            for i in range(0, 3):
                if i==0:
                    char ='x'
                elif i==1:
                    char ='y'
                else:
                    char ='z'
                axs[i].plot(time_lin, states_nl[:, i] - states_lin[:, i], 'r--')
                axs[i].plot(time_nl, states_corr[:, i] - states_lin[:, i], 'b--')
                axs[i].set_ylabel('Error on ' + char +' axis (m)',size=12)
                axs[i].set_xlabel('Time (s)',size=12)
                axs[i].grid(**grid_opt)
                axs[i].legend(legend, loc='upper center', bbox_to_anchor=(0, 1, 1, .08),
                              fancybox=True, shadow=False, ncol=1,prop={'size': 12})
        out1 = mse(states_nl[:, opt], states_lin[:, opt])
        out2 = np.std(states_nl[:, opt] - states_lin[:, opt])
        out3 = mse(states_corr[:, opt], states_lin[:, opt])
        out4 = np.std(states_corr[:, opt] - states_lin[:, opt])

        return (out1, out2, out3, out4)

def close():
    plt.close('all')

def ros_states(bag):

    x = [];y = [];z = []
    vx = [];vy = [];vz = []
    phi = [];theta = [];psi = []
    p = []
    q = []
    r = []

    tt = []
    for topic1, msg, t in bag.read_messages(topics=['/trajectory_control/statecommanddata']):
        x.append(msg.positions[0])
        y.append(msg.positions[1])
        z.append(msg.positions[2])
        vx.append(msg.velocities_linear[0])
        vy.append(msg.velocities_linear[1])
        vz.append(msg.velocities_linear[2])
        phi.append(msg.euler_angles[0])
        theta.append(msg.euler_angles[1])
        psi.append(msg.euler_angles[2])
        p.append(msg.velocities_angular[0])
        q.append(msg.velocities_angular[1])
        r.append(msg.velocities_angular[2])
        tt.append(msg.time)
    return tt, np.array([x, y, z, phi, theta, psi,vx,vy,vz,p,q,r]).transpose()

def ros_states_ref(bag):

    xref = [];yref = [];zref = [];psiref = []
    tt = []
    for topic1, msg, t in bag.read_messages(topics=['/trajectory_control/statecommanddata']):
        xref.append(msg.positions_ref[0])
        yref.append(msg.positions_ref[1])
        zref.append(msg.positions_ref[2])
        psiref.append(msg.positions_ref[3])
        tt.append(msg.time)
    return tt, np.array([xref, yref, zref, psiref]).transpose()

def ros_states_desired(bag):

    xdes = [];ydes = [];zdes = [];tt=[];
    for topic1, msg, t in bag.read_messages(topics=['/trajectory_control/statecommanddata']):
        xdes.append(msg.positions_linear_des[0])
        ydes.append(msg.positions_linear_des[1])
        zdes.append(msg.positions_linear_des[2])
        tt.append(msg.time)
    return tt, np.array([xdes, ydes, zdes]).transpose()

def display_ros_states(bag):
    tt, states = ros_states(bag)
    display_states(tt, states)

def display_ros_3d_ref(bag):
    """ Plot evolutions of states
    """
    tt_ref, states_ref = ros_states_ref(bag)

    grid_opt = {"color":"lightgray", "linestyle":"--"}

    # plt.figure()
    # plt.plot(states_ref)

    fig = plt.figure(figsize=((12, 8)))
    axis = fig.add_subplot(221, projection='3d')
    axis.plot(states_ref[10:6800, 0], states_ref[10:6800, 1], states_ref[10:6800, 2], 'blue',\
    linestyle='--', linewidth=1.5)
    axis.grid(**grid_opt)
    plt.locator_params(nbins=3)
    axis = fig.add_subplot(222, projection='3d')
    axis.plot(states_ref[9300:15140, 0], states_ref[9300:15140, 1], states_ref[9300:15140, 2], 'blue',\
    linestyle='--', linewidth=1.5)
    axis.grid(**grid_opt)
    plt.locator_params(nbins=3)
    axis = fig.add_subplot(223, projection='3d')
    axis.plot(states_ref[15140:21353, 0], states_ref[15140:21353, 1], states_ref[15140:21353, 2], 'blue',\
    linestyle='--', linewidth=1.5)
    axis.grid(**grid_opt)
    plt.locator_params(nbins=3)
    axis = fig.add_subplot(224, projection='3d')
    axis.plot(states_ref[21353:-1, 0], states_ref[21353:-1, 1], states_ref[21353:-1, 2], 'blue',\
    linestyle='--', linewidth=1.5)
    axis.grid(**grid_opt)
    # axis.tick_params(which='major', length=0)
    # axis.xaxis.set_minor_locator(AutoMinorLocator())
    plt.locator_params(nbins=3)

    plt.subplots_adjust(wspace=0, hspace=0.1)
    legend = ['reference']


def display_ros_pos(bag):
    """ Plot evolutions of states
    """
    tt, states = ros_states(bag)
    tt_ref, states_ref = ros_states_ref(bag)

    grid_opt = {"color":"lightgray", "linestyle":"--"}

    fig, axs = plt.subplots(3, 1, figsize=((12, 9)))
    fig.suptitle('Evolution of positions', fontsize=19)
    legend = ['Position','Reference']


    for i in range(0,3):
        axs[i].plot(tt_ref, states_ref[:, i], color='r', linestyle='--')
        axs[i].plot(tt, states[:, i], color='tab:blue', linestyle='-')
        axs[i].set_ylabel('Amplitude (m)')
        axs[i].grid(**grid_opt)

    axs[0].set_title('Evolution of x')
    axs[1].set_title('Evolution of y')
    axs[2].set_title('Evolution of z')
    axs[2].set_xlabel('Time (s)')

def compare_ros_pos(bag1, bag2, offset=0, data_range=[]):
    """ Plot evolutions of states
    """
    tt1, states1 = ros_states(bag1)
    tt_ref11, states_ref11 = ros_states_ref(bag1)
    tt_ref1, states_ref1 = ros_states_desired(bag1)

    tt2, states2 = ros_states(bag2)
    tt_ref22, states_ref22 = ros_states_ref(bag2)
    tt_ref2, states_ref2 = ros_states_desired(bag2)

    plt.figure()
    plt.plot(np.asarray(tt_ref11)-tt_ref11[0]-offset, states_ref11)
    plt.plot(np.asarray(tt_ref22)-tt_ref22[0], states_ref22)

    tt1 = np.asarray(tt1)-tt1[0]-offset
    tt_ref1 = np.asarray(tt_ref1)-tt_ref1[0]-offset
    tt2 = np.asarray(tt2)-tt2[0]
    tt_ref2 = np.asarray(tt_ref2)-tt_ref2[0]
    data_range_tt1 = [0, 0]
    data_range_tt2 = [0, 0]
    data_range_tt_ref1 = [0, 0]
    data_range_tt_ref2 = [0, 0]

    if not data_range == []:

        idx0 = np.where(tt1 >= data_range[0])
        # print(idx0[0][0])
        # print(idx0[0])
        data_range_tt1[0] = idx0[0][0]
        idx1 = np.where(tt1 >= data_range[1])
        data_range_tt1[1] = idx1[0][0]

        idx0 = np.where(tt2 >= data_range[0])
        data_range_tt2[0] = idx0[0][0]
        idx1 = np.where(tt2 >= data_range[1])
        data_range_tt2[1] = idx1[0][0]

        idx0 = np.where(tt_ref1 >= data_range[0])
        data_range_tt_ref1[0] = idx0[0][0]
        idx1 = np.where(tt_ref1 >= data_range[1])
        data_range_tt_ref1[1] = idx1[0][0]

        idx0 = np.where(tt_ref2 >= data_range[0])
        data_range_tt_ref2[0] = idx0[0][0]
        idx1 = np.where(tt_ref2 >= data_range[1])
        data_range_tt_ref2[1] = idx1[0][0]

        tt1 = tt1[data_range_tt1[0]:data_range_tt1[1]]-tt1[data_range_tt1[0]]
        tt2 = tt2[data_range_tt2[0]:data_range_tt2[1]]-tt2[data_range_tt2[0]]
        tt_ref1 = tt_ref1[data_range_tt_ref1[0]:data_range_tt_ref1[1]]-tt_ref1[data_range_tt_ref1[0]]
        tt_ref2 = tt_ref2[data_range_tt_ref2[0]:data_range_tt_ref2[1]]-tt_ref2[data_range_tt_ref2[0]]
        states1 = states1[data_range_tt1[0]:data_range_tt1[1]]
        states2 = states2[data_range_tt2[0]:data_range_tt2[1]]
        states_ref1 = states_ref1[data_range_tt_ref1[0]:data_range_tt_ref1[1]]
        states_ref2 = states_ref2[data_range_tt_ref2[0]:data_range_tt_ref2[1]]

    grid_opt = {"color":"lightgray", "linestyle":"--"}

    fig, axs = plt.subplots(3, 1, figsize=((9, 8)))
    plt.subplots_adjust(top=0.95, bottom=0.15)
    # fig.suptitle('Evolution of positions', fontsize=16)
    axs[0].set_title('Evolution of positions', fontsize=19)
    legend = ['Cascaded Linear Controller', 'DNN Enhanced Linear Controller','Linear Expected Behavior']
    step = 10
    for i in range(0,3):
        axs[i].plot(tt1[0:-1:step], states1[0:-1:step, i], color='tab:red', linestyle='-')
        # axs[i].plot(tt_ref2, states_ref2[:, i], color='tab:brown', linestyle='--')
        axs[i].plot(tt2[0:-1:step], states2[0:-1:step, i], color='tab:blue', linestyle='-')
        axs[i].plot(tt_ref1[0:-1:step], states_ref1[0:-1:step, i], color='C2', linestyle='--')
        # axs[i].set_ylabel('Amplitude (m)')
        axs[i].grid(**grid_opt)
        axs[i].set_xlim([tt_ref1[0], tt_ref1[-1]])
        axs[i].set_xlabel('Time (s)', size=12, labelpad=1)
    axs[0].set_ylabel('Amplitude of x (m)', size=14)
    axs[1].set_ylabel('Amplitude of y (m)', size=14)
    axs[2].set_ylabel('Amplitude of z (m)', size=14)

    axs[2].legend(legend, loc='lower center', bbox_to_anchor=(-0.03, -0.66, 1, .08),
                              fancybox=True, shadow=False, ncol=2, prop={'size': 15})
                              # ,fancybox=True, shadow=False, ncol=2,prop={'size': 12})

def disp_mse(tt, xmse):
    grid_opt = {"color":"lightgray", "linestyle":"--"}
    fig, axs = plt.subplots(1, 1, figsize=((12, 9)))
    # plt.title('Evolution of altitude (z-axis)')
    axs.plot(tt, xmse, 'b')
    axs.set_xlabel('Time (s)', size=12)
    axs.set_ylabel('MSE (m)', size=12)
    axs.grid(**grid_opt)

def display_error_ros(bag1, bag2, offset=0, data_range=[]):
    """ Plot evolutions of states
    """
    tt1, states1 = ros_states(bag1)
    tt_ref1, states_des1 = ros_states_desired(bag1)

    tt2, states2 = ros_states(bag2)
    tt_ref2, states_des2 = ros_states_desired(bag2)

    tt1 = np.asarray(tt1)-tt1[0]-offset
    tt_ref1 = np.asarray(tt_ref1)-tt_ref1[0]-offset
    tt2 = np.asarray(tt2)-tt2[0]
    tt_ref2 = np.asarray(tt_ref2)-tt_ref2[0]

    data_range_tt1 = [0, 0]
    data_range_tt2 = [0, 0]
    data_range_tt_ref1 = [0, 0]
    data_range_tt_ref2 = [0, 0]

    if not data_range == []:

        idx0 = np.where(tt1 >= data_range[0])
        # print(idx0[0][0])
        # print(idx0[0])
        data_range_tt1[0] = idx0[0][0]
        idx1 = np.where(tt1 >= data_range[1])
        data_range_tt1[1] = idx1[0][0]

        idx0 = np.where(tt2 >= data_range[0])
        data_range_tt2[0] = idx0[0][0]
        idx1 = np.where(tt2 >= data_range[1])
        data_range_tt2[1] = idx1[0][0]

        idx0 = np.where(tt_ref1 >= data_range[0])
        data_range_tt_ref1[0] = idx0[0][0]
        idx1 = np.where(tt_ref1 >= data_range[1])
        data_range_tt_ref1[1] = idx1[0][0]

        idx0 = np.where(tt_ref2 >= data_range[0])
        data_range_tt_ref2[0] = idx0[0][0]
        idx1 = np.where(tt_ref2 >= data_range[1])
        data_range_tt_ref2[1] = idx1[0][0]

        tt1 = tt1[data_range_tt1[0]:data_range_tt1[1]]-tt1[data_range_tt1[0]]
        tt2 = tt2[data_range_tt2[0]:data_range_tt2[1]]-tt2[data_range_tt2[0]]
        tt_ref1 = tt_ref1[data_range_tt_ref1[0]:data_range_tt_ref1[1]]-tt_ref1[data_range_tt_ref1[0]]
        tt_ref2 = tt_ref2[data_range_tt_ref2[0]:data_range_tt_ref2[1]]-tt_ref2[data_range_tt_ref2[0]]
        states1 = states1[data_range_tt1[0]:data_range_tt1[1]]
        states2 = states2[data_range_tt2[0]:data_range_tt2[1]]
        states_des1 = states_des1[data_range_tt_ref1[0]:data_range_tt_ref1[1]]
        states_des2 = states_des2[data_range_tt_ref2[0]:data_range_tt_ref2[1]]


    fig, axs = plt.subplots(3, 1, figsize=((12, 9)))
    for i in range(0, 3):
        if i==0:
            char ='x'
        elif i==1:
            char ='y'
        else:
            char ='z'
        axs[i].plot(tt1, states1[:, i] - states_des1[:, i], 'r--')
        axs[i].plot(tt2, states2[:, i] - states_des2[:, i], 'b--')
        axs[i].set_ylabel('Error on ' + char +' axis (m)',size=12)
        axs[i].set_xlabel('Time (s)',size=12)
        # axs[i].grid(**grid_opt)
        # axs[i].legend(legend, loc='upper center', bbox_to_anchor=(0, 1, 1, .08),
        #               fancybox=True, shadow=False, ncol=1,prop={'size': 12})
    out = []
    out.append(mse(states1[:, 0], states_des1[:, 0]))
    out.append(mse(states1[:, 1], states_des1[:, 1]))
    out.append(mse(states1[:, 2], states_des1[:, 2]))

    out2 = []
    out2.append(mse(states2[:, 0], states_des2[:, 0]))
    out2.append(mse(states2[:, 1], states_des2[:, 1]))
    out2.append(mse(states2[:, 2], states_des2[:, 2]))

    return out, out2

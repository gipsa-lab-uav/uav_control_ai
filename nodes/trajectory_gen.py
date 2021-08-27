#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as time

import rospy
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class TrajectoryGeneration:
    def __init__(self, node_name='trajectory_gen_node', state='mavros/state' , subscriber='mavros/local_position/odom', publisher='mavros/desired_trajectory'):

        rospy.init_node(node_name, anonymous=True)

        # Define suscribers & publishers
        rospy.Subscriber("/mavros/state", State, self.drone_state_acquire_callback)
        rospy.Subscriber(subscriber, Odometry, self.callback)

        self.pub = rospy.Publisher(publisher, JointTrajectory, queue_size=10)

        # Define & initialize private variables
        self.YAW_HEADING = ['auto', [1, 0]]  # options: ['auto'], ['center', [x, y]], ['axes', [x, y]]
        self.TRAJECTORY_REQUESTED_SPEED = 1.5  # req. trajectory linear speed [m.s-1] (used when arg velocity in not specify in discretise_trajectory())
        self.LANDING_SPEED = 0.3  # [m.s-1]
        self.PUBLISH_RATE = 10  # publisher frequency [Hz]
        self.FREQUENCY = 100  # point trajectory frequency [Hz]
        self.BOX_LIMIT = [[-2., 2.], [-1., 3.], [-.01, 2.]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        self.WINDOW_FRAME = .5  # publish future states comprise in the window time frame [s]

        self.MAX_LINEAR_ACC_XY = 2.5  # max. linear acceleration [m.s-2]
        self.MAX_LINEAR_ACC_Z = 3.0  # max. linear acceleration [m.s-2]

        self.MAX_LINEAR_SPEED_XY = 10.0  # max. linear speed [m.s-1] (only used by generate_states_filtered(), not by generate_states_sg_filtered())
        self.MAX_LINEAR_SPEED_Z = 12.0  # max. linear speed [m.s-1] (only used by generate_states_filtered(), not by generate_states_sg_filtered())

        self.drone_state = State()

        # Define & initialize flags
        self.is_filtered = False
        self.is_first_callback = False
        self.print_time_info = False

    def discretise_trajectory(self, parameters=[], velocity=False, acceleration=False, heading=False, relative=False):
        # Trajectory definition - shape/vertices in inertial frame (x, y, z - up)
        #
        # Define trajectory by using:
        # trajectory_object.discretise_trajectory(parameters=['name', param], (opt.) velocity=float, (opt.) acceleration=float, (opt.) heading=options, , (opt.) relative=bool)
        #
        # Possible parameters:
        # parameters=['takeoff', z] with z in meters
        # parameters=['hover', time] with time in seconds
        # parameters=['vector', [x, y, z]] with x, y, z the target position
        # parameters=['circle', [x, y, z], (opt.) n] with x, y, z the center of the circle and n (optional) the number of circle. Circle
        # is defined by the drone position when starting the circle trajectory and the center. The drone will turn around this point.
        # parameters=['landing']
        # parameters=['returnhome']
        # parameters=['wave', [x, y, z], amplitude, freq] with x, y, z the target position, 'amplitude' the wave amplitude and 'freq' the wave frequency
        #
        # Optional argument:
        # velocity=float
        # acceleration=float
        # heading=options with options: ['auto'], ['still'], ['center', [x, y]], ['axes', [x, y]]
        # relative=bool

        start = time()

        if not hasattr(self, 'x_discretized'):
            self.x_discretized = [.0] * self.FREQUENCY
            self.y_discretized = [.0] * self.FREQUENCY
            self.z_discretized = [.0] * self.FREQUENCY

        if not hasattr(self, 'ya_info'):
            self.ya_info = self.YAW_HEADING * self.FREQUENCY

        if not velocity:
            velocity = self.TRAJECTORY_REQUESTED_SPEED

        if not acceleration:
            acceleration = 0.0

        if not heading:
            heading = self.YAW_HEADING

        x1 = self.x_discretized[-1]
        y1 = self.y_discretized[-1]
        z1 = self.z_discretized[-1]

        v0 = np.array([x1, y1, z1])

        if parameters[0] == 'takeoff':
            profil = self.get_linear_position_profil(abs(parameters[1] - z1), acceleration, velocity, self.MAX_LINEAR_ACC_Z, self.FREQUENCY)

            sign = math.copysign(1.0, parameters[1] - z1)

            x = x1 * np.ones(len(profil))
            y = y1 * np.ones(len(profil))
            z = [z1 + sign * l for l in profil]

        if parameters[0] == 'takeoff_ech':
            profil =  np.ones(len(self.get_linear_position_profil(abs(parameters[1] - z1), acceleration, velocity, self.MAX_LINEAR_ACC_Z, self.FREQUENCY)))

            sign = math.copysign(1.0, parameters[1] - z1)

            x = x1 * np.ones(len(profil))
            y = y1 * np.ones(len(profil))
            z = [z1 + sign * profil[-1] for l in profil]

        elif parameters[0] == 'hover':
            steps = int(parameters[1] * self.FREQUENCY)

            x = x1 * np.ones(steps)
            y = y1 * np.ones(steps)
            z = z1 * np.ones(steps)

        elif parameters[0] == 'vector':
            vf = np.array(parameters[1])
            vector = vf if relative else vf - v0
            d = np.linalg.norm(vector)
            if (d == 0):
                return
            vector_u = vector / d

            profil = self.get_linear_position_profil(d, acceleration, velocity, self.MAX_LINEAR_ACC_XY, self.FREQUENCY)

            x = [x1 + l * vector_u[0] for l in profil]
            y = [y1 + l * vector_u[1] for l in profil]
            z = [z1 + l * vector_u[2] for l in profil]

        elif parameters[0] == 'vector_ech':
            vf = np.array(parameters[1])
            vector = vf if relative else vf - v0
            d = np.linalg.norm(vector)
            if (d == 0):
                return
            vector_u = vector / d

            profil = np.ones(len(self.get_linear_position_profil(d, acceleration, velocity, self.MAX_LINEAR_ACC_XY, self.FREQUENCY)))

            x = [x1 + l * vector_u[0] for l in profil]
            y = [y1 + l * vector_u[1] for l in profil]
            z = [z1 + l * vector_u[2] for l in profil]

        elif parameters[0] == 'circle':
            n = parameters[2] if len(parameters) > 2 else 1
            center = np.array(parameters[1]) + v0 if relative else np.array(parameters[1])
            r = np.linalg.norm(center - v0)
            circumference = 2 * math.pi * r
            height = parameters[3]+z1 if len(parameters) > 3 else z1
            cos_a = (x1 - center[0]) / r
            sin_a = (y1 - center[1]) / r

            profil = self.get_linear_position_profil(n * circumference, acceleration, velocity, self.MAX_LINEAR_ACC_XY / math.sqrt(2), self.FREQUENCY)
            profil2 = [z1+(l-profil[0])*(height-z1)/(profil[-1]-profil[0]) for l in profil]

            x = [(cos_a*math.cos(l/r) - sin_a*math.sin(l/r)) * r + center[0] for l in profil]
            y = [(sin_a*math.cos(l/r) + cos_a*math.sin(l/r)) * r + center[1] for l in profil]
            z = [l for l in profil2]
	    # z = z1 * np.ones(len(profil))

        elif parameters[0] == 'landing':
            profil = self.get_linear_position_profil(abs(z1), acceleration, self.LANDING_SPEED, self.MAX_LINEAR_ACC_Z, self.FREQUENCY)

            x = x1 * np.ones(len(profil))
            y = y1 * np.ones(len(profil))
            z = [z1 - l for l in profil]

        elif parameters[0] == 'returnhome':
            vf = np.array([self.x_discretized[0], self.y_discretized[0], z1])
            vector = vf - v0
            d = np.linalg.norm(vector)
            if (d == 0):
                return
            vector_u = vector / d

            profil1 = self.get_linear_position_profil(d, acceleration, velocity, self.MAX_LINEAR_ACC_XY, self.FREQUENCY)

            x = [x1 + l * vector_u[0] for l in profil1]
            y = [y1 + l * vector_u[1] for l in profil1]
            z = z1 * np.ones(len(profil1))

            profil2 = self.get_linear_position_profil(abs(z1 - self.z_discretized[0]), acceleration, self.LANDING_SPEED, self.MAX_LINEAR_ACC_Z, self.FREQUENCY)

            x = np.concatenate((x, x[-1] * np.ones(len(profil2))), axis=None)
            y = np.concatenate((y, y[-1] * np.ones(len(profil2))), axis=None)
            z = np.concatenate((z, [z1 - l for l in profil2]), axis=None)

        elif parameters[0] == 'wave':
            vf = np.array(parameters[1])
            vector = vf if relative else vf - v0
            d = np.linalg.norm(vector)
            if (d == 0):
                return
            vector_u = vector / d

            profil = self.get_linear_position_profil(d, acceleration, velocity, self.MAX_LINEAR_ACC_XY, self.FREQUENCY)

            x = [x1 + l * vector_u[0] for l in profil]
            y = [y1 + l * vector_u[1] for l in profil]
            z = [z1 + l * vector_u[2] for l in profil]

            z += parameters[2]*np.sin(np.arange(0, len(x))*2.*math.pi*parameters[3]/self.FREQUENCY)

        elif parameters[0] == 'lemniscate':
            d = parameters[2] # if len(parameters) > 2 else 1
            height = z1+parameters[3]
            center = np.array(parameters[1]) + v0 if relative else np.array(parameters[1])
            profil = self.get_linear_position_profil(5.24412*d, acceleration, velocity, self.MAX_LINEAR_ACC_XY / math.sqrt(2), self.FREQUENCY)
            profil2 = [-np.pi+(l-profil[0])*(2*np.pi)/(profil[-1]-profil[0]) for l in profil]
            profil3 = [z1+(l-profil[0])*(height-z1)/(profil[-1]-profil[0]) for l in profil]

            x = [center[0]+d*math.sin(l)/(1+pow(math.cos(l),2)) for l in profil2]
            y = [center[1]+d*math.sin(l)*math.cos(l)/(1+pow(math.cos(l),2)) for l in profil2]
            z = [l for l in profil3]

        # elif parameters[0] == 'square':
        # elif parameters[0] == 'inf':

        self.x_discretized.extend(x[1:])
        self.y_discretized.extend(y[1:])
        self.z_discretized.extend(z[1:])

        self.ya_info.extend([heading] * len(x[1:]))

        if self.print_time_info: print('discretise_trajectory() - {} runs in {} s'.format(parameters[0], time() - start))

    def get_linear_position_profil(self, distance, a, vmax, amax, freq):
        if (distance == 0):
            return [0.]

        dt = 1./freq
        t1 = vmax/amax

        if distance > vmax*t1:
            if a == 0:
                dt2 = distance/vmax - t1
                t2 = t1 + dt2
                tf = t2 + t1
            else:
                A = 0.5 * a * (1 + a/amax)
                B = vmax * (1 + a/amax)
                C = vmax * t1 - distance
                D = B*B - 4*A*C
                if D < 0:
                    rospy.logfatal("[trajectory_gen] ERROR: in discretize_trajectory(), negative acceleration too high")
                    exit()
                dt2 = 0.5 * (-B + math.sqrt(D)) / A
                t2 = t1 + dt2
                tf = t2 + (a/amax) * dt2 + t1
        else:
            tf = math.sqrt(4*distance/amax)
            t1 = tf/2
            t2 = t1

        dv = []

        for t in self.xfrange(tf, dt):
            if t <= t1:
                dv.append(amax * t)
            elif t1 < t <= t2:
                dv.append(amax * t1 + a * (t - t1))
            else:
                dv.append(amax * t1 + a * (t2 - t1) - amax * (t - t2))

        return [x for x in self.xfintegrate(dv, dt)]

    def xfrange(self, end, step):
        x = .0
        while x < end:
            yield x
            x += step

    def xfintegrate(self, l, dt):
        x = .0
        for element in l:
            x += element * dt
            yield x

    def constraint_trajectory_to_box(self):

        self.x_discretized = [self.BOX_LIMIT[0][0] if x < self.BOX_LIMIT[0][0] else x for x in self.x_discretized]
        self.x_discretized = [self.BOX_LIMIT[0][1] if x > self.BOX_LIMIT[0][1] else x for x in self.x_discretized]
        self.y_discretized = [self.BOX_LIMIT[1][0] if x < self.BOX_LIMIT[1][0] else x for x in self.y_discretized]
        self.y_discretized = [self.BOX_LIMIT[1][1] if x > self.BOX_LIMIT[1][1] else x for x in self.y_discretized]
        self.z_discretized = [self.BOX_LIMIT[2][0] if x < self.BOX_LIMIT[2][0] else x for x in self.z_discretized]
        self.z_discretized = [self.BOX_LIMIT[2][1] if x > self.BOX_LIMIT[2][1] else x for x in self.z_discretized]

    def generate_states(self):

        start = time()

        self.x_discretized.extend([self.x_discretized[-1]] * self.FREQUENCY)
        self.y_discretized.extend([self.y_discretized[-1]] * self.FREQUENCY)
        self.z_discretized.extend([self.z_discretized[-1]] * self.FREQUENCY)

        self.ya_info.extend(['still'] * self.FREQUENCY)

        self.ya_discretized = [.0]
        self.vx_discretized = [.0]
        self.vy_discretized = [.0]
        self.vz_discretized = [.0]
        self.ax_discretized = [.0]
        self.ay_discretized = [.0]
        self.az_discretized = [.0]
        self.ti_discretized = [.0]

        prevHeading = np.array([.0, .0])

        for s, _ in enumerate(self.x_discretized[1:]):
            p1 = np.array([self.x_discretized[s], self.y_discretized[s]])
            p2 = np.array([self.x_discretized[s+1], self.y_discretized[s+1]])

            if self.ya_info[s][0] == 'center':
                heading = np.array(self.ya_info[s][1]) - p1
            elif self.ya_info[s][0] == 'axes':
                heading = np.array(self.ya_info[s][1])
            elif self.ya_info[s][0] == 'still':
                heading = prevHeading
            else:
                heading = p2 - p1

            if (np.linalg.norm(heading) < 0.001) or (self.ya_info[s][0] == 'still'):
                heading = prevHeading
            else:
                heading = heading / np.linalg.norm(heading)
                prevHeading = heading

            self.ya_discretized.append(math.atan2(heading[1], heading[0]))
            self.vx_discretized.append((self.x_discretized[s+1] - self.x_discretized[s]) * self.FREQUENCY)
            self.vy_discretized.append((self.y_discretized[s+1] - self.y_discretized[s]) * self.FREQUENCY)
            self.vz_discretized.append((self.z_discretized[s+1] - self.z_discretized[s]) * self.FREQUENCY)
            self.ax_discretized.append((self.vx_discretized[-1] - self.vx_discretized[-2]) * self.FREQUENCY)
            self.ay_discretized.append((self.vy_discretized[-1] - self.vy_discretized[-2]) * self.FREQUENCY)
            self.az_discretized.append((self.vz_discretized[-1] - self.vz_discretized[-2]) * self.FREQUENCY)
            self.ti_discretized.append((s + 1.) / self.FREQUENCY)

        if self.print_time_info: print('generate_states() runs in {} s'.format(time() - start))

    def generate_states_filtered(self):

        start = time()

        self.x_filtered = [self.x_discretized[0]]
        self.y_filtered = [self.y_discretized[0]]
        self.z_filtered = [self.z_discretized[0]]

        self.vx_filtered = [.0]
        self.vy_filtered = [.0]
        self.vz_filtered = [.0]

        self.ax_filtered = [.0]
        self.ay_filtered = [.0]
        self.az_filtered = [.0]

        for s, _ in enumerate(self.vx_discretized[1:]):
            self.ax_filtered.append(self.saturate((self.vx_discretized[s+1] - self.vx_filtered[-1]) * self.FREQUENCY, self.MAX_LINEAR_ACC_XY))
            self.ay_filtered.append(self.saturate((self.vy_discretized[s+1] - self.vy_filtered[-1]) * self.FREQUENCY, self.MAX_LINEAR_ACC_XY))
            self.az_filtered.append(self.saturate((self.vz_discretized[s+1] - self.vz_filtered[-1]) * self.FREQUENCY, self.MAX_LINEAR_ACC_Z))

            self.vx_filtered.append(self.saturate(self.vx_filtered[-1] + (self.ax_filtered[-1] / self.FREQUENCY), self.MAX_LINEAR_SPEED_XY))
            self.vy_filtered.append(self.saturate(self.vy_filtered[-1] + (self.ay_filtered[-1] / self.FREQUENCY), self.MAX_LINEAR_SPEED_XY))
            self.vz_filtered.append(self.saturate(self.vz_filtered[-1] + (self.az_filtered[-1] / self.FREQUENCY), self.MAX_LINEAR_SPEED_Z))

            self.x_filtered.append(self.x_filtered[-1] + (self.vx_filtered[-1] / self.FREQUENCY))
            self.y_filtered.append(self.y_filtered[-1] + (self.vy_filtered[-1] / self.FREQUENCY))
            self.z_filtered.append(self.z_filtered[-1] + (self.vz_filtered[-1] / self.FREQUENCY))

        self.is_filtered = True

        if self.print_time_info: print('generate_states_filtered() runs in {} s'.format(time() - start))

    def generate_states_sg_filtered(self, window_length=51, polyorder=3, deriv=0, delta=1.0, mode='mirror', on_filtered=False):
        # Info: Apply Savitzky-Golay filter to velocities
        start = time()

        self.x_filtered = [self.x_discretized[0]]
        self.y_filtered = [self.y_discretized[0]]
        self.z_filtered = [self.z_discretized[0]]

        if on_filtered:
            self.vx_filtered = signal.savgol_filter(x=self.vx_filtered, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode=mode)
            self.vy_filtered = signal.savgol_filter(x=self.vy_filtered, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode=mode)
            self.vz_filtered = signal.savgol_filter(x=self.vz_filtered, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode=mode)
        else:
            self.vx_filtered = signal.savgol_filter(x=self.vx_discretized, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode=mode)
            self.vy_filtered = signal.savgol_filter(x=self.vy_discretized, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode=mode)
            self.vz_filtered = signal.savgol_filter(x=self.vz_discretized, window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode=mode)

        self.ax_filtered = [.0]
        self.ay_filtered = [.0]
        self.az_filtered = [.0]

        for s, _ in enumerate(self.vx_filtered[1:]):
            self.ax_filtered.append(self.saturate((self.vx_filtered[s+1] - self.vx_filtered[s]) * self.FREQUENCY, self.MAX_LINEAR_ACC_XY))
            self.ay_filtered.append(self.saturate((self.vy_filtered[s+1] - self.vy_filtered[s]) * self.FREQUENCY, self.MAX_LINEAR_ACC_XY))
            self.az_filtered.append(self.saturate((self.vz_filtered[s+1] - self.vz_filtered[s]) * self.FREQUENCY, self.MAX_LINEAR_ACC_Z))

            self.vx_filtered[s+1] = self.vx_filtered[s] + (self.ax_filtered[-1] / self.FREQUENCY)
            self.vy_filtered[s+1] = self.vy_filtered[s] + (self.ay_filtered[-1] / self.FREQUENCY)
            self.vz_filtered[s+1] = self.vz_filtered[s] + (self.az_filtered[-1] / self.FREQUENCY)

            self.x_filtered.append(self.x_filtered[-1] + (self.vx_filtered[s+1] / self.FREQUENCY))
            self.y_filtered.append(self.y_filtered[-1] + (self.vy_filtered[s+1] / self.FREQUENCY))
            self.z_filtered.append(self.z_filtered[-1] + (self.vz_filtered[s+1] / self.FREQUENCY))

        self.is_filtered = True

        if self.print_time_info: print('generate_states_sg_filtered() runs in {} s'.format(time() - start))

    def generate_yaw_filtered(self):

        if not self.is_filtered:
            return

        start = time()

        self.ya_filtered = []

        prevHeading = np.array([.0, .0, .0])

        for s, _ in enumerate(self.vx_filtered[1:]):
            p1 = np.array([self.x_filtered[s], self.y_filtered[s]])
            p2 = np.array([self.x_filtered[s+1], self.y_filtered[s+1]])

            if self.ya_info[s][0] == 'center':
                heading = np.array(self.ya_info[s][1]) - p1
            elif self.ya_info[s][0] == 'axes':
                heading = np.array(self.ya_info[s][1])
            elif self.ya_info[s][0] == 'still':
                heading = prevHeading
            else:
                heading = p2 - p1

            if (np.linalg.norm(heading) < 0.001) or (self.ya_info[s][0] == 'still'):
                heading = prevHeading
            else:
                heading = heading / np.linalg.norm(heading)
                prevHeading = heading

            self.ya_filtered.append(math.atan2(heading[1], heading[0]))

        self.ya_filtered.append(self.ya_filtered[-1])

        cos_ya = [math.cos(yaw) for yaw in self.ya_filtered]
        sin_ya = [math.sin(yaw) for yaw in self.ya_filtered]

        cos_ya = signal.savgol_filter(x=cos_ya, window_length=53, polyorder=1, deriv=0, delta=1.0, mode='mirror')
        sin_ya = signal.savgol_filter(x=sin_ya, window_length=53, polyorder=1, deriv=0, delta=1.0, mode='mirror')

        self.ya_filtered = []

        for s, _ in enumerate(cos_ya):
            self.ya_filtered.append(math.atan2(sin_ya[s], cos_ya[s]))

        if self.print_time_info: print('generate_yaw_filtered() runs in {} s'.format(time() - start))

    def plot_trajectory_extras(self):

        start = time()

        n = int(self.FREQUENCY / self.PUBLISH_RATE)
        alpha = .3  # Transparancy for velocity and heading arrows

        ti = self.ti_filtered if hasattr(self, 'ti_filtered') else self.ti_discretized

        fig = plt.figure(figsize=(16, 8))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(self.x_discretized[0::n], self.y_discretized[0::n], self.z_discretized[0::n], label='trajectory_desired', color='blue')
        if self.is_filtered:
            ax1.scatter(self.x_filtered, self.y_filtered, self.z_filtered, label='trajectory_filtered', color='red')
            ax1.quiver(
                self.x_filtered[0::n], self.y_filtered[0::n], self.z_filtered[0::n],
                self.vx_filtered[0::n], self.vy_filtered[0::n], self.vz_filtered[0::n],
                length=.05, color='red', alpha=alpha, label='velocity_filtered')
            if hasattr(self, 'ya_filtered'):
                ax1.quiver(
                    self.x_filtered[0::n], self.y_filtered[0::n], self.z_filtered[0::n],
                    [math.cos(a) for a in self.ya_filtered[0::n]], [math.sin(a) for a in self.ya_filtered[0::n]], [.0 for a in self.ya_filtered[0::n]],
                    length=.3, color='green', alpha=alpha, label='heading_filtered')
            else:
                ax1.quiver(
                    self.x_filtered[0::n], self.y_filtered[0::n], self.z_filtered[0::n],
                    [math.cos(a) for a in self.ya_discretized[0::n]], [math.sin(a) for a in self.ya_discretized[0::n]], [.0 for a in self.ya_discretized[0::n]],
                    length=.3, color='green', alpha=alpha, label='heading_discretized')
        else:
            ax1.quiver(
                self.x_discretized[0::n], self.y_discretized[0::n], self.z_discretized[0::n],
                self.vx_discretized[0::n], self.vy_discretized[0::n], self.vz_discretized[0::n],
                length=.05, color='red', alpha=alpha, label='velocity')
            ax1.quiver(
                self.x_discretized[0::n], self.y_discretized[0::n], self.z_discretized[0::n],
                [math.cos(a) for a in self.ya_discretized[0::n]], [math.sin(a) for a in self.ya_discretized[0::n]], [.0 for a in self.ya_discretized[0::n]],
                length=.3, color='green', alpha=alpha, label='heading')
        plt.legend()
        plt.title('Trajectory')

        ax2 = fig.add_subplot(322)
        ax2.plot(self.ti_discretized, self.vx_discretized, color='red', label='vx_desired')
        ax2.plot(self.ti_discretized, self.vy_discretized, color='green', label='vy_desired')
        ax2.plot(self.ti_discretized, self.vz_discretized, color='blue', label='vz_desired')
        if self.is_filtered:
            ax2.plot(ti, self.vx_filtered, color='red', label='vx_filtered', linestyle='--')
            ax2.plot(ti, self.vy_filtered, color='green', label='vy_filtered', linestyle='--')
            ax2.plot(ti, self.vz_filtered, color='blue', label='vz_filtered', linestyle='--')
        plt.legend()
        plt.title('Velocity')

        ax3 = fig.add_subplot(324)
        ax3.plot(self.ti_discretized, self.ax_discretized, color='red', label='ax_desired')
        ax3.plot(self.ti_discretized, self.ay_discretized, color='green', label='ay_desired')
        ax3.plot(self.ti_discretized, self.az_discretized, color='blue', label='az_desired')
        if self.is_filtered:
            ax3.plot(ti, self.ax_filtered, color='red', label='ax_filtered', linestyle='--')
            ax3.plot(ti, self.ay_filtered, color='green', label='ay_filtered', linestyle='--')
            ax3.plot(ti, self.az_filtered, color='blue', label='az_filtered', linestyle='--')
        ax3.set_ylim([-max(self.MAX_LINEAR_ACC_XY, self.MAX_LINEAR_ACC_Z), max(self.MAX_LINEAR_ACC_XY, self.MAX_LINEAR_ACC_Z)])
        plt.legend()
        plt.title('Acceleration')

        ax4 = fig.add_subplot(326)
        ax4.plot(self.ti_discretized, self.ya_discretized, color='blue', marker='o', markersize='1.', linestyle='None', label='ya_desired')
        if hasattr(self, 'ya_filtered'):
            ax4.plot(ti, self.ya_filtered, color='red', marker='o', markersize='1.', linestyle='None', label='ya_filtered')
        plt.legend()
        plt.title('Yaw')

        if self.print_time_info: print('plot_trajectory_extras_filtered() runs in {} s'.format(time() - start))

        fig.tight_layout()
        plt.ion()
        plt.show()
        plt.pause(.001)

    def start(self):

        rate = rospy.Rate(self.PUBLISH_RATE)
        ratio = int(self.FREQUENCY / self.PUBLISH_RATE)
        window_points = self.WINDOW_FRAME * self.FREQUENCY
        time_inc = 1. / self.PUBLISH_RATE
        i = 0
        s = 0

        x = self.x_filtered if hasattr(self, 'x_filtered') else self.x_discretized
        y = self.y_filtered if hasattr(self, 'y_filtered') else self.y_discretized
        z = self.z_filtered if hasattr(self, 'z_filtered') else self.z_discretized
        ya = self.ya_filtered if hasattr(self, 'ya_filtered') else self.ya_discretized
        vx = self.vx_filtered if hasattr(self, 'vx_filtered') else self.vx_discretized
        vy = self.vy_filtered if hasattr(self, 'vy_filtered') else self.vy_discretized
        vz = self.vz_filtered if hasattr(self, 'vz_filtered') else self.vz_discretized
        ax = self.ax_filtered if hasattr(self, 'ax_filtered') else self.ax_discretized
        ay = self.ay_filtered if hasattr(self, 'ay_filtered') else self.ay_discretized
        az = self.az_filtered if hasattr(self, 'az_filtered') else self.az_discretized
        ti = self.ti_filtered if hasattr(self, 'ti_filtered') else self.ti_discretized

        stamp = rospy.get_rostime()

        while not (rospy.is_shutdown() or s >= len(x)-window_points):
            # Build JointTrajectory message
            header = Header()
            header.seq = s
            header.stamp = stamp
            header.frame_id = 'map'

            joint_trajectory_msg = JointTrajectory()
            joint_trajectory_msg.header = header
            joint_trajectory_msg.joint_names = ['base_link']

            points_in_next_trajectory = int(min(window_points, len(x)-s))

            for i in range(points_in_next_trajectory):
                joint_trajectory_point = JointTrajectoryPoint()
                joint_trajectory_point.positions = [x[s+i], y[s+i], z[s+i], ya[s+i]]
                joint_trajectory_point.velocities = [vx[s+i], vy[s+i], vz[s+i]]
                joint_trajectory_point.accelerations = [ax[s+i], ay[s+i], az[s+i]]
                joint_trajectory_point.effort = []
                joint_trajectory_point.time_from_start = rospy.Duration.from_sec(float(i) / self.FREQUENCY)

                joint_trajectory_msg.points.append(joint_trajectory_point)

            stamp += rospy.Duration(time_inc)
            s += ratio

            self.pub.publish(joint_trajectory_msg)
            rate.sleep()

    def saturate(self, x, y):

        return math.copysign(min(x, y, key=abs), x)

    def wait_drone_armed(self):
        rospy.loginfo("[trajectory_gen] Waiting for the drone to be armed ...")
        while not (rospy.is_shutdown() or self.drone_state.armed):
            pass
        rospy.loginfo("[trajectory_gen] Drone armed.")

    def wait_drone_offboard(self):
        rospy.loginfo("[trajectory_gen] Waiting for offboard mode ...")
        while not (rospy.is_shutdown() or (self.drone_state.mode == "OFFBOARD")):
            pass
        rospy.loginfo("[trajectory_gen] Offboard enabled.")

    def callback(self, odom):

        if not self.is_first_callback:

            position = odom.pose.pose.position

            self.x_discretized = [position.x] * self.FREQUENCY
            self.y_discretized = [position.y] * self.FREQUENCY
            self.z_discretized = [position.z] * self.FREQUENCY

            self.is_first_callback = True

    def check_callback(self):

        rospy.loginfo("[trajectory_gen] Waiting for position measurement callback ...")
        while not (rospy.is_shutdown() or self.is_first_callback):
            pass
        rospy.loginfo("[trajectory_gen] Position measurement callback ok.")

    def drone_state_acquire_callback(self, state):
        self.drone_state = state


    def rdm_sequence(self, nb_actions=4, seed=None, max_vel=0.6, max_x=1.5, max_y=1.5, max_z=2):
        np.random.seed(seed)

        num_action = np.random.randint(0, 2, nb_actions)
        speed = np.random.randint(1, max_vel*10)/10.0

        coord = np.zeros((nb_actions, 3))

        for i in range(nb_actions):
            coord[i, :] = [np.random.randint(-max_x*4, max_x*4)/4.0, np.random.randint(-max_y*4, max_y*4)/4.0,
                           np.random.randint(2, max_z*4)/4.0]
        print(coord)
        self.gen_sequence(num_action, coord, speed)

    def gen_sequence(self, num_action, coord, speed):

        actions = ['vector', 'circle']
        # actions = ['vector', 'circle', 'circlexz', 'circleyz', 'spiral']

        trajectory_object.discretise_trajectory(parameters=['takeoff', 1.0], velocity=0.6)
        trajectory_object.discretise_trajectory(parameters=['hover', 3.], heading=['still'])

        for i in range(len(num_action)):

            if actions[num_action[i]] == 'circle':
                #coord[i, 2] = self.z_trj[-1]
                param = ['circle', coord[i, :], 1]
            elif actions[num_action[i]] == 'vector':
                param = ['vector', coord[i, :]]

            trajectory_object.discretise_trajectory(parameters=param, velocity=speed)
            trajectory_object.discretise_trajectory(parameters=['hover', 3.])
        trajectory_object.discretise_trajectory(parameters=['landing'])


if __name__ == '__main__':

    node_name = 'trajectory_gen_node'
    subscriber = 'mavros/local_position/odom'
    publisher = 'mavros/desired_trajectory'

    try:
        trajectory_object = TrajectoryGeneration(node_name=node_name, subscriber=subscriber, publisher=publisher)

        # Wait for the first measurement callback to initialize the starting position of the trajectory
        trajectory_object.check_callback()

        ########################################################################
        # Configuration
        trajectory_object.YAW_HEADING = ['auto', [1, 0]]  # options: ['auto'], ['still'], ['center', [x, y]], ['axes', [x, y]]

        trajectory_object.TRAJECTORY_REQUESTED_SPEED = 0.6 #0.6 # req. trajectory linear speed [m.s-1] (used when arg velocity in not specify in discretise_trajectory())
        trajectory_object.LANDING_SPEED = 0.3  # [m.s-1]

        trajectory_object.MAX_LINEAR_ACC_XY = 6.0  # max. linear acceleration [m.s-2]
        trajectory_object.MAX_LINEAR_ACC_Z = 5.0  # max. linear acceleration [m.s-2]
        ########################################################################

        ########################################################################
        # Trajectory definition - shape/vertices in inertial frame (x, y, z - up)
        #
        # Define trajectory by using:
        # trajectory_object.discretise_trajectory(parameters=['name', param], (opt. arg) velocity=float, (opt. arg) acceleration=float, (opt. arg) heading=[] (see YAW_HEADING), (opt. arg) relative=bool)
        #
        # Possible parameters:
        # parameters=['takeoff', z] with z in meters
        # parameters=['hover', time] with time in seconds
        # parameters=['vector', [x, y, z]] with x, y, z the target position
        # parameters=['circle', [x, y, z], (opt.) n] with x, y, z the center of the circle and n (optional) the number of circle. Circle
        # is defined by the drone position when starting the circle trajectory and the center. The drone will turn around this point.
        # parameters=['landing']
        # parameters=['returnhome']
        #
        # Optional argument:
        # velocity=float (desired velocity for trajectory, positive only)
        # velocitymax=float (maximum velocity for trajectory, positive only)
        # acceleration=float (acceleration during trajectory, can be positive or negative, be careful with negative acceleration as it can lead to impossible trajectory)
        # heading=options with options: ['auto'], ['still'], ['center', [x, y]], ['axes', [x, y]]
        # relative=bool (if True: trajectory relative to the position of the drone)
        # vend=bool (if True: keep the desired velocity at the end of the trajectory)

        vel = trajectory_object.TRAJECTORY_REQUESTED_SPEED
        ech_val = 1

        # for i in range(5):
        #     trajectory_object.rdm_sequence(nb_actions=5, seed=i+150-100, max_vel=vel)

        trajectory_object.discretise_trajectory(parameters=['takeoff', 1], velocity=vel, heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])

        # # Echelons sur z
        # trajectory_object.discretise_trajectory(parameters=['vector', [0, 0, 1.5]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['vector', [0, 0, 0.5]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['vector', [0, 0, 1.5]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['vector', [0, 0, 1]], heading=['still'])

        # Echelon y
        trajectory_object.discretise_trajectory(parameters=['vector_ech', [0, ech_val, 1]], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector_ech', [0, 0, 1]], heading=['still'])

        # Echelon x
        trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector_ech', [ech_val, 0, 1]], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector_ech', [0, 0, 1]], heading=['still'])

        # Echelon xy
        trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector_ech', [ech_val, ech_val, 1]], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector_ech', [0, 0, 1]], heading=['still'])

        # Retour a 0.5m
        trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector', [0, 0, 0.5]], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])

        # Echelon xyz
        trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector', [ech_val, ech_val, 0.5+ech_val]], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 4.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector', [0, 0, 1]], heading=['still'])
        #
        # Cercle
        trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['circle', [0, 0.5, 1], 2], velocity=vel, heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 5], heading=['still'])

        # Spirale montee
        trajectory_object.discretise_trajectory(parameters=['vector', [0., -.3, 1.]], velocity=vel, heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 5], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector', [0., -.3, 0.5]], velocity=vel, heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['circle', [.0, 0.5, 0.], 2, 1], velocity=vel, acceleration=0.1, heading=['still'], relative=True)
        trajectory_object.discretise_trajectory(parameters=['hover', 5.])

        # Spirale descente
        trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['circle', [.0, 0.5, 0.], 2, -1], velocity=vel, acceleration=0.1, heading=['still'], relative=True)
        trajectory_object.discretise_trajectory(parameters=['hover', 5.])

        # Lemniscate
        trajectory_object.discretise_trajectory(parameters=['hover', 4], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['lemniscate', [0, 0, 1], 0.9, 0], velocity=vel, heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 4], heading=['still'])

        # Lemniscate
        trajectory_object.discretise_trajectory(parameters=['hover', 4], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['lemniscate', [0, 0, 1], 0.8, 1], velocity=vel, heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 4], heading=['still'])

        # Wave
        trajectory_object.discretise_trajectory(parameters=['hover', 3.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector', [1.5, -1.5, 1.5]], velocity=0.6, heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 3.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['wave', [-1.5, 1.5, 1.5], 0.1, 0.5], velocity=1., heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 3.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector', [0., 0., 1.5]], velocity=0.6, heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 3.])

        # Square
        trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector', [1., -.5, 1.]], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector', [1., 1.5, 1.]], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector', [-1., 1.5, 1.]], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector', [-1., -.5, 1.]], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['vector', [0., -.5, 1.]], heading=['still'])
        trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])

        # # Forme sablier
        # trajectory_object.discretise_trajectory(parameters=['vector', [0., 0., 1.]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['vector', [0.5, 0, 1.]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['vector', [0., 1, 1.]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['vector', [0.5, 1, 1.]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['vector', [0., 0., 1.]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])

        # # Forme sablier hauteur
        # trajectory_object.discretise_trajectory(parameters=['vector', [0., 0., 1.]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['vector', [0.5, 0, 0.5]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['vector', [0., 1, 1.5]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['vector', [0.5, 1, 0.5]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['vector', [0., 0., 1.]], heading=['still'])
        # trajectory_object.discretise_trajectory(parameters=['hover', 5.], heading=['still'])

        # Landing
        trajectory_object.discretise_trajectory(parameters=['landing'])

        ########################################################################

        # Limit the trajectory to the BOX_LIMIT
        # trajectory_object.constraint_trajectory_to_box()

        # Generate the list of states - start by generating the states and then filter them
        trajectory_object.generate_states()
        # trajectory_object.generate_states_sg_filtered(window_length=53, polyorder=1, mode='mirror')
        # trajectory_object.generate_states_sg_filtered(window_length=13, polyorder=1, mode='mirror', on_filtered=True)
        trajectory_object.generate_yaw_filtered()

        if not rospy.is_shutdown():
            # Plot the trajectory
            trajectory_object.plot_trajectory_extras()

            # Checks before publishing
            trajectory_object.wait_drone_armed()
            trajectory_object.wait_drone_offboard()

            # Publish trajectory states
            trajectory_object.start()

    except rospy.ROSInterruptException:
        pass

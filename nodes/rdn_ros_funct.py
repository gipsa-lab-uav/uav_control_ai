#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Functions for NN & ROS

"""

import numpy as np
import rotations_tools as rt
import scipy
from matplotlib import pyplot as plt

def derivative(x, a, time=[]):
    y = [0]*len(x)
    y[0] = a * x[0]
    for i in range(len(x)):
        if time != []:
            a = time[i+1]-time[i]
        y[i] = (x[i] -x[i-1])/a
    return np.asarray(y)

def readBagTopicList(bag):
    """
    Read and save the initial topic list from bag
    """
    print("[OK] Reading topics in this bag ...")
    topicList = []
    for topic, msg, t in bag.read_messages():
        if topicList.count(topic) == 0:
            topicList.append(topic)

    print('{} topics found:'.format(len(topicList)))
    return topicList

def lowpass(x, a):
    y = [0]*len(x)

    y[0] = a * x[0]
    for i in range(len(x)):
        y[i] = a * x[i] + (1-a) * y[i-1]
        y[i] = a * x[i] + (1-a) * y[i-1]
    return y


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

def get_data_range(bag, hover_coef):
    # Automatic data range selection
    if hover_coef == -1:
        hover_coef = get_hover_coef(bag)

    time_v = []
    start_time = 0
    total_time = 0
    end_time = 0

    for topic1, msg1, t1 in bag.read_messages(topics=['/uav_control_ai/statecommanddata']):
        total_time = total_time + 1
        if msg1.thrust_cmd != hover_coef:
            start_time = total_time
            break

    total_time = 0
    for topic1, msg1, t1 in bag.read_messages(topics=['/uav_control_ai/statecommanddata']):
        total_time = total_time + 1
        end_time = total_time

    time_v = []
    for topic, msg, t in bag.read_messages(topics=['/uav_control_ai/statecommanddata']):
        time_v.append(msg.time)

    time_info_s = time_v[-1] - time_v[0]
    # print('Temps = ', round(time_v[-1]-time_v[0], 2), '(s) = ', round((time_v[-1]-time_v[0])/60,2), ' (min) ')

    return [start_time, end_time-start_time], time_info_s

def get_positions_data(bag, data_range=[0, -1], hover_coef=-1):
    if hover_coef == -1:
        hover_coef = get_hover_coef(bag)

    if data_range==[0, -1]:
        data_range, _ = get_data_range(bag, hover_coef)

    x = []
    y = []
    z = []
    x_ref = []
    y_ref = []
    z_ref = []

    for _, msg, _ in bag.read_messages(topics=['/uav_control_ai/statecommanddata']):
        # Positions
        x.append(msg.positions[0])
        y.append(msg.positions[1])
        z.append(msg.positions[2])

        # References
        x_ref.append(msg.positions_ref[0])
        y_ref.append(msg.positions_ref[1])
        z_ref.append(msg.positions_ref[2])
    return np.array([x, y, z, x_ref, y_ref, z_ref])

def get_hover_coef(bag):
    for topic1, msg, t in bag.read_messages(topics=['/uav_control_ai/statecommanddata']):
        hover_coef = msg.thrust_cmd
        break
    return hover_coef

def get_data(bag, hover_coef=-1, data_range=[0, -1], filter_imu=1, battery_v=0):
    def lowpass(x, a):
        y = [0]*len(x)
        y[0] = a * x[0]
        for i in range(len(x)):
            y[i] = a * x[i] + (1-a) * y[i-1]
            y[i] = a * x[i] + (1-a) * y[i-1]
        return y

    if hover_coef == -1:
        hover_coef = get_hover_coef(bag)

    if data_range == [0, -1]:
        data_range, time_info_s = get_data_range(bag, hover_coef)
    # print('Data range :', data_range)
    x = []
    y = []
    z = []
    x_ref = []
    y_ref = []
    z_ref = []
    phi = []
    theta = []
    psi = []
    thrust = []
    vx = []
    vy = []
    vz = []
    ax = []
    ay = []
    az = []
    ax_mod = []
    ay_mod = []
    az_mod = []
    time_v = []
    x_vicon = []
    y_vicon = []
    z_vicon = []
    batt_v = []

    for topic1, msg, t in bag.read_messages(topics=['/uav_control_ai/statecommanddata']):
        # Positions
        x.append(msg.positions[0])
        y.append(msg.positions[1])
        z.append(msg.positions[2])
        x_vicon.append(msg.positions_vicon[0])
        y_vicon.append(msg.positions_vicon[1])
        z_vicon.append(msg.positions_vicon[2])

        # References
        x_ref.append(msg.positions_ref[0])
        y_ref.append(msg.positions_ref[1])
        z_ref.append(msg.positions_ref[2])

        # Euler angles
        phi.append(msg.euler_angles[0])
        theta.append(msg.euler_angles[1])
        psi.append(msg.euler_angles[2])

        # Thrust
        thrust.append(msg.thrust_cmd)

        # Speed
        vx.append(msg.velocities_linear[0])
        vy.append(msg.velocities_linear[1])
        vz.append(msg.velocities_linear[2])

        # Battery voltage
        if battery_v==1:
            batt_v.append(msg.battery)

        # Accelerations
        if msg.accelerations_linear != ():
            ax.append(msg.accelerations_linear[0])
            ay.append(msg.accelerations_linear[1])
            az.append(msg.accelerations_linear[2])
        else:
            ax.append(150)
            ay.append(150)
            az.append(150)

        time_v.append(msg.time)

    if data_range != [0, -1]:
        # print('La: ',data_range)
        time_info_s = time_v[data_range[1]-1] - time_v[data_range[0]]

    count_filt = 0
    for i in range(2, len(az)):
        if abs(az[i]) >= 25:
            count_filt+=1
            az[i] = az[i-1]
    for i in range(2, len(ay)):
        if abs(ay[i]) >= 25:
            count_filt+=1
            ay[i] = ay[i-1]
    for i in range(2, len(ax)):
        if abs(ax[i]) >= 25:
            count_filt+=1
            ax[i] = ax[i-1]

    az_raw = az
    ay_raw = ay
    ax_raw = ax

    if filter_imu == 1:
        az = np.asarray(lowpass(az_raw, 0.02))
        ay = np.asarray(lowpass(ay_raw, 0.02))
        ax = np.asarray(lowpass(ax_raw, 0.02))

    a_rot = np.zeros((3, len(az)))
    a_rot_raw = np.zeros((3, len(az)))
    for i in range(len(az)):
        a_rot[:, i] = np.matmul(rt.rot_m(psi[i], theta[i], phi[i], 'ZYX'),
                                np.array([[ax[i], ay[i], az[i]]]).T).reshape(-1)
        a_rot_raw[:, i] = np.matmul(rt.rot_m(psi[i], theta[i], phi[i], 'ZYX'),
                                    np.array([[ax_raw[i], ay_raw[i], az_raw[i]]]).T).reshape(-1)
    ax = a_rot[0, :]
    ay = a_rot[1, :]
    az = a_rot[2, :]
    ax_raw = a_rot_raw[0, :]
    ay_raw = a_rot_raw[1, :]
    az_raw = a_rot_raw[2, :]

    x = x[data_range[0]:data_range[1]]
    y = y[data_range[0]:data_range[1]]
    z = z[data_range[0]:data_range[1]]
    x_ref = x_ref[data_range[0]:data_range[1]]
    y_ref = y_ref[data_range[0]:data_range[1]]
    z_ref = z_ref[data_range[0]:data_range[1]]
    phi = phi[data_range[0]:data_range[1]]
    theta = theta[data_range[0]:data_range[1]]
    psi = psi[data_range[0]:data_range[1]]
    thrust = thrust[data_range[0]:data_range[1]]
    ax = ax[data_range[0]:data_range[1]]
    ay = ay[data_range[0]:data_range[1]]
    az = az[data_range[0]:data_range[1]]
    ax_raw = ax_raw[data_range[0]:data_range[1]]
    ay_raw = ay_raw[data_range[0]:data_range[1]]
    az_raw = az_raw[data_range[0]:data_range[1]]
    vx = vx[data_range[0]:data_range[1]]
    vy = vy[data_range[0]:data_range[1]]
    vz = vz[data_range[0]:data_range[1]]
    time_v = time_v[data_range[0]:data_range[1]]
    if battery_v==1:
        batt_v = batt_v[data_range[0]:data_range[1]]

    for i in range(len(theta)):
        ax_mod.append(9.81*theta[i])
        ay_mod.append(-9.81*phi[i])
        az_mod.append(thrust[i]*(1.5*9.81/hover_coef/1.5))


    #### Acceleration from vicon
    ax_vicon = savitzky_golay(np.asarray(x_vicon), 51, 3, 2)/(0.01*0.01)
    ay_vicon = savitzky_golay(np.asarray(y_vicon), 51, 3, 2)/(0.01*0.01)
    az_vicon = savitzky_golay(np.asarray(z_vicon), 51, 3, 2)/(0.01*0.01)+9.81

    ax_vicon = ax_vicon[data_range[0]:data_range[1]]
    ay_vicon = ay_vicon[data_range[0]:data_range[1]]
    az_vicon = az_vicon[data_range[0]:data_range[1]]

    z_vicon = z_vicon[data_range[0]:data_range[1]]

    plt.figure()
    plt.plot(x)
    plt.plot(y)
    plt.plot(z)
    plt.figure()
    plt.plot(ax)
    plt.plot(ay)
    plt.plot(az)

    if battery_v==0:
        x_train = np.array([np.cos(phi), np.cos(theta), np.cos(psi),
                            np.sin(phi), np.sin(theta), np.sin(psi),
                            np.asarray(vx), np.asarray(vy), np.asarray(vz),
                            np.asarray(thrust), np.asarray(z_vicon)]).T
    else:
        x_train = np.array([np.cos(phi), np.cos(theta), np.cos(psi),
                            np.sin(phi), np.sin(theta), np.sin(psi),
                            np.asarray(vx), np.asarray(vy), np.asarray(vz),
                            np.asarray(thrust), np.asarray(z_vicon), np.asarray(batt_v)]).T
    if filter_imu==2:
        y_train = np.array([ax_vicon-np.asarray(ax_mod),
                            ay_vicon-np.asarray(ay_mod),
                            az_vicon-np.asarray(az_mod)]).T
    else:
        y_train = np.array([np.asarray(ax)-np.asarray(ax_mod),
                            np.asarray(ay)-np.asarray(ay_mod),
                            np.asarray(az)-np.asarray(az_mod)]).T
    nb_data = x_train.shape
    infos = np.array([[time_info_s, nb_data[0], count_filt]])

    return x_train, y_train, infos

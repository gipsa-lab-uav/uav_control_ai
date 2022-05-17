#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to provide tools for rotations

"""
import numpy as np

def rot_m(angle1, angle2, angle3, sequence):
    """ Return Rotation Matrix """
    # Tait-Bryan Angles
    if sequence == 'ZYX':
        rm1 = rot_z(angle1)
        rm2 = rot_y(angle2)
        rm3 = rot_x(angle3)
        rot_matrix = np.matmul(np.matmul(rm1, rm2), rm3)
    else:
        rot_matrix = np.eye(3)
        print('erreur')
    return rot_matrix

def rot_x(roll_angle):
    """ Perform rotation around x axis by roll_angle"""
    c_a = np.cos(roll_angle)
    s_a = np.sin(roll_angle)
    rot_matrix = np.array([[1, 0, 0],\
                   [0, c_a, -s_a],\
                   [0, s_a, c_a]])
    return rot_matrix

def rot_y(pitch_angle):
    """ Perform rotation around y axis by pitch_angle"""
    c_a = np.cos(pitch_angle)
    s_a = np.sin(pitch_angle)
    rot_matrix = np.array([[c_a, 0, s_a],\
                   [0, 1, 0],\
                   [-s_a, 0, c_a]])
    return rot_matrix

def rot_z(yaw_angle):
    """ Perform rotation around z axis by yaw_angle"""
    c_a = np.cos(yaw_angle)
    s_a = np.sin(yaw_angle)
    rot_matrix = np.array([[c_a, -s_a, 0],\
                   [s_a, c_a, 0],\
                   [0, 0, 1]])
    return rot_matrix

def t_matrix(phi, theta, psi, sequence):
    """ Return t_matrix to convert angle rate to angular velocity"""
    if sequence == 'ZYX':
        t_m = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],\
                      [0, np.cos(phi), -np.sin(phi)],\
                      [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])
    else:
        t_m = np.eye(3)
    return t_m

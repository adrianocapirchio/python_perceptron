# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:17:18 2017

@author: Alex
"""

import numpy as np
import scipy as scy

def gaussian(x, mu, std_dev):    
    dist = (x - mu) ** 2
    den = 2 * std_dev ** 2
    return np.exp(- dist / den)

def sigmoid(x):
    return 1 / (1.0 + clipped_exp(-(2.0*x)))

def clipped_exp(x):
    cx =np.clip(x, -700, 700)
    return np.exp(cx)

def computate_noise(previous_noise, delta_time, tau): 
    C1 = delta_time / tau
    C2 = 1.1
    return previous_noise + C1 * (C2 * np.random.randn(*previous_noise.shape) - previous_noise)

def derivative(x1, x2, delta_time, tau):
    return (x1 - x2) / (delta_time / tau)

def Cut_range(x, x_min, x_high):
    return np.maximum(x_min, np.minimum(1,x))

def squared_distance(x1,x2):
    return np.absolute((x1 - x2)**2)

def Distance( x1 , x2 , y1 , y2  ):
    return np.sqrt( squared_distance(x1,x2) + squared_distance(y1,y2) )

def change_range(old_value, old_min, old_max, new_min, new_max):
    return (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

def screw(EP3d):    
    screw_anti_EP3d = np.zeros([3,3])  
    screw_anti_EP3d[0,1] = -EP3d[2]
    screw_anti_EP3d[0,2] = EP3d[1]
    screw_anti_EP3d[1,0] = EP3d[2]
    screw_anti_EP3d[1,2] = -EP3d[0]
    screw_anti_EP3d[2,0] = -EP3d[1]
    screw_anti_EP3d[2,1] = EP3d[0]
    return screw_anti_EP3d

def conversion2d(EP3d):
    conversion_matrix = np.zeros([2,3])
    conversion_matrix[0,1] = -1
    conversion_matrix[1,2] = 1                 
    x_orient = np.array([1,0,0])
    theta = np.linalg.norm(EP3d)
    screw_anti_EP3d = screw(EP3d)
    R = scy.linalg.expm(screw_anti_EP3d * theta)
    e1 = np.dot(R,x_orient)
    EP2d = np.dot(e1, conversion_matrix.T)
    return EP2d

def error(EP, actual_position):
    return EP - actual_position 

def PID_controller(actual_error, previous_error, delta_time, tau): 
    Kp = 0.6
    Kd = 0.2
    force = Kp * (actual_error) + Kd * derivative(actual_error, previous_error, delta_time, tau)
    return force 

def compute_movement(EP, actual_position, previous_error, previous_velocity, previous_position, mass, delta_time, tau):            
    actual_error = error(EP,actual_position)
    actual_acceleration = PID_controller(actual_error, previous_error, delta_time, tau) / mass
    actual_velocity = previous_velocity + actual_acceleration * delta_time
    actual_position = previous_position + actual_velocity * delta_time
    actual_position = Cut_range(actual_position, 0.00001, 0.99999)
    return actual_position 

def save_movement(actual_error, actual_position, actual_velocity, actual_acceleration):
    previous_error = actual_error.copy()
    previous_position = actual_position.copy() 
    previous_velocity = actual_velocity.copy()
    previous_acceleration = actual_acceleration.copy()
    return previous_error, previous_position, previous_velocity, previous_acceleration  









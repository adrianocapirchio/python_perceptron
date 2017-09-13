# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 12:39:54 2017

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rdm
from mpl_toolkits.mplot3d import *


# WORLD PARAMETERS
world = 1.
dimensions = 3
n_interval = 10
gaussian_number = n_interval + 1 
interval_lenght = world / n_interval
X = np.zeros(gaussian_number)
std_dev = 1. / ((n_interval -1) * 2)



# TIME PARAMETERS
delta_time = 0.1
tau = 1
n_trial = 500
max_trial_movements = 3000



# AGENT'S PARAMETERS
mass = 1



# POSITION'S & MOVEMENT'S ARRAYS

# positions 
reward_position = np.zeros(3)
agent_starting_position = np.zeros(3)
actual_position = np.zeros(3)
previous_position = np.zeros(3)

# movements
actual_velocity = np.zeros(3)
actual_acceleration = np.zeros(3)
actual_noise = np.zeros(3)
previous_velocity = np.zeros(3)
previous_error = np.zeros(3)
previous_noise = np.zeros(3)
EP = np.zeros(3)
actual_error = np.zeros(3)



# STATE ARRAYS    
actual_state = np.zeros((gaussian_number , 3))
previous_state = np.zeros((gaussian_number , 3))



# ANN's PARAMETERS

# learning
a_eta = 0.01
c_eta = 0.008
discount_factor = 0.98

# input units
n_input = gaussian_number

# output units
actor_n_output = 3
critic_n_output = 1

# weights
critic_weights = np.zeros((n_input*dimensions))
x_actor_weights = np.zeros((n_input*dimensions))
y_actor_weights = np.zeros((n_input*dimensions))
z_actor_weights = np.zeros((n_input*dimensions))

# critic output parameters
actual_critic_output = np.zeros(critic_n_output)
previous_critic_output = np.zeros(critic_n_output)



# init storage arrays
number_needed_steps = np.zeros(n_trial)








# utils
def gaussian(x, mu, std_dev):    
    dist = (x - mu) ** 2
    den = 2 * std_dev ** 2
    return np.exp(- dist / den)

def sigmoid(x):
    return 1 / (1.0 + np.exp(-(2.0*x)))

def spreading(w , pattern):
    return np.dot(w, pattern)

def error(EP, actual_position):
    return EP - actual_position 

def computate_noise(previous_noise, std_dev):    
    C1 = delta_time / tau
    C2 = 0.9
    return previous_noise + C1 * (C2 * np.random.randn() - previous_noise)

def derivative(x1, x2, delta_time, tau):
    return (x1 - x2) / (delta_time / tau)

def PID_controller(actual_error, previous_error): 
    Kp = 0.6
    Kd = 0.15
    force = Kp * (actual_error) + Kd * derivative(actual_error, previous_error, delta_time, tau)
    return force 

def TDerror(actual_reward, actual_critic_output, previous_critic_output):
    x = discount_factor * actual_critic_output - previous_critic_output 
    return  x + actual_reward 

def squared_distance(x1,x2):
    return np.absolute((x1 - x2)**2)

def Distance( x1 , x2 , y1 , y2 , z1 , z2 ):
    return np.sqrt( squared_distance(x1,x2) + squared_distance(y1,y2) + squared_distance(z1,z2))

def training(eta, surprise, previous_state, previous_noise, actor_weights):    
    actor_weights += eta * surprise * previous_state * previous_noise
    return actor_weights

# main    
if __name__ == "__main__":
    
    # computate gaussian's average values
    for gn in xrange(gaussian_number):
        if gn == 0 :
            X[0] = 0    
        else:
            X[gn] = X[gn-1] + interval_lenght
             
    # place reward
    reward_position = np.array([rdm.uniform(0,1),rdm.uniform(0,1),rdm.uniform(0,1)])
    
    # init plotting
    fig1   = plt.figure("Workspace", figsize=(15,15))
    ax1    = fig1.add_subplot(111, projection='3d')
    
    # set limits
    ax1.set_xlim([0,world])
    ax1.set_ylim([0,world])
    ax1.set_zlim([0,world])
    
    # set ticks
    ax1.set_xticks(np.arange(0, world, 0.1))
    ax1.set_yticks(np.arange(0, world, 0.1))
    ax1.set_zticks(np.arange(0, world, 0.1))
    
    # add counters
    text1 = plt.figtext(0.9, 0.2, "trial = %s" % (0), style='italic', bbox={'facecolor':'green'})
    text2 = plt.figtext(0.1, 0.2, "movement = %s" % (0), style='italic', bbox={'facecolor':'red'})
    
    # plot reward
    reward, = ax1.plot([reward_position[0]] , [reward_position[1]] , [reward_position[2]] , 'x')
        
    # start trials
    for trial in xrange(n_trial):
        
        # temperature magnitude
        T = 1 * np.exp(- trial * 0.1 / float(n_trial))
        
        # place agent   
        agent_starting_position = np.array([rdm.uniform(0,1),rdm.uniform(0,1),rdm.uniform(0,1)])
        
        # plot agent
        if trial == 0:
            agent, = ax1.plot([agent_starting_position[0]], [agent_starting_position[1]], [agent_starting_position[2]], 'o', color = "red")
        
        # show plotting over without noise
        if trial > 400:
            T = 0
            text1.set_text("trial = %s" % (trial))    
            agent.remove()
            agent, = ax1.plot([agent_starting_position[0]], [agent_starting_position[1]], [agent_starting_position[2]], 'o', color = "red")
            plt.pause(0.000000001)
            
            
        # start movement
        for movement in xrange(max_trial_movements):
            
            # refresh counters
            if trial > 400:
                text2.set_text("movement = %s" % (movement))
           
           # init first movement's parameters
            if movement == 0:
                actual_position = agent_starting_position.copy()
                previous_position = actual_position.copy()
                
            # computate actual state
            actual_x_state = gaussian(actual_position[0] , X , std_dev)
            actual_y_state = gaussian(actual_position[1] , X , std_dev)
            actual_z_state = gaussian(actual_position[2] , X , std_dev)
            actual_state = np.hstack([actual_x_state,actual_y_state,actual_z_state])
            
            # compute actor output
            x_actor_output = sigmoid(np.dot(x_actor_weights, actual_state))
            y_actor_output = sigmoid(np.dot(y_actor_weights, actual_state))
            z_actor_output = sigmoid(np.dot(z_actor_weights, actual_state))
        
            # storage old noise and compute new noise
            previous_noise = actual_noise.copy()
            actual_noise[0] = computate_noise(previous_noise[0], std_dev) * T
            actual_noise[1] = computate_noise(previous_noise[1], std_dev) * T
            actual_noise[2] = computate_noise(previous_noise[2], std_dev) * T
            
            # compute EP
            EP[0] = x_actor_output  + actual_noise[0]
            EP[1] = y_actor_output  + actual_noise[1]
            EP[2] = z_actor_output  + actual_noise[2]
            
            # delete overshoot
            if EP[0] > 1:
                EP[0] = 1
                  
            if EP[0] < 0:
                EP[0] = 0
            
            if EP[1] > 1:
                EP[1] = 1
                  
            if EP[1] < 0:
                EP[1] = 0
            
            if EP[2] > 1:
                EP[2] = 1
                  
            if EP[2] < 0:
                EP[2] = 0
    
            # storage old movement values    
            if movement > 0:
                previous_error = actual_error.copy()
                previous_position = actual_position.copy() 
                previous_velocity = actual_velocity.copy()
                previous_acceleration = actual_acceleration.copy()
                
            # compute movement 
            
            actual_error = error(EP,actual_position)
            actual_acceleration = PID_controller(actual_error, previous_error) / mass
            actual_velocity = previous_velocity + actual_acceleration * delta_time
            actual_position = previous_position + actual_velocity * delta_time
         
            # set position limits
            if actual_position[0] > 1:
                actual_position[0] = 0.9999999
                  
            if actual_position[0] < 0:
                actual_position[0] = 0.0000001
            
            if actual_position[1] > 1:
                actual_position[1]= 0.9999999
                  
            if actual_position[1]< 0:
                actual_position[1] = 0.0000001
            
            if actual_position[2] > 1:
                actual_position[2] = 0.9999999
                  
            if actual_position[2] < 0:
                actual_position[2] = 0.0000001
            
            # ploting movement
            if trial > 400:
                agent.remove()
                agent, = ax1.plot([actual_position[0]], [actual_position[1]], [actual_position[2]], 'o', color = "red")
                plt.pause(0.000000001)
            
            # storage old state and compute new state
            previous_state = actual_state.copy()
            actual_x_state = gaussian(actual_position[0] , X , std_dev)
            actual_y_state = gaussian(actual_position[1] , X , std_dev)
            actual_z_state = gaussian(actual_position[2] , X , std_dev)
            actual_state = np.hstack([actual_x_state,actual_y_state,actual_z_state])
            
            
            if movement > 0:
                
                # compute reward distance
                distance = Distance(actual_position[0],reward_position[0],actual_position[1],reward_position[1],actual_position[2],reward_position[2]) 
                
                # get the reward
                if distance < 0.1:
                    print "presa"
                    print movement
                    actual_reward = 1    
                else:
                    actual_reward = 0
                
                # storage old critic output, compute new and get surprise
                previous_critic_output = actual_critic_output
                actual_critic_output = np.dot(critic_weights,actual_state)
                surprise = TDerror(actual_reward, actual_critic_output, previous_critic_output)
             
                # weights adjustement
                x_actor_weights +=  a_eta * surprise * previous_state * previous_noise[0] 
                y_actor_weights +=  a_eta * surprise * previous_state * previous_noise[1]
                z_actor_weights +=  a_eta * surprise * previous_state * previous_noise[2]
                critic_weights +=  c_eta * surprise * previous_state
            
                # storage min movements to get reward
                number_needed_steps[trial] = movement
                
                # end trial if agent got reward                   
                if actual_reward == 1:
                    break             
                
                                   
    plt.figure(figsize=(80, 4), num=4, dpi=80)
    plt.title('number of movement to get reward')
    plt.xlim([0, n_trial])
    plt.ylim([0, max_trial_movements])
    plt.xticks(np.arange(0,n_trial, 100))
    plt.plot(number_needed_steps)
            
            
           
           
          
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

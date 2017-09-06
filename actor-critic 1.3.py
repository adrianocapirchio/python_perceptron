# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 19:03:07 2017

@author: Alex

"""

import numpy as np
import matplotlib.pyplot as plt
import random as rdm

# init world's parameters 
world = 1.
n_interval = 100
gaussian_number = n_interval + 1 
interval_lenght = world / n_interval
X = np.zeros(gaussian_number)
std_dev = 0.2
#std_dev = 1 / ((n_interval -1) * 2)

# init time's parameters
#simulation_duration = 1
delta_time = 0.01
#steps = simulation_duration / delta_time
tau = 0.1

# init agent's parameters
mass = 1

# init position's arrays        
actual_state = np.zeros(gaussian_number, dtype=np.uint8)

# init ANN's parameters
eta = 0.05
discount_factor = 0.99

# input parameters
n_input = gaussian_number
critic_weights = np.zeros(n_input)
actor_weights = np.zeros(n_input)

# critic output parameters
critic_n_output = 1
actual_critic_output = np.zeros(critic_n_output)
previous_critic_output = np.zeros(critic_n_output)

# actor output parameters
actor_n_output = 1
actor_output = np.zeros(actor_n_output)

n_trial = 30
max_trial_movements = 300

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
    C2 = 1.7
    return previous_noise + C1 * (C2 * np.random.randn() - previous_noise)

def derivative(x1, x2, delta_time, tau):
    return (x1 - x2) / (delta_time / tau)

def PID_controller(actual_error, previous_error): 
    Kp = 21
    Kd = 7
    force = Kp * (actual_error) + Kd * derivative(actual_error, previous_error, delta_time, tau)
    return force 

def TDerror(actual_reward, actual_critic_output, previous_critic_output):
    return actual_reward + discount_factor * actual_critic_output - previous_critic_output  

# main    
if __name__ == "__main__":
    
    # computate gaussian's average values
    for gn in xrange(gaussian_number):
        
        if gn == 0 :
            X[0] = 0
            
        else:
            X[gn] = X[gn-1] + interval_lenght
    
    # reward position         
    reward_position =  rdm.uniform(0,1)
    
    # init plotting
    fig1   = plt.figure("Workspace",figsize=(80,3))
    ax1    = fig1.add_subplot(111)
    ax1.set_xlim([0,world])
    ax1.set_ylim([-3,3])
    plt.xticks(np.arange(0, world, 0.02))
    
    reward, = ax1.plot([reward_position], [0], 'x')
    
    text1 = ax1.text(0.9, 2, "trial = %s" % (0), style='italic', bbox={'facecolor':'green'})
    text2 = ax1.text(0.9, 1.5, "movement = %s" % (0), style='italic', bbox={'facecolor':'red'})
    
    # init trials
    for trial in xrange(n_trial):
        
        # temperature magnitude
        T = 1 * np.exp(- trial * 0.1 / float(n_trial))
     #   print "T"
       # print T
        # init starting trial's position
        agent_starting_position = rdm.uniform(0,1)
        
     #   actual_reward = 0
        
        # plotting first trial agent's starting position
        if trial == 0:
            
            agent, = ax1.plot([agent_starting_position], [0], 'o')
        
        # plotting other trials agent's starting position
        else:
            
            text1.set_text("trial = %s" % (trial))
            agent.set_data(agent_starting_position ,0)
            plt.pause(0.1)
            
        # init movements
        for movement in xrange(max_trial_movements):
            
            text2.set_text("movement = %s" % (movement))
            
            # init first movement's parameters
            if movement == 0:
                
                actual_position = agent_starting_position
                previous_position = actual_position
                actual_velocity = 0
                actual_acceleration = 0
                actual_noise = 0
                previous_velocity = 0
                previous_error = 0
                previous_noise = 0
                
            # computate equilibrium point
            previous_state = actual_state.copy()
            
            actual_state = gaussian(actual_position , X , std_dev)
            actor_output = sigmoid(spreading(actor_weights, actual_state))
        
            previous_noise = actual_noise
            actual_noise = computate_noise(previous_noise, std_dev) * T
           # print "actual noise"
           # print actual_noise
            
            EP = actor_output + actual_noise
            
            if EP > 1:    
                EP = 1
                    
            if EP < 0:    
                EP = 0
                
          #  print "EP"
           # print EP
            
            
            # storage previous movement values
            if movement > 0:
                previous_position = actual_position 
                previous_velocity = actual_velocity
                previous_acceleration = actual_acceleration
            
            # compute new movement values
            actual_error = error(EP,actual_position)
            actual_acceleration = PID_controller(actual_error, previous_error) / mass
            actual_velocity = previous_velocity + actual_acceleration * delta_time
            actual_position = previous_position + actual_velocity * delta_time
            
            # graphic movement
            agent.set_data(actual_position , 0)
            plt.pause(0.0001)
            
            actual_state = gaussian(actual_position , X , std_dev)
            previous_error = actual_error
            
            # seeking for reward and weights adjustement
            if movement > 0:
                
                if np.absolute(actual_position - reward_position) < 0.05:
                    print movement
                    print "presa"
                    actual_reward = 1
                    
                else:
                    actual_reward = 0
                    
                # storage old critic output & compute new critic output 
                previous_critic_output = actual_critic_output
                actual_critic_output = sigmoid(spreading(critic_weights, actual_state))
                
                # computing surprise & weights adjustement
                surprise = TDerror(actual_reward, actual_critic_output, previous_critic_output)
                actor_weights +=  eta * surprise * previous_state * previous_noise
                critic_weights += eta * surprise * previous_state
                number_needed_steps[trial] = movement
                                   
                if actual_reward == 1:
                    break
                
    plt.figure(figsize=(30, 4), num=4, dpi=80)
    plt.title('number of movement to get reward')
    plt.xlim([0, n_trial])
    plt.ylim([0, max_trial_movements])
    plt.xticks(np.arange(0,n_trial, 1))
    plt.plot(number_needed_steps)
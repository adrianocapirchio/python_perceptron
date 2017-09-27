# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:07:56 2017

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rdm
import math_utils as utils

class wrist:
    
    mass = 1.
    
    def init(self):
        
        # 3d movements
        self.actual_3derror = np.zeros(3)
        self.previous_3derror = np.zeros(3)
        self.actual_3dposition = np.zeros(3)
        self.previous_3dposition = np.zeros(3)
        self.actual_3dvelocity = np.zeros(3)
        self.previous_3dvelocity = np.zeros(3)
        self.actual_3dacceleration = np.zeros(3)
        
        # 2d movements
        self.actual_2derror = np.zeros(2)
        self.previous_2derror = np.zeros(2)
        self.next_2dposition = np.zeros(2)
        self.actual_2dposition = np.zeros(2)
        self.previous_2dposition = np.zeros(2)
        self.actual_2dvelocity = np.zeros(2)
        self.previous_2dvelocity = np.zeros(2)
        self.actual_2dacceleration = np.zeros(2)
        
class actor_critic:
    
    # GENERAL PARAMETERS
    mov_range = 1.
    dimensions = 3
    n_interval = 10
    gaussian_number = n_interval + 1 
    interval_lenght = mov_range / n_interval
    X = np.zeros(gaussian_number)
    std_dev = 1. / ((n_interval -1) * 2)
    
    # TIME PARAMETERS
    simulation_duration = 1.
    delta_time = 0.1
    steps = simulation_duration / delta_time
    tau = 1.  
    n_trial = 4000
    max_trial_movements = 7000
    
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
    
    def init(self, n_input, dimensions, critic_n_output, gaussian_number, actor_n_output, n_trial):
    
        # weights
        self.x_actor_weights = np.zeros((n_input * dimensions))
        self.y_actor_weights = np.zeros((n_input * dimensions))
        self.z_actor_weights = np.zeros((n_input * dimensions))
        self.critic_weights = np.zeros((n_input * dimensions))
        
        self.actual_noise = np.zeros(actor_n_output)
        self.previous_noise = np.zeros(actor_n_output)
        
        self.EP3d = np.zeros(3)
        self.EP2d = np.zeros(2)
    
        # critic output parameters
        self.surprise = np.zeros(1)
        self.actual_critic_output = np.zeros(critic_n_output)
        self.previous_critic_output = np.zeros(critic_n_output)
        
        # STATE ARRAYS    
        self.actual_state = np.zeros((gaussian_number , dimensions))
        self.previous_state = np.zeros((gaussian_number , dimensions))
        
        self.needed_steps = np.zeros(n_trial)
               
    def spreading(self, w, pattern):
        return np.dot( w, pattern)

    def TDerror(self, actual_reward, actual_critic_output, previous_critic_output, discount_factor):
        x = discount_factor * actual_critic_output - previous_critic_output 
        return x + actual_reward 
        
    def act_training(self, a_eta, surprise, previous_state, previous_noise):    
        return a_eta * surprise * previous_state * previous_noise
        
    def crit_training(self, c_eta, surprise, previous_state):
        return c_eta * surprise * previous_state
    
# main    
if __name__ == "__main__":
    
    #♣ INIT 2d plotting         
    fig1   = plt.figure("Workspace",figsize=(50,50))
    ax1    = fig1.add_subplot(111)
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    
    #◘ place reward         
    reward_position = np.array([rdm.uniform(0,1),rdm.uniform(0,1)])         
    reward, = ax1.plot([reward_position[0]] , [reward_position[1]], "x") 
    
    BG = actor_critic()
    WRIST = wrist()
    
    BG.init(BG.n_input, BG.dimensions, BG.critic_n_output, BG.gaussian_number, BG.actor_n_output, BG.n_trial)
    WRIST.init()
    
    # computate gaussian's average values
    for gn in xrange(BG.gaussian_number):
        if gn == 0 :
            BG.X[0] = 0    
        else:
            BG.X[gn] = BG.X[gn-1] + BG.interval_lenght
    
    # start trials
    for trial in xrange(BG.n_trial):
        
        # temperature magnitude
        T = 1 * utils.clipped_exp(- trial * 0.2 / float(BG.n_trial))
        
        # place agent   
        agent_starting_3dposition = np.array([0.5,0.5,0.5])
        agent_starting_2dposition = np.array( [rdm.uniform(-1,1) , rdm.uniform(-1,1)] )
        
        if trial == 0:
            agent, = ax1.plot(agent_starting_2dposition[0], agent_starting_2dposition[1], 'o')
            
        if trial > 0:
            agent.remove()
            agent, = ax1.plot(agent_starting_2dposition[0], agent_starting_2dposition[1], 'o')
            plt.pause(0.001)
            
        # start movement
        for movement in xrange(BG.max_trial_movements):
            
            # init first movement's parameters
            if movement == 0:
                
                WRIST.actual_3dposition = agent_starting_3dposition.copy()
                WRIST.previous_3dposition = WRIST.actual_3dposition.copy()
                
                WRIST.actual_2dposition = agent_starting_2dposition.copy()
                WRIST.previous_2dposition = WRIST.actual_2dposition.copy()
                
            # computate actual state
            actual_x_state = utils.gaussian(WRIST.actual_3dposition[0] , BG.X , BG.std_dev)
            actual_y_state = utils.gaussian(WRIST.actual_3dposition[1] , BG.X , BG.std_dev)
            actual_z_state = utils.gaussian(WRIST.actual_3dposition[2] , BG.X , BG.std_dev)
            BG.actual_state = np.hstack([actual_x_state,actual_y_state,actual_z_state])
            
            # compute actor output
            x_actor_output = utils.sigmoid(BG.spreading(BG.x_actor_weights, BG.actual_state))
            y_actor_output = utils.sigmoid(BG.spreading(BG.y_actor_weights, BG.actual_state))
            z_actor_output = utils.sigmoid(BG.spreading(BG.z_actor_weights, BG.actual_state))
                
            # storage old noise and compute new noise
            BG.previous_noise = BG.actual_noise.copy()
            BG.actual_noise = utils.computate_noise(BG.previous_noise, BG.delta_time, BG.tau) * T 
                                                
            # compute 3D EPs
            BG.EP3d = np.array([x_actor_output, y_actor_output, z_actor_output]) + BG.actual_noise              
            BG.EP3d = utils.Cut_range(BG.EP3d, 0, 1)
            
            if movement > 0:
                WRIST.previous_3derror = WRIST.actual_3derror.copy()
                WRIST.previous_3dposition = WRIST.actual_3dposition.copy() 
                WRIST.previous_3dvelocity = WRIST.actual_3dvelocity.copy()
                WRIST.previous_3dacceleration = WRIST.actual_3dacceleration.copy()
            
            # compute 3d movement 
            WRIST.actual_3derror = utils.error(BG.EP3d,WRIST.actual_3dposition)
            WRIST.actual_3dacceleration = utils.PID_controller(WRIST.actual_3derror, WRIST.previous_3derror, BG.delta_time, BG.tau) / WRIST.mass
            WRIST.actual_3dvelocity = WRIST.previous_3dvelocity + WRIST.actual_3dacceleration * BG.delta_time
            WRIST.actual_3dposition = WRIST.previous_3dposition + WRIST.actual_3dvelocity * BG.delta_time
            WRIST.actual_3dposition = utils.Cut_range(WRIST.actual_3dposition, 0.00001, 0.99999) 
            
            # conversion to 2d movement
            BG.EP2d = utils.conversion2d(utils.change_range(BG.EP3d, 0, 1, -1, 1)) * 0.3
     
            # compute 2d final position
            WRIST.next_2dposition[0] = WRIST.actual_2dposition[0] - BG.EP2d[0]
            WRIST.next_2dposition[0] = utils.Cut_range(WRIST.next_2dposition[0], -0.99999, 0.99999)       
            WRIST.next_2dposition[1] = WRIST.actual_2dposition[1] + BG.EP2d[1]
            WRIST.next_2dposition[1] = utils.Cut_range(WRIST.next_2dposition[1], -0.99999, 0.99999) 
            
            for t in xrange(int(BG.steps)):
                WRIST.previous_2derror = WRIST.actual_2derror.copy()
                WRIST.previous_2dposition = WRIST.actual_2dposition.copy() 
                WRIST.previous_2dvelocity = WRIST.actual_2dvelocity.copy()
                WRIST.previous_2dacceleration = WRIST.actual_2dacceleration.copy()   
               
                WRIST.actual_2derror = utils.error(WRIST.next_2dposition,WRIST.actual_2dposition)
                WRIST.actual_2dacceleration = utils.PID_controller(WRIST.actual_2derror, WRIST.previous_2derror, BG.delta_time, BG.tau) / WRIST.mass
                WRIST.actual_2dvelocity = WRIST.previous_2dvelocity + WRIST.actual_2dacceleration * BG.delta_time
                WRIST.actual_2dposition = WRIST.previous_2dposition + WRIST.actual_2dvelocity * BG.delta_time
                WRIST.actual_2dposition = utils.Cut_range(WRIST.actual_2dposition, -0.99999, 0.99999)
                
                agent.set_data(WRIST.actual_2dposition[0], WRIST.actual_2dposition[1])
                plt.pause(0.0001)
                
            # storage old state 
            BG.previous_state = BG.actual_state.copy()
            
            # compute actual state
            actual_x_state = utils.gaussian(WRIST.actual_3dposition[0] , BG.X , BG.std_dev)
            actual_y_state = utils.gaussian(WRIST.actual_3dposition[1] , BG.X , BG.std_dev)
            actual_z_state = utils.gaussian(WRIST.actual_3dposition[2] , BG.X , BG.std_dev)
            BG.actual_state = np.hstack([actual_x_state,actual_y_state,actual_z_state])
            
            if movement > 0:
                
                # compute reward distance
                distance = utils.Distance(WRIST.actual_2dposition[0],reward_position[0],WRIST.actual_2dposition[1],reward_position[1])
                
                # get the reward
                if distance < 0.1:
                    print "presa"
                    print movement
                    actual_reward = 1    
                else:
                    actual_reward = 0
                    
                # storage old critic output, compute new and get surprise
                BG.previous_critic_output = BG.actual_critic_output
                BG.actual_critic_output = BG.spreading(BG.critic_weights,BG.actual_state)
                BG.surprise = BG.TDerror(actual_reward, BG.actual_critic_output, BG.previous_critic_output, BG.discount_factor)
                
                # weights adjustement
                BG.x_actor_weights +=  BG.act_training(BG.a_eta, BG.surprise, BG.previous_state, BG.previous_noise[0])   
                BG.y_actor_weights +=  BG.act_training(BG.a_eta, BG.surprise, BG.previous_state, BG.previous_noise[1])
                BG.z_actor_weights +=  BG.act_training(BG.a_eta, BG.surprise, BG.previous_state, BG.previous_noise[2])  
                BG.critic_weights +=  BG.crit_training(BG.c_eta, BG.surprise, BG.previous_state)
                
                # storage min movements to get reward
                BG.needed_steps[trial] = movement
                               
                # end trial if agent got reward                   
                if actual_reward == 1:
                    break 
    
    # PLOT TRIAL'S MOVEMENTS TO GET REWARD            
    plt.figure(figsize=(80, 4), num=4, dpi=80)
    plt.title('number of movement to get reward')
    plt.xlim([0, BG.n_trial])
    plt.ylim([0, BG.max_trial_movements])
    plt.xticks(np.arange(0,BG.n_trial, 100))
    plt.plot(BG.needed_steps)
                                   
                                
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    
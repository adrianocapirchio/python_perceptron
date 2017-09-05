# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:24:55 2017

@author: Alex
"""

# added dynamic movements in graphic
import numpy as np
import matplotlib.pyplot as plt

# init world 
possible_states = 10

starting_state = np.zeros(possible_states, dtype=np.uint8)
starting_state[int(np.random.rand(1) * 10)] = 1 
         
actual_state = np.zeros(possible_states, dtype=np.uint8)

reward_position = np.zeros(possible_states, dtype=np.uint8)
reward_position[int(np.random.rand(1) * 10)] = 1
    
# time's parameters
simulation_duration = 1.
delta_time = 0.1
steps = simulation_duration / delta_time
tau = 1.  

# init network & agent
mass = 1
n_input = int(starting_state.shape[0])
critic_n_output = 1
actor_n_output = 2
eta = 0.3
discount_factor = 0.99
max_trial_movements = 10
n_trial = 200
previous_critic_output = 0
             
critic_weights = np.zeros(n_input)
actor_weights = np.zeros([n_input,actor_n_output])

critic_output = np.zeros(critic_n_output)
actor_output = np.zeros(actor_n_output)

actions_probabilities = np.zeros(actor_n_output)

# init storage arrays
state_history = np.zeros([possible_states, max_trial_movements* n_trial])
critic_output_history = np.zeros(max_trial_movements * n_trial)             
number_needed_steps = np.zeros(n_trial)

# utilities

def derivative(x1, x2, delta_time, tau):
    return (x1 - x2) / (delta_time / tau)

def error(EP, actual_position):
    return EP - actual_position

def PID_controller(actual_error, previous_error): 
    Kp = 21
    Kd = 7
    force = Kp * (actual_error) + Kd * derivative(actual_error, previous_error, delta_time, tau)
    return force

def activation(x):
    return 1 / (1.0 + np.exp(-(2.0*x)))
        
def spreading(w , pattern):
    return np.dot(w, pattern)    
 
def softmax(x , T):
    e_x = np.exp(x / T)
    return e_x / e_x.sum()

def TDerror(actual_reward, critic_output, previous_critic_output):
    return actual_reward + discount_factor * critic_output - previous_critic_output

# main    
if __name__ == "__main__":
    
    # init plotting
    fig1   = plt.figure("Workspace",figsize=(possible_states,3))
    ax1    = fig1.add_subplot(111)
    ax1.set_xlim([0,possible_states])
    ax1.set_ylim([-1,3])
    plt.xticks(np.arange(0, possible_states, 1))
    
    agent, = ax1.plot([np.argmax(starting_state)], [0], 'o')
    reward, = ax1.plot([np.argmax(reward_position)], [0], 'x')
    
    text1 = ax1.text(8, 2, "trial = "  , style='italic', bbox={'facecolor':'green'})
    text2 = ax1.text(8, 1.5, "movement = " , style='italic', bbox={'facecolor':'red'})
    
    for trial in xrange(n_trial):
        
        text1.set_text("trial = %s" % (trial))
        
        if trial > 0:
        
            starting_state = np.zeros(possible_states, dtype=np.uint8)   
            starting_state[int(np.random.rand(1) * 10)] = 1
        
        movement = 0
        
        T = 1 * np.exp(- trial * 5.0 / float(n_trial))

        actual_state = starting_state.copy()
        actual_position = np.argmax(actual_state)
        actual_velocity = 0
        actual_acceleration = 0
        previous_error = 0
        
        # compute actor's output at movement = 0
        actor_output = activation(spreading(actor_weights.T, actual_state))
                            
        # compute possible action's probabilities                    
        actions_probabilities = softmax(actor_output , T)
                                     
        # move the agent
        i = np.argmax(actual_state)
        state_history[:,movement] = actual_state
                     
        probability = np.random.rand(1)
        
        actual_state[i] = actual_state[i] - 1
        
        if probability <= actions_probabilities[0]:
        
            i = (i+1) % possible_states
                        
        else: 
        
            i = (i-1) % possible_states
            
        actual_state[i] = 1
        EP = np.argmax(actual_state)
        actual_error = error(EP,actual_position)
        
        for t in xrange(int(steps)):
            
            previous_error = actual_error
            previous_position = actual_position 
            previous_velocity = actual_velocity
            previous_acceleration = actual_acceleration
            
            actual_error = error(EP,actual_position)
            actual_acceleration = PID_controller(actual_error, previous_error) / mass
            actual_velocity = previous_velocity + actual_acceleration * delta_time
            actual_position = previous_position + actual_velocity * delta_time
            agent.set_data(actual_position,0)
            plt.pause(0.0001)
        
        for movement in xrange(max_trial_movements):
            
            text2.set_text("movement = %s" % (movement))
            
            # compute critic's output
            critic_output = spreading(critic_weights, actual_state)
            critic_output_history[movement] = critic_output
            
            # compute actor's output 
            actor_output = activation(spreading(actor_weights.T, actual_state))
                        
            # compute possible action's probabilities                    
            actions_probabilities = softmax(actor_output , T)
                                     
            # move the agent
            i = np.argmax(actual_state)
            state_history[:,movement] = actual_state
             
            probability = np.random.rand(1)
            
            actual_state[i] = actual_state[i] - 1 
            
            if probability <= actions_probabilities[0]:
        
                i = (i+1) % possible_states
            
            else: 
        
                i = (i-1) % possible_states 
                            
            actual_state[i] = 1 
            EP = np.argmax(actual_state)
            actual_error = error(EP,actual_position)
            
            for t in xrange(int(steps)):
                
                previous_error = actual_error
                previous_position = actual_position 
                previous_velocity = actual_velocity
                previous_acceleration = actual_acceleration
                actual_error = error(EP,actual_position)
                actual_acceleration = PID_controller(actual_error, previous_error) / mass
                actual_velocity = previous_velocity + actual_acceleration * delta_time
                actual_position = previous_position + actual_velocity * delta_time
                agent.set_data(actual_position,0)
                plt.pause(0.0001)
                
            # seek for reward
            if np.argmax(actual_state) == np.argmax(reward_position):
                
                actual_reward = 1
            
            else:
                    
                actual_reward = 0
                
            # compute critic's output
            critic_output = spreading(critic_weights, actual_state)
            critic_output_history[movement] = critic_output
            
            previous_state = state_history[:,movement]
            previous_critic_output = critic_output_history[movement-1]
            
            critic_weights += eta * TDerror(actual_reward, critic_output, previous_critic_output) * previous_state 
                
            if probability <= actions_probabilities[0]:
                
                actor_weights[:,0] +=  eta * TDerror(actual_reward, critic_output, previous_critic_output) * previous_state 
                
            else:
                    
                actor_weights[:,1] +=  eta * TDerror(actual_reward, critic_output, previous_critic_output)* previous_state 
                
                             
                    
            if actual_reward == 1: 
                
                number_needed_steps[trial]  = movement
                
                break
            
    plt.figure(figsize=(30, 4), num=4, dpi=80)
    plt.title('number of steps to get reward')
    plt.xlim([0,n_trial])
    plt.xticks(np.arange(0,n_trial, 5))
    plt.yticks(np.arange(0, max_trial_movements, 1))
    plt.plot(number_needed_steps)
                              
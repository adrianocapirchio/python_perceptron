# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:23:10 2017

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt

# init world 
possible_states = 10
starting_state = np.zeros(possible_states, dtype=np.uint8)             
actual_state = np.zeros(possible_states, dtype=np.uint8)

reward_position = np.zeros(possible_states, dtype=np.uint8)
reward_position[possible_states / 2] = 1



# init network & agent
n_input = int(starting_state.shape[0])
critic_n_output = 1
actor_n_output = 2
eta = 0.5
discount_factor = 0.99
steps = 10
n_trial = 100
previous_critic_output = 0
             
critic_weights = np.zeros(n_input)
actor_weights = np.zeros([n_input,actor_n_output])

critic_output = np.zeros(critic_n_output)
actor_output = np.zeros(actor_n_output)

actions_probabilities = np.zeros(actor_n_output)

agent_position = np.zeros(steps * n_trial)



# init storage arrays
state_history = np.zeros([possible_states, steps])
matched_reward_history = np.zeros(steps * n_trial)             

critic_weights_history = np.zeros([n_input, steps * n_trial])
actor_weights_history = np.zeros([n_input, steps * n_trial])
  
critic_output_history = np.zeros([steps])
actor_output_history = np.zeros([actor_n_output, steps])

actions_probabilities_history = np.zeros([actor_n_output, steps])
probability_history = np.zeros(steps)

TDerror_history = np.zeros(steps)

number_needed_steps = np.zeros(steps * n_trial)



# utilities
def activation(x):
    
    y = 1 / (1.0 + np.exp(-(2.0*x)))
    return y 
    
  
    
def spreading(w , pattern):
    
    y = np.dot(w, pattern)    
    return y
 
    

def softmax(x , T):
    
    e_x = np.exp(x / T)
    return e_x / e_x.sum()



def TDerror(actual_reward, critic_output, previous_critic_output):
    
    TDerror = actual_reward + discount_factor * critic_output - previous_critic_output
    return TDerror



# main    
if __name__ == "__main__":
       
    for trial in xrange(n_trial):
        
        T = 1 * np.exp(- trial * 3.0 / float(n_trial))
        
        t = 0
        
        starting_state = np.zeros(possible_states, dtype=np.uint8)
        starting_state[trial % possible_states] = 1

        actual_state = starting_state.copy()
        
        state_history = np.zeros([possible_states, steps])
        
        # compute actor's output at t = 0
        actor_output = activation(spreading(actor_weights.T, actual_state)) 
        actor_output_history[:,t] = actor_output
                            
        # compute possible action's probabilities                    
        actions_probabilities = softmax(actor_output , T)
        actions_probabilities_history[:,t] = actions_probabilities
                                     
        # move the agent
        i = np.argmax(actual_state)
        state_history[:,t] = actual_state
                     
        probability = np.random.rand(1)
        probability_history[t] = probability
    
        if probability <= actions_probabilities[0]:
        
            actual_state[i] = actual_state[i] - 1
            i = (i+1) % possible_states
            actual_state[i] = 1
            
        else: 
        
            actual_state[i] = actual_state[i] - 1
            i = (i-1) % possible_states
            actual_state[i] = 1
                        
        t = 1
        
        for t in xrange(steps):
            
            # storage critic weights
            critic_weights_history[:,t + trial * steps] = critic_weights
            
            # compute critic's output
            critic_output = spreading(critic_weights, actual_state)
            critic_output_history[t] = critic_output
            
            # compute actor's output 
            actor_output = activation(spreading(actor_weights.T, actual_state)) 
            actor_output_history[:,t] = actor_output
                        
            # compute possible action's probabilities                    
            actions_probabilities = softmax(actor_output , T)
            actions_probabilities_history[:,t] = actions_probabilities
                                     
            # move the agent
            i = np.argmax(actual_state)
            state_history[:,t] = actual_state
             
            probability = np.random.rand(1)
            probability_history[t] = probability
                                
            if probability <= actions_probabilities[0]:
        
                actual_state[i] = actual_state[i] - 1
                i = (i+1) % possible_states
                actual_state[i] = 1
            
            else: 
        
                actual_state[i] = actual_state[i] - 1
                i = (i-1) % possible_states
                actual_state[i] = 1 
                 
            # seek for reward
            if np.argmax(actual_state) == np.argmax(reward_position):
                
                actual_reward = 1
            
            else:
                    
                actual_reward = 0
                
            # compute critic's output
            critic_output = spreading(critic_weights, actual_state)
            critic_output_history[t] = critic_output
            
            previous_state = state_history[:,t] 
            previous_critic_output = critic_output_history[t-1]
                    
            TDerror_history[t] = TDerror(actual_reward, critic_output, previous_critic_output)
            
            critic_weights += eta * TDerror(actual_reward, critic_output, previous_critic_output) * previous_state 
                
            if probability <= actions_probabilities[0]:
                
                actor_weights[:,0] +=  eta * TDerror(actual_reward, critic_output, previous_critic_output) * previous_state 
                
            else:
                    
                actor_weights[:,1] +=  eta * TDerror(actual_reward, critic_output, previous_critic_output)* previous_state 
                
            if actual_reward == 1: 
                
                number_needed_steps[t + trial * steps]  = t
                
                if trial % 10 == 0:
                      
                    plt.figure(figsize=(12, 4), num=trial, dpi=80)
                    plt.yticks(np.arange(0, possible_states, 1))
                    plt.xticks(np.arange(0, t+1, 1))
                    plt.title('agent_position')
                    plt.imshow(state_history, aspect = "auto", interpolation = "none")
            
                    plt.figure(figsize=(12, 4), num=trial+1, dpi=80)
                    plt.xlim([0,t])
                    plt.xticks(np.arange(0, t+1, 1))
                    plt.title('actions probabilities')
                    plt.plot(actions_probabilities_history[0,:], color = 'blue', label = 'unit 1 output --> move right')
                    plt.plot(actions_probabilities_history[1,:], color = 'green', label = 'unit 2 output --> move left')
                    plt.legend(loc='upper right')
            
                    plt.figure(figsize=(12, 4), num=trial+2, dpi=80)
                    plt.title('critic weights adjustement')
                    plt.imshow(critic_weights_history, aspect = "auto", interpolation = "none")
                    
                break
                
    plt.figure(figsize=(12, 4), num=trial+3, dpi=80)
    plt.title('number of steps to get reward')
    plt.xlim([0,steps * n_trial])
    plt.xticks(np.arange(0, steps * n_trial, 1))
    plt.plot(number_needed_steps)
        
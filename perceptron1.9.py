# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 15:45:42 2017

@author: Alex
"""

# added main
# plotting function
# 3 input units 
# clean code

import numpy as np
import matplotlib.pyplot as plt


#costanti
n = 3
eta = 0.5
trial_time = 100
n_trial = 20
total_trial_time = trial_time * n_trial

#variabili
errori_test = np.zeros(total_trial_time)

# input/desired output init
def init(n, total_trial_time, n_trial):
    
    input1 = np.zeros(total_trial_time)
    input2 = np.zeros(total_trial_time)
    input3 = np.zeros(total_trial_time)
    
    bias = np.ones([1])
    x = np.hstack([np.zeros(n),bias])
    pattern_history = np.zeros([n+1,total_trial_time])
    
    output = np.zeros([total_trial_time])
    desired_output = np.zeros([total_trial_time])
    
    error_history = np.zeros([total_trial_time])
    w = np.zeros(n+1)
    w_history = np.zeros([n+1,total_trial_time])
    
    test_errors_history = np.zeros([trial_time])
    
    for trial in xrange(n_trial):
    
        for t in xrange(trial_time):
    
            p1 = 1. / (3. * (t  + 1)) 
            input1[t + trial * trial_time] = p1
                  
            p2 = 1. / (0.5 * (t + 1))
            input2[t + trial * trial_time] = p2
                  
            p3 = 1 + 0.002 * t
            input3[t + trial * trial_time] = p3
                  
            x[0] = p1
            x[1] = p2
            x[2] = p3
            pattern_history[:, t + trial * trial_time] = x
                           
            ok = 1. / (t + 1.) 
            desired_output[t + trial * trial_time] = ok
    
    return input1, input2, input3, \
           output, pattern_history,\
           desired_output, error_history,\
           w_history, w, test_errors_history

# funzione di attivazione
def activation(x):
    
    y = 1 / (1.0 + np.exp(-(2.0*x)))
    
    return y

def derivative(x):
    
    y = x*(1-x)
    
    return y

# spreading della rete
def spreading(w,pattern):
    
    v = np.dot(pattern,w)
    y = activation(v)
    
    return y
        
# network training/weigth adjustement
def training(w, desired_output, real_output, pattern):
    
    error = desired_output - real_output
    squared_error = 0.5 * error ** 2
    w += eta * error * derivative(real_output) * pattern 
    
    return squared_error, w
    
def plotting(indices):    
    # plot di input1, input2, output desiderato, errori durante il training, varizione dei pesi    
    
    if 1 in indices:
        plt.figure(figsize=(15, 6), num=1, dpi=80)
        plt.ylim([0,1])
        plt.title('input 1')
        plt.plot(input1, color='blue', label='input1')   
    
    if 2 in indices:
        plt.figure(figsize=(15, 6), num=2, dpi=80)
        plt.ylim([0,1])
        plt.title('input 2')
        plt.plot(input2, color='green', label='input2')
    
    if 3 in indices:
        plt.figure(figsize=(15, 6), num=3, dpi=80)
        plt.ylim([0,2])
        plt.title('input 3')
        plt.plot(input3, color='red', label='input3')
    
    if 4 in indices:
        plt.figure(figsize=(15, 6), num=4, dpi=80)
        plt.ylim([0,2])
        plt.title('input1-2-3')
        plt.plot(input1, color='blue', label='input1')   
        plt.plot(input2, color='green', label='input2')
        plt.plot(input3, color='red', label='input3')
        plt.legend(loc='upper right')
    
    if 5 in indices:
        plt.figure(figsize=(15, 6), num=5, dpi=80)
        plt.ylim([0,1])
        plt.title('desired ouput')
        plt.plot(desired_output,color='red', label='desired output')
    
    if 6 in indices:
        plt.figure(figsize=(15, 6), num=6, dpi=80)
        plt.title('desired output vs real output')
        plt.plot(desired_output,color='red', label='desired output')
        plt.plot(output,color='blue', label='real output')
        plt.legend(loc='upper right')
    
    if 7 in indices:
       plt.figure(figsize=(15, 6), num=7, dpi=80)
       plt.ylim([0,1])
       plt.title('training errors')
       plt.plot(error_history)
    
    if 8 in indices:
        plt.figure(figsize=(15, 6), num=8, dpi=80)
        plt.title('weight adjustement')
        plt.plot(w_history[0,:], color='blue', label='w1')
        plt.plot(w_history[1,:], color='green', label='w2')
        plt.plot(w_history[2,:], color='brown', label='w3')
        plt.plot(w_history[3,:], color='red', label='w4-bias')
        plt.legend(loc='upper left')
    
    if 9 in indices:
        plt.figure(figsize=(15, 6), num=9, dpi=80) 
        plt.title('test errors')
        plt.plot(test_errors_history)
      
def test(w, desired_output):
     
    p = np.random.rand(n)
    pattern = np.hstack([p,1])
    current_output = spreading(w, pattern)
    current_error = desired_output - current_output
    squared_error = 0.5 * current_error ** 2
    
    return squared_error
     
#main
if __name__ == "__main__":
    
    input1, input2, \
    input3, output, \
    pattern_history,\
    desired_output, \
    error_history,  \
    w_history, w, \
    test_errors_history = init(n, total_trial_time, n_trial)
    
     # ciclo di iterazione dei trial
    for trial in xrange(n_trial):
       # ciclo di iterazione in ogni trial
        for t in xrange(trial_time):
            w_history[:,t+trial*trial_time] = w 
            pattern = pattern_history[:,t+trial*trial_time]
            current_desired_output = desired_output[t+trial*trial_time]
            current_output = spreading(w, pattern)
            output[t+trial*trial_time] = current_output
            current_error, w = training(w, current_desired_output, current_output, pattern)
            error_history[t+trial*trial_time] = current_error
    
    for t in xrange(trial_time):
        current_desired_output = desired_output[t]
        current_error = test(w, current_desired_output)
        test_errors_history[t] = current_error    
    
    indices = [1,2,3,4,5,6,7,8,9]
    plotting(indices)                 
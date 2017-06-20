# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:53:04 2017

@author: Alex
"""

# 3 output
# added comments

import numpy as np
import matplotlib.pyplot as plt

#costanti
n_input = 3
n_output = 3
eta = 0.5
trial_time = 100
n_trial = 20
total_trial_time = trial_time * n_trial

def init(n_input, n_output, total_trial_time, n_trial):
    
    # init input history on each input's unit
    input1 = np.zeros(total_trial_time)
    input2 = np.zeros(total_trial_time)
    input3 = np.zeros(total_trial_time)
    
    # init bias
    bias = np.ones([1])
    
    # init single pattern
    x = np.hstack([np.zeros(n_input),bias])
    
    # init pattern history
    pattern_history = np.zeros([n_input+1,total_trial_time])
    
    # init real output for each output's unit
    output1 = np.zeros([total_trial_time])
    output2 = np.zeros([total_trial_time])
    output3 = np.zeros([total_trial_time])
    
    # init desired output for each output's unit
    desired_output1 = np.zeros([total_trial_time])
    desired_output2 = np.zeros([total_trial_time])
    desired_output3 = np.zeros([total_trial_time])
    
    # init error variation during training
    unit1_training_error_history = np.zeros([total_trial_time])
    unit2_training_error_history = np.zeros([total_trial_time])
    unit3_training_error_history = np.zeros([total_trial_time])
    
    # init weights matrix
    w = np.zeros([n_input+1,n_output])
    
    # init output units weight connection history
    w1_history = np.zeros([n_input+1,total_trial_time])
    w2_history = np.zeros([n_input+1,total_trial_time])
    w3_history = np.zeros([n_input+1,total_trial_time])
    
    # init test errors history
    unit1_test_errors_history = np.zeros([trial_time])
    unit2_test_errors_history = np.zeros([trial_time])
    unit3_test_errors_history = np.zeros([trial_time])
    
    # generate dataset
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
                           
            ok1 = 1. / (t + 1.) 
            desired_output1[t + trial * trial_time] = ok1
                          
            ok2 = 1. / ((t ** 2) + 1)
            desired_output2[t + trial * trial_time] = ok2
            
            ok3 = 1. / ((3 * t) + 1) 
            desired_output3[t + trial * trial_time] = ok3
    
    return input1, input2, input3,                            \
           pattern_history,                                   \
           output1, output2, output3,                         \
           desired_output1, desired_output2, desired_output3, \
           unit1_training_error_history,                      \
           unit2_training_error_history,                      \
           unit3_training_error_history,                      \
           unit1_test_errors_history,                         \
           unit2_test_errors_history,                         \
           unit3_test_errors_history,                         \
           w, w1_history, w2_history, w3_history            
            

# activation function
def activation(x):
    
    y = 1 / (1.0 + np.exp(-(2.0*x)))
    
    return y

# derivative sigmoid function
def derivative(x):
    
    y = x*(1-x)
    
    return y

# network spreading
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

# plotting function    
def plotting(indices):    
    # plot di input1, input2, output desiderato, errori durante il training, varizione dei pesi    
    
    if 1 in indices:
        plt.figure(figsize=(15, 4), num=1, dpi=80)
        plt.ylim([0,2])
        plt.title('input 1')
        plt.plot(input1, color='blue', label='input1')   
    
    if 2 in indices:
        plt.figure(figsize=(15, 4), num=2, dpi=80)
        plt.ylim([0,2])
        plt.title('input 2')
        plt.plot(input2, color='green', label='input2')
    
    if 3 in indices:
        plt.figure(figsize=(15, 4), num=3, dpi=80)
        plt.ylim([0,2])
        plt.title('input 3')
        plt.plot(input3, color='red', label='input3')
    
    if 4 in indices:
        plt.figure(figsize=(15, 4), num=4, dpi=80)
        plt.ylim([0,2])
        plt.title('input1-2-3')
        plt.plot(input1, color='blue', label='input1')   
        plt.plot(input2, color='green', label='input2')
        plt.plot(input3, color='red', label='input3')
        plt.legend(loc='upper right')
    
    if 5 in indices:
        plt.figure(figsize=(15, 4), num=5, dpi=80)
        plt.ylim([0,2])
        plt.title('desired output 1')
        plt.plot(desired_output1,color='red', label='desired output 1')
        
    if 6 in indices:
        plt.figure(figsize=(15, 4), num=6, dpi=80)
        plt.ylim([0,2])
        plt.title('desired output 2')
        plt.plot(desired_output2,color='red', label='desired output 2')
        
    if 7 in indices:
        plt.figure(figsize=(15, 4), num=7, dpi=80)
        plt.ylim([0,2])
        plt.title('desired output 3')
        plt.plot(desired_output3,color='red', label='desired output 3')
    
    if 8 in indices:
        plt.figure(figsize=(15, 4), num=8, dpi=80)
        plt.title('desired output 1 vs real output 1')
        plt.plot(desired_output1,color='red', label='desired output 1')
        plt.plot(output1,color='blue', label='real output 1')
        plt.legend(loc='upper right')
        
    if 9 in indices:
        plt.figure(figsize=(15, 4), num=9, dpi=80)
        plt.title('desired output 2 vs real output 2')
        plt.plot(desired_output2,color='red', label='desired output 2')
        plt.plot(output2,color='blue', label='real output 2')
        plt.legend(loc='upper right')
        
    if 10 in indices:
        plt.figure(figsize=(15, 4), num=10, dpi=80)
        plt.title('desired output 3 vs real output 3')
        plt.plot(desired_output3,color='red', label='desired output 3')
        plt.plot(output3,color='blue', label='real output 3')
        plt.legend(loc='upper right')
    
    if 11 in indices:
        plt.figure(figsize=(15, 4), num=11, dpi=80)
        plt.ylim([0,1])
        plt.title('unit 1 training errors')
        plt.plot(unit1_training_error_history)
       
    if 12 in indices:
        plt.figure(figsize=(15, 4), num=12, dpi=80)
        plt.ylim([0,1])
        plt.title('unit 2 training errors')
        plt.plot(unit2_training_error_history)
       
    if 13 in indices:
        plt.figure(figsize=(15, 4), num=13, dpi=80)
        plt.ylim([0,1])
        plt.title('unit 3 training errors')
        plt.plot(unit3_training_error_history)
    
    if 14 in indices:
        plt.figure(figsize=(15, 4), num=14, dpi=80)
        plt.ylim([-3,3])
        plt.title("first output's unit weight adjustement")
        plt.plot(w1_history[0,:], color='blue', label='w1')
        plt.plot(w1_history[1,:], color='green', label='w2')
        plt.plot(w1_history[2,:], color='brown', label='w3')
        plt.plot(w1_history[3,:], color='red', label='w4-bias')
        plt.legend(loc='upper left')
        
    if 15 in indices:
        plt.figure(figsize=(15, 4), num=15, dpi=80)
        plt.ylim([-3,3])
        plt.title("second output's unit weight adjustement")
        plt.plot(w2_history[0,:], color='blue', label='w1')
        plt.plot(w2_history[1,:], color='green', label='w2')
        plt.plot(w2_history[2,:], color='brown', label='w3')
        plt.plot(w2_history[3,:], color='red', label='w4-bias')
        plt.legend(loc='upper left')
        
    if 16 in indices:
        plt.figure(figsize=(15, 4), num=16, dpi=80)
        plt.ylim([-3,3])
        plt.title("third output's unit weight adjustement")
        plt.plot(w3_history[0,:], color='blue', label='w1')
        plt.plot(w3_history[1,:], color='green', label='w2')
        plt.plot(w3_history[2,:], color='brown', label='w3')
        plt.plot(w3_history[3,:], color='red', label='w4-bias')
        plt.legend(loc='upper left')
    
    if 17 in indices:
        plt.figure(figsize=(15, 4), num=17, dpi=80) 
        plt.title('unit1 test errors')
        plt.plot(unit1_test_errors_history)
        
    if 18 in indices:
        plt.figure(figsize=(15, 4), num=18, dpi=80) 
        plt.title('unit 2 test errors')
        plt.plot(unit2_test_errors_history)

    if 19 in indices:
        plt.figure(figsize=(15, 4), num=19, dpi=80) 
        plt.title('unit 3 test errors')
        plt.plot(unit3_test_errors_history)


# testing function      
def test(w, desired_output):
     
    p = np.random.rand(n_input)
    pattern = np.hstack([p,1])
    current_output = spreading(w, pattern)
    current_error = desired_output - current_output
    squared_error = 0.5 * current_error ** 2
    
    return squared_error
     
# main
if __name__ == "__main__":
    
    input1, input2, input3,                            \
    pattern_history,                                   \
    output1, output2, output3,                         \
    desired_output1, desired_output2, desired_output3, \
    unit1_training_error_history,                      \
    unit2_training_error_history,                      \
    unit3_training_error_history,                      \
    unit1_test_errors_history,                         \
    unit2_test_errors_history,                         \
    unit3_test_errors_history,                         \
    w, w1_history, w2_history, w3_history = init(n_input, n_output, total_trial_time, n_trial)
    
    # learning from dataset
    for trial in xrange(n_trial):
       
        for t in xrange(trial_time):
            
            # storage current weight
            w1_history[:,t+trial*trial_time] = w[:,0]
            w2_history[:,t+trial*trial_time] = w[:,1]
            w3_history[:,t+trial*trial_time] = w[:,2]
            
            # load current pattern
            pattern = pattern_history[:,t+trial*trial_time]
            
            # load current desired output
            current_desired_output1 = desired_output1[t+trial*trial_time]
            current_desired_output2 = desired_output2[t+trial*trial_time]
            current_desired_output3 = desired_output3[t+trial*trial_time]
            
            # compute actual current output
            current_output1 = spreading(w[:,0], pattern)
            current_output2 = spreading(w[:,1], pattern)
            current_output3 = spreading(w[:,2], pattern)
            
            # storage actual current output
            output1[t+trial*trial_time] = current_output1
            output2[t+trial*trial_time] = current_output2
            output3[t+trial*trial_time] = current_output3 
            
            # compute current error and weight adjustement
            current_error1, w[:,0] = training(w[:,0], current_desired_output1, current_output1, pattern)
            current_error2, w[:,1] = training(w[:,1], current_desired_output2, current_output2, pattern)
            current_error3, w[:,2] = training(w[:,2], current_desired_output3, current_output3, pattern)
            
            # storage current error
            unit1_training_error_history[t+trial*trial_time] = current_error1
            unit2_training_error_history[t+trial*trial_time] = current_error2
            unit3_training_error_history[t+trial*trial_time] = current_error3
    
    # testing with random patterns
    for t in xrange(trial_time):
        
        # load current desider output
        current_desired_output1 = desired_output1[t]
        current_desired_output2 = desired_output2[t]
        current_desired_output3 = desired_output3[t]
        
        # compute current error
        current_error1 = test(w[:,0], current_desired_output1)
        current_error2 = test(w[:,1], current_desired_output2)
        current_error3 = test(w[:,2], current_desired_output3)
        
        # storage current error
        unit1_test_errors_history[t] = current_error1
        unit2_test_errors_history[t] = current_error2
        unit3_test_errors_history[t] = current_error3
    
    # plotting
    indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    plotting(indices)                 
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:03:34 2017

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt

#costanti
n = 2
eta = 0.1
trial_time = 100
n_trial = 4
total_trial_time = trial_time * n_trial

#variabili
input1 = np.zeros(total_trial_time)
input2 = np.zeros(total_trial_time)
bias = np.ones([1])
x = np.hstack([np.zeros(2),bias])
w = np.zeros(n+1)
dw = np.zeros([n+1,total_trial_time])
desired_output = np.zeros(total_trial_time)
real_output = np.zeros(total_trial_time)
errori = np.zeros(total_trial_time)
errori_test = np.zeros(total_trial_time)

# funzione di attivazione
def activation(v):
    y = 1 / (1.0 + np.e**(-(2.0*v)))
    return y

# spreading della rete
def training(w):
    
    # ciclo di iterazione dei trial
    for trial in xrange(n_trial):
        
        # ciclo di iterazione in ogni trial
        for t in xrange(trial_time):
            p1 = 1. / ((3. * t)+1) 
            input1[t+trial*trial_time] = p1
            p2 = 1. / ((0.1 * t) + 1)
            input2[t+trial*trial_time] = p2
            x[0] = p1
            x[1] = p2
            ok =  1. / (t + 1.)
            desired_output[t+trial*trial_time] = ok
            v = np.dot(x,w)
            y = activation(v)
            real_output[t+trial*trial_time] = y
            error = ok - y          
            errori[t+trial*trial_time] = 0.5 * (desired_output[t+trial*trial_time] - real_output[t+trial*trial_time]) ** 2
            squared_error = 0.5 * error ** 2
            errori[t+trial*trial_time] = error
            dw[:,t+trial*trial_time] = w
            w += eta * error * x
    
    
    # plot di input1, input2, output desiderato, errori durante il training, varizione dei pesi    
    plt.figure(figsize=(8, 6), num=1, dpi=80)
    ylim([0,1])
    plt.title('input 1')
    plt.plot(input1, color='blue', label='input1')   
    
    plt.figure(figsize=(8, 6), num=2, dpi=80)
    ylim([0,1])
    plt.title('input 2')
    plt.plot(input2, color='green', label='input2')
    
    plt.figure(figsize=(8, 6), num=3, dpi=80)
    ylim([0,1])
    plt.title('desired ouput')
    plt.plot(desired_output,color='red', label='desired_output')
    
    plt.figure(figsize=(8, 6), num=4, dpi=80)
    plt.title('training phase errors')
    plt.plot(errori)
    
    plt.figure(figsize=(8, 6), num=5, dpi=80)
    plt.title('weight training')
    plt.plot(dw[0,:], color='blue', label='w1')
    plt.plot(dw[1,:], color='red', label='w2')
    plt.plot(dw[2,:], color='green', label='w3-bias')
    plt.legend(loc='bottom left')
    return w
      
def test(w):
    
    for trial in xrange(n_trial):
    
        for t in xrange(trial_time):
            p = np.random.rand(n)
            x = np.hstack([p,1])
            v = np.dot(x,w)
            y = activation(v)
            error = desired_output[t+trial*trial_time] - y
            errori_test[t+trial*trial_time] = 0.5 * error ** 2
    
    plt.figure(figsize=(8, 6), num=6, dpi=80) 
    plt.title('test phase errors')
    plt.plot(errori_test)

#main
training(w)     
test(w)
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 15:16:48 2017

@author: Alex
"""

# aggiunti cicli di apprendimento
# migliorati grafici

import numpy as np
import random as stdrandom
import matplotlib.pyplot as plt

#costanti
n = 2
eta = 0.001
trial_time = 10000
n_trial = 4
total_trial_time = trial_time * n_trial
test_time = 1000

dataset = [ (np.array([0,0,1]), 0),
            (np.array([0,1,1]), 1), 
            (np.array([1,0,1]), 1), 
            (np.array([1,1,1]), 1), ]

#variabili
w = np.zeros(n+1)
dw = np.zeros([n+1,total_trial_time])
errori = np.zeros(total_trial_time)
errori_test = np.zeros(test_time)

def activation(pot):
    y = 1 / (1.0 + np.e**(-(2.0*pot)))
    return y

def training(w):
    
    for trial in xrange(n_trial):       
        
        for t in xrange(trial_time):
            x, desired = stdrandom.choice(dataset)
            pot = np.dot(x,w) 
            output = activation(pot)
            error = desired - output
            errori[t+trial*trial_time] = 0.5 * error ** 2
            dw[:,t+trial*trial_time] = w
            w += eta * error * x
            
    plt.figure(figsize=(8, 6), num=1, dpi=80)
    plt.title('training phase errors')
    plt.plot(errori)
    
    plt.figure(figsize=(8, 6), num=2, dpi=80)
    plt.title('weight training')
    plt.plot(dw[0,:], color='blue', label='w1')
    plt.plot(dw[1,:], color='red', label='w2')
    plt.plot(dw[2,:], color='green', label='w3-bias')
    plt.legend(loc='upper left')
    
    
    
def test(w):
    
    for t in xrange(test_time):
        p = np.random.rand(n)
        x = np.hstack([p,1])
        pot = np.dot(x,w)
       
        if p[0] < 0.2:
            if p[1] < 0.2:
                desired = 0
            else:
                desired = 1
        else:
            desired = 1
       
        output = activation(pot)
        error = desired - output
        errori_test[t] = 0.5 * error ** 2
    
    
    plt.figure(figsize=(8, 6), num=3, dpi=80) 
    plt.title('test phase errors')
    plt.plot(errori_test.T)

#main
training(w)     
test(w)
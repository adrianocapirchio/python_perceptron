# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 13:19:25 2017

@author: Alex
"""

# inizializzati array

import numpy as np
import random as stdrandom
from pylab import *

#costanti
n = 2
eta = 0.1
k = 1000
dataset = [ (np.array([0,0,1]), 0),
            (np.array([0,1,1]), 1), 
            (np.array([1,0,1]), 1), 
            (np.array([1,1,1]), 1), ]

#variabili
w = np.zeros(n+1)
dw = np.zeros([n+1,time])
errori = np.zeros(time)
errori_test = np.zeros(time)

def activation(pot):
    y = 1 / (1.0 + math.e**(-(2.0*pot)))
    return y

def training(w):
    
    for t in xrange(k):
        x, desired = stdrandom.choice(dataset)
        pot = np.dot(x,w) 
        output = activation(pot)
        error = desired - output
        errori[t] = 0.5 * error ** 2
        dw[:,t] = w
        w += eta * error * x
    
    figure()
    plot(errori)
    figure()
    plot(dw.T)
    return w
    
def test(w):
    
    for t in xrange(time):
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
    
    ylim([-1,1])
    plot(errori_test.T)

#main
training(w)
figure()     
test(w)

                
            
        
        
    

    
       
    
    
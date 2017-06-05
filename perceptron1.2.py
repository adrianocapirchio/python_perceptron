# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:41:01 2017

@author: Alex
"""

import random as stdrandom
from pylab import *

def f_attivazione(v):
    den = 1.0 + math.e**(-(3.0*v))
    y = 1.0 / den
    return y

def training(w):   
    
    esempi = [ (array([0,0,1]), 0),
               (array([0,1,1]), 1), 
               (array([1,0,1]), 1), 
               (array([1,1,1]), 1), ]
    
    for i in xrange(n):
        x, attesi = stdrandom.choice(esempi) 
        v = dot(x,w) 
        errore = attesi - f_attivazione(v)
        errori.append(errore)
        pesi.append(w)
        w += t_apprendimento * errore * x   
   
    ylim([-1,1])
    plot(errori)
    plot(pesi)
    return w,x


def test(w):
    
    prova = [ (array([0.1,0.1,1]), 0),
              (array([0.1,0.9,1]), 1),
              (array([0.9,0.1,1]), 1),
              (array([0.9,0.9,1]), 1), ]
    
    for x, _ in prova: 
        x,attesi = stdrandom.choice(prova)
        v = dot(x,w) 
        errore = attesi - f_attivazione(v)
        errori_test.append(errore)
    
    ylim([-1,1])
    plot(errori_test)
    
if __name__ == "__main__":
    
    w = rand(3,1) 
    errori = []
    errori_test = []
    pesi = [] 
    t_apprendimento = 0.1
    n = 10000

    figure()
    training(w)
    figure()     
    test(w)

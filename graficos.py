# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 14:53:13 2021

@author: cleiton
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp

d = np.arange(-1,-51,-1)
def p_accept(delta,T):
    return min(1,exp(delta/T))

T = [500,100,50,20,10,5,3,2,1]


plt.figure()
plt.title('Prob. de aceite x Temperatura')
for t in T:
    plt.plot([p_accept(delta,t) for delta in d], label=t)
plt.xlabel(r'$f(s)-f(s_0)$')
plt.legend()
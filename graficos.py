# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 14:53:13 2021

@author: cleiton
"""

import numpy as np
import matplotlib.pyplot as plt
from math import exp, log

d = np.arange(-1,-51.1,-0.1)
def p_accept(delta,T):
    return min(1,exp(delta/T))

T = [10,5,3,2,1]

def log_ann(t,a,b):
    return a/(log(t+b))

plt.figure()
plt.title('Probabilidades de aceite')
for t in T:
    plt.plot(-d,[p_accept(delta,t) for delta in d], label=t)
plt.xlabel(r'$-(f(s)-f(s_0))$')
plt.ylim(0,1)
plt.legend()


plt.figure()
a=1
b=0.1
x = np.arange(1,1001)
y1=[log_ann(t,a,b) for t in x]

a=2
y2=[log_ann(t,a,b) for t in x]

a=4
b=4
y3=[log_ann(t,a,b) for t in x]

plt.plot(x,y1,x,y2,x,y3)

plt.figure()
plt.plot(x,y3)
print(y3[0])
print(y3[-1])
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 14:53:13 2021

@author: cleiton
"""

import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, log2, log10
plt.rcParams.update({'font.size': 8})

d = np.arange(-1,-10.1,-0.1)
def p_accept(delta,T):
    return min(1,exp(delta/(T)))

T = [20, 10,5,3,2,1,0.5]

def linear_ann(t,T0,beta):
    return T0-beta*t

def exp_ann(t,T0,beta):
    return T0*beta**t

def log_ann(t,a,b):
    return a/(log2(t+b))

def log_inv_ann(t,T0):
    return (Ti/log(N))*log(N-1*t)


# Probabilidades de aceite x delta f(s) x temperatura
plt.figure(figsize=(4,3))
plt.title('Prob. de Aceite vs. Pontuação e Temperatura')
for t in T:
    plt.plot(d,[p_accept(delta,t) for delta in d], label=t)
plt.axvline(-1.98,c='r',linestyle='--', label=r'$\overline{\Delta f(s)}$')
plt.xlabel(r'Variação da Pontuação ($f(s)-f(s_0)$)')
plt.ylabel(r'$\mathbb{P}[Aceite]$')
plt.ylim(0,1)
plt.legend()
plt.tight_layout()

N = 100000
Ti=5
Tf=0.001

print([Ti*beta**N for beta in [0.9989, 0.9990, 0.9991]])


beta = (Tf/Ti)**(1/N)
print(beta)

# Curvas de decaimento de temperatura

# Linear
y1 = np.linspace(Ti,Tf,num=N-1)

# Exp
x = np.arange(1,N)
y2=[exp_ann(t,Ti,beta) for t in x]

# Log
a=5
b=1
y3 = [log_ann(t,a,b) for t in x]

a = 20
b = 200
y4 = [log_ann(t,a,b) for t in x]

# Log INv
# y5=[log_inv_ann(t,Ti) for t in x]

plt.figure(figsize=(4,3))
plt.plot(x,y1,x,y2,x,y3,x,y4)
plt.legend(['Linear', 'Exp', 'Log (a=5,b=1)', 'Log (a=20,b=200)'])
plt.xlabel('Passos')
plt.ylabel('Temperatura')
plt.tight_layout()



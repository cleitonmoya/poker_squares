# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:31:51 2021

@author: cleiton
"""

import numpy as np
from math import exp
import matplotlib.pyplot as plt

# Baralho
# (value, suit)
deck = np.empty(52,dtype=object)

deck1 = [(v,s)
        for s in ['\u2660','\u2661','\u2662','\u2663']   # 4 suits
        for v in range(1,14)] # 13 values

for i,e in enumerate(deck1):
    deck[i] = e

# Se todas as cartas são do mesmo naipe
def score_hand(v):
    values = sorted([c for c,_ in v])
    suits = [s for _,s in v]
    if all([s==suits[0] for s in suits]):
        if values == [1,10,11,12,13]:
            # royal flush
            return 30
        elif all([values[i+1]==values[i]+1 for i in range(4)]):
            # straight_flush
            return 30
        else:
            # flush
            return 5
    
    elif all([values[i+1]==values[i]+1 for i in range(4)]):
        # straight
        return 12
    
    else:
        x,c = np.unique(values, return_counts=True)
        c = np.array(c)
        if 4 in c:
            # 4 of a kind
            return 16
        elif 3 in c and 2 in c:
            # full_house
            return 10
        elif 3 in c:
            # 3 of a kind
            return 6
        elif (c==2).sum()==2:
            # 2 pairs
            return 3
        elif 2 in c:
            # 1 pair
            return 1
        else:
            return 0


def print_s(s):
    for row in s:
        print(''.join(str(c[0])+c[1]+'\t'  for c in row),score_hand(row))
    print('')
    print(''.join(str(score_hand(v))+'\t' for v in s.T)+' '+str(f(s)))

def f(s):
    scr_lin = sum([score_hand(v) for v in s])
    scr_col = sum([score_hand(v) for v in s.T])
    return scr_lin+scr_col


T = 2
def p_accept(delta,T):
    return min(1,exp(delta/T))

def permut(s0):
    i1,j1,i2,j2 = np.random.randint(5,size=4)
    s=s0.copy()
    s[i2][j2],s[i1][j1] = s[i1][j1],s[i2][j2]
    return s

np.random.seed(42)

N = 20
S = []
s0 = np.random.choice(deck, size=(5,5), replace=False) # Mesa (estado inicial)
s00 = s0
F = []
D = []
A = []
Elite = [(f(s0),s0)]

# Define uma mesa (estado) inicial
F.append(f(s0))
verb=False
for n in range(N):
    
    # Movimento proposto
    s = permut(s0)

    # Simulated Annealing
    fs = f(s)
    fs0 = f(s0)
    delta = fs-fs0
    D.append(delta)
    if verb: print(f'\nt{n}:Delta:', delta)
    
    # Se a proposta possui maior pontuação, aceita
    if delta>=0:
        s0=s
        F.append(fs)
        A.append(1)
        if verb: print(f't{n}: Delta>=0 ({delta}), aceitando')
        if fs> Elite[-1][0]:
            Elite.append((fs,s))
    
    # Caso contrário, aceita probabilisticamente
    else:
        p_a = p_accept(delta, T)
        if verb: print(f't{n}: pa: {p_a}')
        a = np.random.binomial(n=1,p=p_a)
        if a:
            if verb: print(f't{n}: aceitando')
            A.append(1)
            s0=s
            F.append(fs)
            if fs> Elite[-1][0]:
                Elite.append((fs,s))
        else:
            if verb: print(f't{n}: rejeitando')
            A.append(0)
            F.append(fs0)

print('Jogo inicial:')
print_s(s00)
print('\nMelhor solução:')
print_s(Elite[-1][1])

fig,ax = plt.subplots()
ax.plot(F)
ax.set_xticks(range(N))

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:31:51 2021

@author: cleiton
"""

import numpy as np
from math import exp, log
import matplotlib.pyplot as plt
from termcolor import colored
plt.rcParams.update({'font.size': 8})

# Baralho
# (value, suit)
deck = np.empty(52,dtype=object)

deck1 = [(v,s)
        for s in ['\u2660','\u2661','\u2662','\u2663']   # 4 suits
        for v in range(1,14)] # 13 values

for i,e in enumerate(deck1):
    deck[i] = e

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=
# Funções Auxiliares
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=

def load_table(file):
    
    table = []
    table2 =[]
    with open(file,'r') as f:
        rows = f.readlines()
        for i,r in enumerate(rows):
            r = r.rstrip('\n')
            cards = r.split(",")
            table.append(cards)
    f.close()
    
    for r in table:
        r2 = []
        for c in r:
            
            if c[0] == 'A':
                c2v = 1
            elif c[0] == 'J':
                c2v = 11
            elif c[0] == 'Q':
                c2v = 12
            elif c[0] == 'K':
                c2v = 13
            else:
                c2v = int(c[0])
                
            if c[1]=='s': # spade
                c2s = '\u2660'
            elif c[1]=='h': #heart
                c2s = '\u2661'
            elif c[1]=='d': #heart
                c2s = '\u2662'
            elif c[1]=='c': #club
                c2s = '\u2663'
            
            r2.append((c2v,c2s))
        table2.append(r2)
        
    dt=np.dtype('O,U1')
    return np.array(table2,dtype=dt).astype('object')

# Calcula a pontuação de uma mão (vetor)
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

# Calcula a pontuação de um jogo
def f(s):
    scr_lin = sum([score_hand(v) for v in s])
    scr_col = sum([score_hand(v) for v in s.T])
    return scr_lin+scr_col

# Probabilidade de aceite (Metropolis-Hastings)
def p_accept(delta,T):
    return min(1,exp(delta/T))

# Faz uma permutação aleatória entre dois elementos
def permut(s0):
    i1,j1,i2,j2 = np.random.randint(5,size=4)
    s=s0.copy()
    s[i2][j2],s[i1][j1] = s[i1][j1],s[i2][j2]
    return s

# Imprime um jogo
def print_s(s):
    for row in s:
        row2=row.copy()
        for j,p in enumerate(row):
            if p[0]==1:
                row2[j]=('A', p[1])
            elif p[0]==11:
                row2[j]=('J', p[1])
            elif p[0]==12:
                row2[j]=('Q', p[1])
            elif p[0]==13:
                row2[j]=('K', p[1])
        print(''.join(str(c[0])+c[1]+'\t'  for c in row2),colored(score_hand(row),'green'))
    print(''.join(colored(score_hand(v),'green')+'\t' for v in s.T),colored(f(s),'red',attrs=['bold']))


# Funções de decaimento da temperatura
def linear_ann2(Ti,Tf,N):
    return np.linspace(Ti,Tf,N)

def geom_ann2(Ti,Tf,N):
    return np.geomspace(Ti,Tf,N)

def log_ann2(t,a,b):
    return a*log(t+b)

def log_ann(t,a,b):
    return a/(log(t+b))


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=
# Simulated Annealing
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=

# Parâmetros gerais
verb=False
np.random.seed(42)

# Agenda de resfiamento
N = 5000
a = 8
b = 5
x = np.arange(1,N+1)
Annealing = [log_ann(t,a,b) for t in x]

# Define uma mesa (estado) inicial
# s0 = np.random.choice(deck, size=(5,5), replace=False) # Mesa (estado inicial)
s0 = load_table('s.txt')


s00 = s0
S = []
F = []
D = []
A = []
Elite = [(f(s0),s0)]
F.append(f(s0))

#%%
for n,T in enumerate(Annealing):
    
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
            
        else:
            if verb: print(f't{n}: rejeitando')
            A.append(0)
            F.append(fs0)

#%%
print('Jogo inicial:\n')
print_s(s00)
print('\nMelhor solução:\n')
print_s(Elite[-1][1])


fig,ax = plt.subplots(2,1)
ax[0].plot(F)
ax[0].set_xlabel('Iteração')
ax[0].set_ylabel('Score')
ax[1].plot(Annealing)
ax[1].set_xlabel('Iteração')
ax[1].set_ylabel('Temperatura')
plt.tight_layout()

plt.figure()
d,c = np.unique(D, return_counts=True)
plt.bar(-d,c,alpha=0.7)
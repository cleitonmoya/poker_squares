# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:31:51 2021

@author: cleiton
"""

import numpy as np
from math import exp, log
import matplotlib.pyplot as plt
from sty import fg, rs
import time
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

# Carrega um arquivo com um jogo específico
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
    
    # Converte símbolos
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
                if len(c)==3: # carta 10
                    c2v = 10
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
            return 27456
        elif all([values[i+1]==values[i]+1 for i in range(4)]):
            # straight_flush
            return 27456
        else:
            # flush
            return 215
    
    elif all([values[i+1]==values[i]+1 for i in range(4)]):
        # straight
        return 108
    
    else:
        x,c = np.unique(values, return_counts=True)
        c = np.array(c)
        if 4 in c:
            # 4 of a kind
            return 1760
        elif 3 in c and 2 in c:
            # full_house
            return 293
        elif 3 in c:
            # 3 of a kind
            return 20
        elif (c==2).sum()==2:
            # 2 pairs
            return 9
        elif 2 in c:
            # 1 pair
            return 1
        else:
            return 0

# Calcula a pontuação de um jogo
def f(s):
    scr_lin = sum([score_hand(v) for v in s])
    scr_col = sum([score_hand(v) for v in s.T])
    scr_d1 = score_hand(s.diagonal())
    scr_d2 = score_hand(np.fliplr(s).diagonal())
    return scr_lin + scr_col + scr_d1 + scr_d2

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
    
    d2_score = str(score_hand(np.fliplr(s).diagonal()))
    print('\t\t\t\t\t '+fg(255,255,0) +d2_score+fg.rs)
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
        rows_score = fg(10, 255, 10) + str(score_hand(row)) + fg.rs
        print(''.join(str(c[0])+c[1]+'\t'  for c in row2), rows_score)
    
    cols_score = fg(10, 255, 10) + ''.join(str(score_hand(v)) +'\t' for v in s.T) + fg.rs
    total_score = fg(255, 10, 10) + str(f(s)) + fg.rs
    d1_score = fg(255,255,0) + str(score_hand(s.diagonal())) + fg.rs
    print(cols_score, d1_score)
    print('\t\t\t\t\t '+fg(255,10,10) +total_score+fg.rs)

# Gera um jogo aleatório
def jogo_aleatorio():
    return np.random.choice(deck, size=(5,5), replace=False)

# Curvas de resfriamento
def annealing(tipo, N, Tf=None, Ti=None, a=None, b=None):
    if tipo == 'linear':
        return np.linspace(Ti,Tf,num=N-1)
    elif tipo == 'greedy':
        return np.zeros(N)
    elif tipo == 'log':
        x = np.arange(1,N+1)
        return [a/(log(t+b)) for t in x]
    elif tipo == 'exp':
        beta = (Tf/Ti)**(1/N)
        x = np.arange(1,N+1)
        return [Ti*beta**t for t in x]
    elif tipo=='log_inv':
        return  [(Ti/log(N))*log(N-1*t) for t in x]

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=
# Simulated Annealing
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=

# Parâmetros gerais

seed = np.random.randint(15000)
seed=14385
print(f'Seed = {seed}')
np.random.seed(seed)

# Agenda de resfiamento
N = 1000000
c_max = 1000000
x = np.arange(1,N+1)

Ti=1500
Tf = 1
Annealing = annealing(tipo='exp', N=N, Ti=Ti, Tf=Tf)

# Jogo inicial
s0 = load_table('jogos/jogo_sep89.txt')


F = []
Elite = [(f(s0),s0)]
F.append(f(s0))

n=0
c=0
tempo_inicial = time.time()
while n<N-1 and c<c_max:
    
    # Movimento proposto
    T = Annealing[n]
    s = permut(s0)

    # Simulated Annealing
    fs = f(s)
    fs0 = f(s0)
    delta = fs-fs0
    
    # Se a proposta possui maior pontuação, aceita
    if delta>=0:
        c=0
        s0=s
        F.append(fs)
        if fs> Elite[-1][0]:
            Elite.append((fs,s))
    
    # Caso contrário, aceita probabilisticamente
    else:
        c = c+1
        p_a = p_accept(delta, T)
        a = np.random.binomial(n=1,p=p_a)
        if a:
            s0=s
            F.append(fs)
            
        else:
            F.append(fs0)

    n=n+1
tempo_execucao = (time.time() - tempo_inicial)

print("\nConcluído em {0:1.2f}s".format(tempo_execucao))
print('Jogo inicial:\n')
print_s(Elite[0][1])
print('\nMelhor solução:\n')
print_s(Elite[-1][1])

fig,ax = plt.subplots(2,1)
ax[0].plot(F)
ax[0].set_xlabel('Passo')
ax[0].set_ylabel('Score')
ax[1].plot(Annealing[:n])
ax[1].set_xlabel('Passo')
ax[1].set_ylabel('Temperatura')
plt.tight_layout()
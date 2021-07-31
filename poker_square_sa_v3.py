# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:31:51 2021
Simula diversos jogos e diversas curvas
@author: cleiton
"""

import numpy as np
from math import exp, log
from scipy.stats import mode
import matplotlib.pyplot as plt
from sty import fg, rs
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
                
            if c[-1]=='s': # spade
                c2s = '\u2660'
            elif c[-1]=='h': #heart
                c2s = '\u2661'
            elif c[-1]=='d': #heart
                c2s = '\u2662'
            elif c[-1]=='c': #club
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
# delta = f(s)-f(s0)
def p_accept(delta,T):
    if T == 0:
        return 0
    else:
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
        rows_score = fg(10, 255, 10) + str(score_hand(row)) + fg.rs
        print(''.join(str(c[0])+c[1]+'\t'  for c in row2), rows_score)
    
    cols_score = fg(10, 255, 10) + ''.join(str(score_hand(v)) +'\t' for v in s.T) + fg.rs
    total_score = fg(255, 10, 10) + str(f(s)) + fg.rs
    print(cols_score, total_score)


# Gera um jogo aleatório
def jogo_aleatorio():
    return np.random.choice(deck, size=(5,5), replace=False)

# Curvas de resfriamento
def annealing(tipo, N, Tf=0.01, Ti=None, a=None, b=None, beta=None):
    if tipo == 'linear':
        return np.linspace(Ti,Tf,num=N-1)
    elif tipo == 'greedy':
        return np.zeros(N)
    elif tipo == 'log':
        x = np.arange(1,N+1)
        return [a/(log(t+b)) for t in x]
    elif tipo == 'exp':
        x = np.arange(1,N+1)
        beta = (Tf/Ti)**(1/N)
        return [Ti*beta**t for t in x]
    elif tipo=='log_inv':
        return  [(Ti/log(N))*log(N-1*t) for t in x]


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=
# Simulated Annealing
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=

# Parâmetros gerais
NJ = 100 # Número de jogos aleatório:
N = 10000    # número máximo de passos
verb=False # verbose

# Agenda de resfiamento
# Log: a = 7, b = 2
# Geom: beta= 0.99, b=4
Ti=5
Tf = 0.001

Ann1 = annealing(tipo='exp', N=N, Ti=Ti, Tf=0.001)
Ann2 = annealing(tipo='linear', N=N, Ti=Ti, Tf=0.001)
Ann3 = annealing(tipo='greedy', N=N)
ANN = [Ann1, Ann2, Ann3]
np.random.seed(42)

DF = {c:[] for c,_ in enumerate(ANN)}

for j in range(NJ):

    # Jogo inicial
    s00 = jogo_aleatorio()
    fs00 = f(s00)
    
    if j%5 == 0: print(f'Simulando jogo {j}')
    
    # Para cada curva
    for C,Ann in enumerate(ANN):
    
        s0 = s00
        b_s = s0
        b_fs = f(s0)
        
        for n,T in enumerate(Ann):
        
            s = permut(s0) # Movimento proposto
        
            # Simulated Annealing
            fs = f(s)
            fs0 = f(s0)
            delta = fs-fs0
    
            # Se a proposta possui maior pontuação ou igual, aceita
            if delta>=0:
                s0=s
                if fs> b_fs:
                    b_fs = fs
                    b_s = s
            
            # Caso contrário, aceita probabilisticamente
            else:
                p_a = p_accept(delta, T)
                a = np.random.binomial(n=1,p=p_a)
                if a:
                    s0=s
    
        delta_f = b_fs - fs00
        DF[C].append(delta_f)


#%%
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=
# Resultados
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=

label = ['Exp', 'Lin', 'T=0']

# Impressão dos resultados
TF = [0.0001, 0.001, 0.05, 0.5]
fig,ax = plt.subplots(3,1,sharex=True)

for c,_ in enumerate(ANN):
    print(f'Soma delta C{c}:', sum(DF[c]))

    # 
    x, fr = np.unique(DF[c], return_counts=True)
    pmf = fr/fr.sum()
    
    ax[c].bar(x,pmf,alpha=0.7)
    ax[c].axvline(np.array(DF[c]).mean(),c='r',linestyle='--', label='média')
    ax[c].set_title(f'PMF Empírica - {label[c]}')
    ax[c].set_xlabel('Delta')
    ax[c].set_ylabel('$p$')
plt.tight_layout()    


fig,ax = plt.subplots(3,1,sharex=True)
for c,_ in enumerate(ANN):
    x, fr = np.unique(DF[c], return_counts=True)
    cdf = fr.cumsum()/fr.sum()
    ccdf = 1-cdf
    ax[c].scatter(x,ccdf,alpha=0.7,s=5)
    ax[c].set_title(f'CCDF Empírica - {label[c]}')
    ax[c].set_xlabel('$Delta$')
    ax[c].set_ylabel('$1-CDF(Delta)$')
plt.tight_layout()  
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:31:51 2021
Poker Squares - Aleatory Player
@author: cleiton
"""

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
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

# Faz uma permutação aleatória entre dois elementos
def permut(s0):
    i1,j1,i2,j2 = np.random.randint(5,size=4)
    s=s0.copy()
    s[i2][j2],s[i1][j1] = s[i1][j1],s[i2][j2]
    return s


# Gera um jogo aleatório
def jogo_aleatorio():
    return np.random.choice(deck, size=(5,5), replace=False)

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=
# Simulação
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=

# Parâmetros gerais
verb=False
np.random.seed(42)

# número de jogos
N = 100000

# Número de jogadas
T = 0

# Jogo inicial
F = []
D = []
DF = []
for n in range(N):
    if n%10000 == 0: print(f'Jogo {n}')
    s0 = jogo_aleatorio()
    #for t in range(T):
        # Troca aleatória
    s = permut(s0)
    fs = f(s)
    fs0 = f(s0)
    F.append(fs)
    D.append(fs-fs0)
        #s0 = s

F = np.array(F)
D = np.array(D)
D2 = np.array([d for d in D if d<0])


# Resultados
print('Pontuação mín.:', F.min())
print('Pontuação máx.:', F.max())
print('Pontuação média.:', F.mean())
print('Moda:', mode(F)[0][0])
print('Desvio padrão:', F.std())

print('\nDelta mín.:', D.min())
print('Delta máx.:', D.max())
print('Delta média.:', D.mean())
print('Moda:', mode(D)[0][0])
print('Desvio padrão:', D.std())

print('Score negativos')
print('Média:', D2.mean())



# Score
plt.figure(figsize=(4,3))
f,c = np.unique(F, return_counts=True)
plt.bar(f,c,alpha=0.7)
plt.title('Pontuação')
plt.tight_layout()


# Delta score
plt.figure(figsize=(4,3))
d,c = np.unique(D, return_counts=True)
plt.bar(d,c,alpha=0.7)
plt.title(r'$\Delta f(s)$')
plt.tight_layout()

plt.figure(figsize=(4,3))
plt.boxplot(D,showfliers=False)
plt.title(r'$\Delta f(s)$')

# Delta score negativos
plt.figure(figsize=(4,3))
d2,c2 = np.unique(D2, return_counts=True)
plt.bar(d2,c2,alpha=0.7)
plt.axvline(D2.mean(),c='r',linestyle='--', label='média')
plt.title(r'$\Delta f(s)$')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(4,3))
plt.boxplot(D2,showfliers=False)
plt.title(r'$\Delta f(s)$')

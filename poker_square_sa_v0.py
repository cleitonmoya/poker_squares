# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:31:51 2021
Simula um jogo e uma curva
@author: cleiton
"""

import numpy as np
from math import exp, log
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
def jogo_aleatorio(seed=None):
    if seed == None:
        seed = np.random.randint(0,15000)
    print(f'seed={seed}')
    np.random.seed(seed)
    return np.random.choice(deck, size=(5,5), replace=False)

# Embaralha um jogo
def embaralha(s):
    s1 = s.reshape((25,))
    np.random.shuffle(s1)
    s2 = s1.reshape(5,5)
    return s2

# Curvas de resfriamento
def annealing(tipo, N, Ti=None, Tf=None, a=None, b=None):
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
N = 1000        # número máximo de passos
c_max = 1000     # número máximo de passos sem melhoria
verb=False      # verbose

# Agenda de resfiamento
# Log: a = 7, b = 2
# Geom: beta= 0.99, b=4
Ti=5
Tf = 0.001
a = 3.5
b = 2

Ann = annealing(tipo='exp', N=N, Ti=Ti, Tf=Tf)

# Jogo inicial
# s0 = jogo_aleatorio(3439)
s0 = load_table('jogos/jogo_print.txt')
print_s(s0)
s0 = embaralha(s0)
print_s(s0)

#%%
S = []
F = []
D = []
A = []
PA = [1]
Elite = [(f(s0),s0)]
F.append(f(s0))

n = 0
c = 0
np.random.seed(42)

while n<N-1 and c<c_max:

    T = Ann[n]     # Temperatura
    s = permut(s0) # Movimento proposto

    # Simulated Annealing
    fs = f(s)
    fs0 = f(s0)
    delta = fs-fs0
    D.append(delta)
    if verb: print(f'\nt{n}:Delta:', delta)
    
    p_a = p_accept(delta, T)
    if p_a != 1:
        PA.append(p_a)
    else:
        PA.append(PA[-1])
    
    # Se a proposta possui maior pontuação, aceita
    if delta>0:
        c = 0
        s0=s
        F.append(fs)
        A.append(1)
        if verb: print(f't{n}: Delta>=0 ({delta}), aceitando')
        if fs> Elite[-1][0]:
            Elite.append((fs,s))
    
    # Caso contrário, aceita probabilisticamente
    else:
        c = c+1
        #p_a = p_accept(delta, T)
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
    n=n+1


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=
# Resultados
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-=-=

# Impressão dos resultados
print('Jogo inicial:\n')
print_s(Elite[0][1])
print('\nMelhor solução:\n')
print_s(Elite[-1][1])

# Score e temperatura
fig,ax = plt.subplots(2,1)
ax[0].plot(F)
ax[0].set_xlabel('Passo')
ax[0].set_ylabel('Score')

ax[1].plot(Ann[:n])
ax[1].set_xlabel('Passo')
ax[1].set_ylabel('Temperatura')
plt.tight_layout()

# Probabilidade de aceite
plt.figure()
plt.plot(PA)
plt.title('Probabilidade de aceite')
ax[0].set_xlabel('Passo')

# Taxa média de aceite
plt.figure()
a_medio = [np.mean(A[:n+1]) for n,_ in enumerate(A)]
plt.plot(a_medio)
plt.title('Taxa média de aceite')

# Delta scores
plt.figure()
plt.plot(D)
ax[1].set_xlabel('Passo')
plt.ylabel(r'$f(s)-f(s_0)$')

# Delta scores - histograma
plt.figure()
d,c = np.unique(D, return_counts=True)
plt.bar(d,c,alpha=0.7)
plt.xlabel(r'$f(s)-f(s_0)$')
plt.ylabel('Freq.')

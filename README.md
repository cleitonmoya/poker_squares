# Playing Poker Squares with Simulated Annealing

**Poker Squares** (aka Poker Solitaire) is patiance type of game. There are 25 cards in a 5x5 grid. Each row and column is like a poker hand and has a score. Summed up, rows and columns scores give the total score. The aim of the game is to arranje the cards in the grid in order to maximize the total score.

There are many variations of game style and scores. In this work, we adopt a game style in witch is possible to permutate the cards on the grid without restrictions. Someones call this style **Poker Shuffle**. Also, we adopt the *English Point System* as stated in [Wikipedia](https://en.wikipedia.org/wiki/Poker_squares). 

Despites their simplicity, exactly solving a Poker Square game is nowaydays prohibitiv because there are ![formula](https://render.githubusercontent.com/render/math?math=\color{red}\frac{25!}{10!}\approeq 10^{18}) possible permutations of cards. So, we have to solve this Combinatorial Optization Problem using some other approach.

In  this project we develop an agent that plays Poker Squares using **Simulated Annealing**. Simmulated Annealing is a Metha-Heuristic and Monte Carlo type Alghoritm based on the **Metropolis Hastings Alghorithm**. In turn, Metropolis Hastings is one of the most important alghorithims of the 20th century. Despite there are no guarantee of achieving the global optmimum, in practical solutions are usually good enough.

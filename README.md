# Playing Poker Squares with Simulated Annealing

**Poker Squares** (aka Poker Solitaire) is a patiance-styled game. There are 25 cards in a 5x5 grid. Each row and column forms a poker hand with a score. Summed up, they give a total score. The aim of the game is to arranje the cards in the grid in order to maximize the total score. So, it's a Combinatorial Optimization Problem.

<img src="jogos/jogo1.jpg" alt="Poker Squares" width="200" class="center"/>

There are many variations in terms of game style and scores. In this work, we adopt a game style (sometimes called **Poker Shuffle**) in witch is possible to permutate the cards on the grid without restrictions. Also, we adopt the *English Point System* [Wikipedia](https://en.wikipedia.org/wiki/Poker_squares). 

Despites their simplicity, exactly solving a Poker Square game is prohibitiv because there are ![formula](https://render.githubusercontent.com/render/math?math=\color{red}\frac{25!}{(5!)^2}\approxeq10^{21}) possible permutations of cards. So, we shall to attack the problem using some other approach.

In this project we play Poker Squares using **Simulated Annealing**. Simmulated Annealing is a Metaheuristic and Monte Carlo type Algorithm based on the **Metropolis Hastings** (in turn, one of the most important algorithms of the 20th century). Although there are no guarantee of achieving the global optmimum, in practical the solutions are usually good enough.

With a simple implementation I got mean score of 78 - above my own average and probably above the average of an average human - and a maximum score of 131 - this would be at **top 3** of ranking of [Poker Solitaire by Bearded Whale] (https://play.google.com/store/apps/details?id=com.beardedwhale.pokersolitaire&hl=en_US&gl=US). I used a exponential annealing changing every step, starting with T=5 and ending with T=0.001, 10,000 steps. I ran 1,000 games.

Also I was able to get 4,500 points using the same point-system and initial tableau proposed by [McLaughlin, 1989](http://dns.uls.cl/~ej/daa_08/Algoritmos/books/book10/8909b/8909b.htm), while he got 4,516 but with a more complex algorithm.

How is your performance in Pokker Shuffle? Can you beat my mean score, or the mean score of my Simulated Annealing implementation? Shall we play a game? :)
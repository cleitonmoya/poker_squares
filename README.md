# Playing Poker Squares with Simulated Annealing

**Poker Squares** (aka Poker Solitaire) is a patience game. There are 25 cards in a 5x5 grid. Each row and column forms a poker hand with a score, which is added up to give a total score. The game aims to arrange the cards in the grid to maximize the total score, so it's a Combinatorial Optimization Problem.

<p align="center">
	<img src="jogos/jogo1.jpg" alt="Poker Squares" width="200"/>
</p>

There are many variations in terms of game style and scores. In this work, we adopt a game style (sometimes called **Poker Shuffle**) in which it is possible to permutate the cards on the grid without restrictions. Also, we adopt the *English Point System* [Wikipedia](https://en.wikipedia.org/wiki/Poker_squares). 

Despite their simplicity, exactly solving a Poker Square game is prohibitive because there are ![formula](https://render.githubusercontent.com/render/math?math=\color{red}\frac{25!}{(5!)^2}\approxeq10^{21}) possible permutations of cards. So, we shall tackle the problem using some other approach.

In this project we play Poker Squares using **Simulated Annealing**. Simulated Annealing is a Metaheuristic and Monte Carlo-type Algorithm based on the **Metropolis-Hastings** (in turn, one of the most important algorithms of the 20th century). Although there is no guarantee of achieving the global optimum, the practical solutions are usually good enough.

With a simple implementation, I got a mean score of 78—above my own average and probably above the average of an average human—and a maximum score of 131—this would be in the top 3 of the ranking of [Poker Solitaire by Bearded Whale](https://play.google.com/store/apps/details?id=com.beardedwhale.pokersolitaire&hl=en_US&gl=US). I used exponential annealing, changing every step, starting with T=5 and ending with T=0.001, 10,000 steps. I ran 1,000 games.

Also, I was able to get 4,500 points using the same point system and initial tableau proposed by [McLaughlin, 1989](http://dns.uls.cl/~ej/daa_08/Algoritmos/books/book10/8909b/8909b.htm), while he got 4,516 but with a more complex algorithm.

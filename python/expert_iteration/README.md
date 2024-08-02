Expert Iteration (ExIt), from [Thinking Fast and Slow with Deep Learning and Tree Search (Anthony, Tian, Barber 2017)](https://arxiv.org/abs/1705.08439).

## Status
The performance on small boards is good. In order to scale it up, I will need to:
- Split into learner and simulator processes. I am not learner-bound currently.
- Simulate many positions in parallel.
- Improve the performance of the naive Hex simulator.
- Improve the performance of the naive MCTS implementation.

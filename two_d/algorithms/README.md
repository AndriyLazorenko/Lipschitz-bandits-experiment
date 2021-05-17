### Implemented algorithms:

**Zooming family:**
- **Zooming**
- **ADTM**
- **ADMM**

All of the above inspired by https://www.lamda.nju.edu.cn/lusy/pdf/lu2019optimal.pdf

**Classical bandits (extended to infinitely-many arms):**
- **Epsilon-greedy** (DIY port to CAB setup)
- **UCB** - inspired by https://proceedings.neurips.cc/paper/2008/file/49ae49a23f67c759bf4fc791ba842aa2-Paper.pdf
- **Thompson sampling** - inspired by https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-thompson-sampling-algorithm-6375271f40d1
  (bugged implementation - needs debugging!)

**Bayesian optimization** 

**LBS (from paper 1)**

There are 2 baselines implemented for comparison's sake:
- **Random algorithm** (select actions at random without learning)
- **Optimal algorithm** (knows underlying reward function)

**Planned (not implemented) algorithms:**
- **UCB-Gaussian** according to https://towardsdatascience.com/multi-armed-bandits-upper-confidence-bound-algorithms-with-python-code-a977728f0e2d
- **Thompson sampling** - debugged
- **https://arxiv.org/pdf/1902.01520.pdf**

All of the algorithms are implemented within `two_d/algorithms` package, according to a single
convention of interaction with the outside world (`abstract_algorithm.py` contains
the abstract class)

Common methods for all algorithms:
* `initialize` - a method called to (re)set all fields and data structures
* `get_arm_value` - a method retrieving the arm value based on the algorithm's logic
* `learn` - a method implementing "learning" of the algorithm based on reward obtained from an arm pull

For all of the algorithms except for `thompson_sampling` and `LBS` the following
methods are implemented:
* `get_arms_batch` - a method that retrieves a batch of arms values
* `batch_learn` - a method that facilitates learning on a batch of arms

### Known bugs

Several algorithms are known to underperform outside of [0,1] range.
It is likely due to hard-coded bug in implementations of these algorithms.
The bug facilitates hard coupling of those algorithms to [0,1] interval
and ensures they don't work as intended outside this interval.

Therefore, it is not advised to use them outside [0,1] interval.
The list of malfunctioning algorithms:

 - Thompson sampling
 - Zooming
 - ADTM
 - ADMM

It is therefore advised to use the following 2 algorithms in experiments
on production:
 - Epsilon-greedy bandit
 - Bayesian optimization

as they have shown the best performance in simulated benchmark
and are known to work as intended both on intervals and batches.   
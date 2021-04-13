# Lipshitz-bandits experiments

### A testbed for function optimization on an interval

### Usage

Main usecase is related to testing multiple algorithms on optimization task
on an interval. The interval is selected to be [0,1]. The optimization task is
finding a maximum of a given function on an interval. See below for an exhaustive 
list of all the reward functions (used in simulations) and algorithms implemented.

Once installed according to guide below, run:
```
python run.py
```
This will start a lengthy (~4 hours on a typical laptop) process of
simulations of maximum search on an interval using all the 
implemented algorithms. The algorithm and simulation parameters are
recorded in `resources/experiments.csv`. The results are visualized and 
stored as `.png` images in `resources` folder. 

It is recommended to use PyCharm to execute code.

To quickly observe some results and to gain a general intuition on how
the testbed works, comment out `run_all()` function and uncomment `run_one()`.
This will run a single experiment with small number of iterations. Results
are recorded in image form in `resources` folder


### Installation
First, create a conda environment from a terminal with
```
conda create --name zooming_bandits python=3.8
```
Second, activate the environment:
```
source activate zooming_bandits
```

Third, install all dependencies:
```
pip install -r requirements.txt
```

### Implemented reward functions

- **Triangular reward**, taken from https://www.lamda.nju.edu.cn/lusy/pdf/lu2019optimal.pdf
- **Quadratic reward**, taken from https://arxiv.org/pdf/1405.4758.pdf


### Implemented algorithms

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

There are 2 baselines implemented for comparison's sake:
- **Random algorithm** (select actions at random without learning)
- **Optimal algorithm** (knows underlying reward function)

**Planned algorithms:**
- **UCB-Gaussian** according to https://towardsdatascience.com/multi-armed-bandits-upper-confidence-bound-algorithms-with-python-code-a977728f0e2d
- **Thompson sampling** - debugged
- **https://arxiv.org/pdf/1902.01520.pdf**

All of the algorithms are implemented within `algorithms.py` module, according to a single
convention of interaction with the outside world.

Common methods for all algorithms:
* `initialize` - a method called to (re)set all fields and data structures
* `get_arm_index` - a method retrieving the arm index based on the algorithm's logic
* `learn` - a method implementing "learning" of the algorithm based on reward obtained from an arm pull

### Helper modules

`timer.py` is used to annotate functions to time them

`bayes_opt.py` is used in experiments with reward visualizations and bayesian optimization


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
## Function optimization in AdTech

Contains the following apps:
1. A testbed for function optimization on an interval (R1)
2. A testbed for function optimization on a landscape (R2)
3. A dataset generator producing synthetic bids data
4. A visualizer to showcase synthetic reward functions used for function optimization
5. An implementation of research paper 1.

The codebase is heavily influenced by 2 research papers:

1. Learning to Bid Optimally and Efficiently
in Adversarial First-price Auctions (https://arxiv.org/pdf/2007.04568.pdf)

2. Optimal Algorithms for Lipschitz Bandits with Heavy-tailed Rewards (http://proceedings.mlr.press/v97/lu19c/lu19c.pdf)

### A testbed for function optimization on an interval (R1)

Main usecase is related to testing multiple algorithms on optimization task
on an interval. The interval is selected to be [0,1]. The optimization task is
finding a maximum of a given function on an interval. 

The function optimization on an interval is similar to finding an optimal parameter theta
in a linear bid shading first-price auction bidding strategy. The search for an optimal theta
using the algorithms above can be a substitute to grid-based search from paper 1 under
constraint of hidden mt information. In case of missing mt information, we can create a DIY
reward function to guide the algorithms optimizing the performance according to the reward 
function selected.

This app also contains implementation of LBS algorithm from paper 1 utilizing simulated pairs (mt, vt).
The LBS algorithm is assessed against the other algorithms on a simulated dataset for efficiency 
according to reward function used in paper.

See below for an exhaustive list of all the reward functions (used in simulations) and algorithms implemented.

It is recommended to use PyCharm to execute code.

#### App 1 usage

Once installed according to guide below, run:
```
python two_d/experiments_2d.py
```
This will start a lengthy (~4 hours on a typical laptop) process of
simulations of maximum search on an interval using all the 
implemented algorithms (see complete list of implemented algorithms
for 2D here: `two_d/algorithms/README.md`)

To make the process shorter, while making results more stochastic and less
representative, decrease parameter `"trials"` found in `configs/experiments_template.json`
from 100 to 10.

The algorithm and simulation parameters are
recorded in `resources/2d/experiments.csv`. The results are visualized and 
stored as `.png` images in `resources/2d` folder. 

The experiments described above showcase algorithms performance
only on two synthetic reward functions (triangular and quadratic, see below
for description of reward functions). This gives a valuable
insight into algorithm's ability to find optimal parameter
theta under varying reward noise.

#### App 2 usage
Once installed according to guide below, run:
```
python three_d/experiments_3d.py
```
This will start a lengthy process of
simulations of maximum search on an interval using all the 
implemented algorithms (see a complete list of algorithms implemented
in 3D here: `three_d/algorithms/README.md`)

The algorithm and simulation parameters are
recorded in `resources/3d/experiments.csv`. The results are visualized and 
stored as `.png` images in `resources/3d` folder. 

The experiments described above showcase algorithms performance
only on two synthetic reward functions (bukin and rosenbrock, see 
below for description of synthetic reward functions).
This gives a valuable
insight into algorithm's ability to find optimal parameter set
theta1, theta2 under varying reward noise.

#### App 3 usage

To generate a dataset based on a shape of distribution observed
in paper 1 (dataset B), run the following:

```
python utils/scenario_generator.py
```

It will generate a dataset of 60 days each containing 10'000 datapoints.
Each datapoint is a 3-tuple of (vt, mt, day). The data is then stored in
`resources/campaign_scenario.csv`

vt-mt pairs are each generated using a specific gamma distribution in a way
to ensure that the final result's distribution somewhat resembles 
the dataset B distribution from the paper 1. This synthetic data is
later used to evaluate LBS, NLBS and SEW algorithms from paper 1 along
with competing algorithms that are not supplied with mt,vt pairs in app 5.

#### App 4 usage
To visualize various synthetic reward shapes (2D) and lanscapes (3D),
run the following:
```
python utils/reward_visualizer.py
```
This will generate and save the visualizations of the 4 synthetic
reward functions used to assess algorithms running optimization of
function on an interval (both 2D and 3D), specifically:
triangular reward, quadratic reward, bukin reward and rosenbrock reward

The shapes are saved in `resources/reward_shapes` as png files along
with meta-description saved in `resources/reward_shapes/info.csv`

#### App 5 usage

To run an implementation of paper 1 on synthetically-generated dataset
of 600'000 {vt, mt} pairs, run the following:
```
python utils/experiments.py
```
This will start a single 10-iterations evaluation run. Within this run,
all the algorithms present in `three_d/configs/algorithms.json` are used
to come up with bid prices on simulated bidding dataset. Their 
aggregated performance across 60 days is visualized and stored in
`resources/3d/` folder.

It is possible to adjust number of iterations and other evaluation
run parameters using `three_d/configs/experiment.json` file.

LBS, NLBS and SEW algorithms implemented from paper 1 will be assessed
along with other algorithms (which are agnostic of mt values).

#### Installation
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

#### Implemented synthetic reward functions

##### Two_d:
- **Triangular reward**, taken from https://www.lamda.nju.edu.cn/lusy/pdf/lu2019optimal.pdf
- **Quadratic reward**, taken from https://arxiv.org/pdf/1405.4758.pdf

##### Three_d:
- **Bukin reward**, inspired by https://en.wikipedia.org/wiki/Test_functions_for_optimization
- **Rosenbrock reward**, inspired by https://en.wikipedia.org/wiki/Test_functions_for_optimization

##### Article reward:

A reward mentioned in article 1. It is completely dependent on 
having access to historical data on oracle's predictions (vt) and
minimal winning bid price (mt) needed to win a specific auction.
Gamma distribution was used in order to generate a dataset similar
to a closed-sourced dataset used in article 1. A function generating
the dataset is present in `utils/scenario_generator.py`

It can be used to:
* evaluate a LBS algorithm and other 2d algorithms
used for learning `theta` parameter for linear bid shading strategy
  
* evaluate a NLBS algorithm and other 3d algorithms used for learning 
`theta1, theta2` params for non-linear bid shading strategy
  
* evaluate SEW algorithm providing a custom mapping from vt to bt.

#### Implemented algorithms

For algorithms used in determining `theta` param used in linear bid
shading strategy,
see detailed description in `two_d/algorithms/README.md`

For algorithms used in determining `theta1, theta2` params used
in non-linear bid shading strategy as well as SEW algorithm,
see detailed description in `three_d/algorithms/README.md`

#### Helper modules

* `timer.py` is used to annotate functions to time them (time profiling)
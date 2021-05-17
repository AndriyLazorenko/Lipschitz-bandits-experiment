### Implemented algorithms:

**Bayesian optimization**

**NLBS (from paper 1)**

**SEW**

There are 2 baselines implemented for comparison's sake:
- **Random algorithm** (select actions at random without learning)
- **Optimal algorithm** (knows underlying reward function, only for synthetic landscapes)

All of the algorithms are implemented within `three_d/algorithms` package, according to a single
convention of interaction with the outside world (`abstract_algorithm.py` contains
the abstract class).

Common methods for all algorithms:
* `initialize` - a method called to (re)set all fields and data structures
* `get_arm_value` - a method retrieving the arm value based on the algorithm's logic
* `learn` - a method implementing "learning" of the algorithm based on reward obtained from an arm pull

**SEW** doesn't conform to that convention, however.
It uses `get_bid_price` method instead of get_arm_value.
It also uses `update` method instead of `learn`.

It also uses a different evaluation policy, as it updates
itself after each of the decisions.

### Known bugs
**SEW** is likely to have some bugs, as its performance is subpar compared
to every other algorithm implemented.

There are several reasons that might have caused it:
- The synthetically generated `{vt, mt}` pairs radically differ from real-world `{vt, mt}` pairs.
This makes comparison on synthetic data non-indicative of real world performance.
  
- bt generating part of SEW is wrong, as it allows for bt to be 
higher than vt, which leads to loss even if the bid was won. It might
  be fixed by adding a cap to ensure bt <= vt.
  
- An index error exists in calculations involving tensors or vectorized operations.
To fix them one would need to recalculate all values by hand and use those as tests.

- An implementation error exists, mathematics is not correctly transferred to code.
To detect it, one would need to proof-read the article's maths and 
  its implementation to search for errors.
  
It is recommended to attempt NLBS (or NLBS and LBS in A/B setting) in production, as it is the best-performing
algorithm on the synthetic dataset. SEW needs debugging and tweaking before production.


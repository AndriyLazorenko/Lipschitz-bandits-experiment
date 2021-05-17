# print(sorted([('abc', 121), ('abc', 148), ('abc', 221), ('abc', 231)], key=lambda x: x[1], reverse=True)[0][0])

# ais = (0., 1.)
# print(tuple([i for i in ais]))
from pprint import pprint
from time import sleep
from typing import Tuple

import numpy as np

from varname import nameof

#
# x = np.linspace(0, 1, 30)
# y = np.logspace(-2, 1/2, 30)
# print(y)

# v = np.random.uniform(0, 10)
# print(v)

# from skopt.space.space import Space
#
# s = Space([[0., 1.], [0., 10.]])
# print(s.dimensions)
# print(s.dimension_names)
# print(s.n_dims)
# print(s.rvs(10))

# vt = np.array([2, 1, 5, 2])  # y axis
# mt = np.array([1, 2, 3, 4])  # x axis
# bt = vt * 0.6
# print(bt)
# profit = np.subtract(vt, bt)
# mask = bt <= mt
# print(mask)
# m = np.ma.masked_where(y > 5, y)  # filter out values larger than 5
# new_x = np.ma.masked_where(np.ma.getmask(m), x)  # applies the mask of m on x
# new_profit = np.ma.masked_where(mask, profit)
# print(profit)
# print(new_profit)
# print(new_profit.sum())
# a = np.array([1, 2, 3])
# b = np.array([9, 8, 7])
# print(a)
# print(b)
# c = np.vstack((a, b)).T
# a_a, b_b = np.meshgrid(a, b)
# print(a_a.reshape(-1))
# print(b_b.reshape(-1))
# print(c)
# for el in c:
#     print(el)
# print(x)
# print(y)
from varname.helpers import Wrapper

#######################################################
# SECTION INIT #
from three_d.algorithms.sew import SEW

T = 60
L = int(np.log2(np.sqrt(T)))

print(f"L: {L}")

M = np.array([np.power(2, l + 1) for l in range(1, L + 1)])
U = np.array([np.power(2, l + 1) - 1 for l in range(1, L + 1)])
W = np.array([np.power(2, l) - 1 for l in range(1, L + 1)])

print(f"M: {M}")
print(f"U: {U}")
print(f"W: {W}")

# I = [[((m - 1) / M[l], m / M[l]) for m in range(M[l])] for l in range(L + 1)]
# for l in range(L):
#     for m in range(1, M[l]+1):
#         print(((m-1)/M[l], (m)/M[l]))
I = np.asarray([[((m - 1) / M[l], (m) / M[l]) for m in range(1, M[l] + 1)] for l in range(L)])
# I = [[((m - 1) / M[l], m / M[l]) for m in M] for l in range(L + 1)]
print("I: ")
pprint(I)

T = np.asarray([[0 for m in range(1, M[l] + 1)] for l in range(L)])

print(f"T: {T}")

print(f"W: {W}")
print(f"U: {U}")
b = np.asarray([[np.multiply(np.power(2., -l), (w + 1)) for w in range(1, W[l - 1] + 1)] for l in
                range(1, L + 1)])  # might contain index errors
# b = [np.power(2, -l) * (w + 1) for l in range(L) for w in W]
print(f"b: {b}")

R = np.asarray(
    [[[0 for u in range(1, U[l] + 1)] for m in range(1, M[l] + 1)] for l in range(L)])  # might contain index errors
R_dash = np.asarray(
    [[[0 for w in range(1, W[l] + 1)] for m in range(1, M[l] + 1)] for l in range(L)])  # might contain index errors

print("R: ")
pprint(R)
print("R_dash: ")
pprint(R_dash)


def infer_m(l: int, vt: float):
    for m in range(M[l]):
        if I[l][m][0] < vt <= I[l][m][1]:
            return m


##########################################################
# SECTION GET ARM #

vt = np.random.uniform()  # 0.4938064747677938
mt = np.random.uniform()

print(f"vt: {vt}, mt: {mt}")

ms = list()
for l in range(L):
    m = infer_m(l, vt)
    ms.append(m)

print(f"ms: {ms}")
for l in range(L):
    T[l][ms[l]] += 1

print(f"T: {T}")
# c = np.log2(774)
# print(c)
# print(W[0])
# for l in range(1, L+1):
#     for w in range(1, W[l-1]+1):
#         print(np.multiply(np.power(2., -l), (w + 1)))


prob = list()
for l in range(1, L + 1):
    entry_l = list()
    for w in range(1, W[l - 1] + 1):
        entry = SEW.exponential_weighting(timestep=T[l - 1][ms[l - 1]],
                                          suboptimality_gap=np.power(2., 1 - l),
                                          rewards=(
                                              R[l - 1][ms[l - 1]][2 * w - 1 - 1],  # Indices might be wrong
                                              R[l - 1][ms[l - 1]][2 * w - 1],  # Indices might be wrong
                                              R[l - 1][ms[l - 1]][2 * w + 1 - 1],  # Indices might be wrong
                                              R_dash[l - 1][ms[l - 1]][w - 1]  # Indices might be wrong
                                          )
                                          )
        entry_l.append(entry)
    prob.append(entry_l)
print(f"prob: {prob}")

w_star = 1
action = None
for l in range(1, L + 1):
    distribution = prob[l - 1][w_star - 1]
    s = np.random.choice([1, 2, 3, 4], replace=False, p=distribution)
    if s == 4:
        action = b[l - 1][w_star - 1]
        break
    elif l < L:
        w_star = 2 * (w_star - 1) + s
    else:
        action = np.power(2., -L - 1) * (2 * (w_star - 1) + s)
        break

print(f"action: {action}")

#################################################
# SECTION LEARN #

r_dash_cached = [0 for i in range(L)]
r_cached = [0 for i in range(L)]
for l in range(L, 0, -1):
    print(f"l:{l}")
    dash_cache = list()
    for w in range(1, W[l - 1] + 1):
        print(f"w:{w}")
        r_dash = SEW.compute_reward(b[l - 1][w - 1], vt, mt)
        dash_cache.append(r_dash)
        R_dash[l - 1][ms[l - 1]][w - 1] += r_dash  # indices might be wrong
    r_dash_cached[l - 1] = dash_cache
    print(f"r_dash_cached: {r_dash_cached}")
    cache = list()
    for u in range(1, U[l - 1] + 1):
        print(f"u:{u}")
        if l == L:
            r = SEW.compute_reward(np.power(2., -L - 1) * u, vt, mt)
        else:
            r = prob[l][u - 1][3] * r_dash_cached[l][u - 1]  # suspicious l (l+1) index
            for s in range(3):
                r += prob[l][u - 1][s] * r_cached[l][2 * (u - 1) + s]  # suspicious l and 2*(u-1)+s indices
        cache.append(r)
        print(f"cache: {cache}")
        R[l - 1][ms[l - 1]][u - 1] += r  # indices might be wrong
    r_cached[l - 1] = cache
    print(f"r_cached: {r_cached}")

print("R")
pprint(R)
print("R_dash")
pprint(R_dash)
# s = np.random.choice()
# print(T[0][ms[0]])
# print(T[1][ms[1]])
# print(T[0])
# print(T[1])

# TODO: write overall documentation and monthly report.
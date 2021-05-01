# print(sorted([('abc', 121), ('abc', 148), ('abc', 221), ('abc', 231)], key=lambda x: x[1], reverse=True)[0][0])

# ais = (0., 1.)
# print(tuple([i for i in ais]))

import numpy as np

#
# x = np.linspace(0, 1, 30)
y = np.logspace(-2, 1/2, 30)
print(y)

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

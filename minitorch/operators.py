"""Collection of the core mathematical operators used throughout the code base."""

import math
import os
os.system("pip install mypy")

# ## Task 0.1
# from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x, y):
    return x * y


def id(x):
    """Returns the input unchanged"""
    return x


def add(x, y):
    return x + y


def neg(x):
    return -x


def lt(x, y):
    return x < y


def eq(x, y):
    return abs(x - y) < 1e-5


def max(x, y):
    if x > y:
        return x
    return y


def is_close(x, y, eps=1e-2):
    if abs(x - y) < eps:
        return 1
    return 0


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x):
    return max(x, 0)


def log(x):
    return math.log(x)


def exp(x):
    return math.exp(x)


def log_back(x, y):
    """dlog(x)/dx * y"""
    if x == 0:
        return float("inf")
    return y * 1.0 / x


def inv(x):
    """1/x"""
    if x == 0:
        return float("inf")
    return 1.0 / x


def inv_back(x, y):
    """dinv(x)/dx * y"""
    if x == 0:
        return float("inf")
    return -y * 1.0 / (x * x)


def relu_back(x, y):
    """drelu(x)/dx * y"""
    if x > 0:
        return y
    return 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(func, iter):
    """Applies a given function to each element of an iterable"""
    return [func(i) for i in iter]


def zipWith(func, iter1, iter2):
    """Combines elements from two iterables using a given function"""
    return [
        func(i, j) for i, j in zip(iter1, iter2)
    ]  # zip()会将iter1和iter2中的元素按位置配对(若不等长，截取短的)成元组: (iter1[1], iter2[1]), (iter1[2], iter2[2])...


def reduce(func, iter, now):
    """Reduces an iterable to a single value using a given function"""
    for i in iter:
        now = func(i, now)
    return now


def negList(ls):
    """Negate all elements in a list using map"""
    return map(neg, ls)


def addLists(ls1, ls2):
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add, ls1, ls2)


def sum(ls):
    """Sum all elements in a list using reduce"""
    return reduce(add, ls, 0)


def prod(ls):
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, ls, 1)

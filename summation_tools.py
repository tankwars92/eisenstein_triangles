from math import isqrt

import numpy as np

NUMPY_LIMIT = 4 * 10 ** 17
CHUNK = 1 << 22


def floor_sum(n, m, a, b):
    total = 0
    while True:
        if a >= m:
            total += (n - 1) * n // 2 * (a // m)
            a %= m
        if b >= m:
            total += n * (b // m)
            b %= m
        top = a * n + b
        if top < m:
            break
        n, b = divmod(top, m)
        m, a = a, m
    return total


def range_floor_sum(y, lo, hi):
    if hi < lo:
        return 0
    if y < NUMPY_LIMIT and hi - lo + 1 >= 48:
        total = 0
        for start in range(lo, hi + 1, CHUNK):
            stop = min(hi, start + CHUNK - 1)
            total += int((y // np.arange(start, stop + 1, dtype=np.int64)).sum())
        return total
    return sum(y // x for x in range(lo, hi + 1))


def odd_range_floor_sum(y, lo, hi):
    if hi < lo:
        return 0
    return range_floor_sum(y, lo, hi) - range_floor_sum(y // 2, (lo + 1) // 2, hi // 2)


def odd_mobius_prefix(limit):
    mu = np.ones(limit + 1, dtype=np.int8)
    rest = np.arange(limit + 1, dtype=np.int32)
    root = isqrt(limit)
    composite = np.zeros(root + 1, dtype=bool)
    for p in range(2, root + 1):
        if composite[p]:
            continue
        composite[p * p::p] = True
        mu[p::p] *= -1
        rest[p::p] //= p
        if p * p <= limit:
            mu[p * p::p * p] = 0
    mu[rest > 1] *= -1
    mu[0::2] = 0
    mu[0] = 0
    return np.cumsum(mu, dtype=np.int32)

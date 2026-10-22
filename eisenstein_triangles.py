from math import isqrt

import numpy as np

from summation_tools import floor_sum, range_floor_sum


def mobius_prefix_skipping(limit, modulus):
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
    mu[0::modulus] = 0
    return np.cumsum(mu, dtype=np.int32)


def linear_floor_sum(p, q, lo, hi):
    if hi < lo:
        return 0
    return floor_sum(hi + 1, q, p, 0) - floor_sum(lo, q, p, 0)


def sum_of_integers(lo, hi):
    return (lo + hi) * (hi - lo + 1) // 2 if hi >= lo else 0


def half_sum_not_div3(lo, hi):
    c_lo, c_hi = (lo + 2) // 3, hi // 3
    return linear_floor_sum(1, 2, lo, hi) - sum_of_integers(c_lo, c_hi) - linear_floor_sum(1, 2, c_lo, c_hi)


def two_thirds_sum_not_div3(lo, hi):
    c_lo, c_hi = (lo + 2) // 3, hi // 3
    return linear_floor_sum(2, 3, lo, hi) - 2 * sum_of_integers(c_lo, c_hi)


def full_rows_sum(lo, hi):
    if hi < lo:
        return 0
    return two_thirds_sum_not_div3(lo, hi) - half_sum_not_div3(lo, hi)


def partial_rows_sum(w, lo, hi):
    if hi < lo:
        return 0
    quotients = range_floor_sum(w, lo, hi) - range_floor_sum(w // 3, (lo + 2) // 3, hi // 3)
    return quotients - half_sum_not_div3(lo, hi)


def last_true(predicate, lo, hi):
    if lo > hi or not predicate(lo):
        return lo - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if predicate(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def row_limits(w):
    top = isqrt(2 * w) + 2
    full = last_true(lambda y: 2 * y // 3 <= w // y, 1, top)
    widest = last_true(lambda y: w // y >= y // 2 + 1, 1, top)
    return full, widest


def count_pairs(z):
    if z < 15:
        return 0
    full, widest = row_limits(z)
    return full_rows_sum(1, full) + partial_rows_sum(z, full + 1, widest)


def weighted_pairs(z, tau_factor=1.0):
    if z < 15:
        return 0
    tau = max(2, int(tau_factor * round(z ** (1 / 3))))
    total = 0
    for y in range(2, tau + 1):
        if y % 3 == 0:
            continue
        lo, hi = y // 2 + 1, 2 * y // 3
        if hi >= lo:
            total += range_floor_sum(z // y, lo, hi)
    k = 1
    while True:
        w = z // k
        full, widest = row_limits(w)
        if widest <= tau:
            break
        total += full_rows_sum(tau + 1, full) + partial_rows_sum(w, max(full, tau) + 1, widest)
        k += 1
    return total


def pair_tables(limit):
    f = np.zeros(limit + 1, dtype=np.int64)
    y = 2
    while y * (y // 2 + 1) <= limit:
        if y % 3:
            x = np.arange(y // 2 + 1, 2 * y // 3 + 1, dtype=np.int64)
            x = x[x * y <= limit]
            np.add.at(f, x * y, 1)
        y += 1
    counts = np.cumsum(f)
    divisor_sums = np.zeros(limit + 1, dtype=np.int64)
    for c in np.nonzero(f)[0]:
        divisor_sums[c::c] += f[c]
    return counts, np.cumsum(divisor_sums)


def mobius_sum(n, function, tables, threshold):
    e_max = isqrt(n // 15)
    prefix = mobius_prefix_skipping(max(e_max, 2), 3)
    split = min(e_max, isqrt(n // max(threshold, 1)))
    total = 0
    for e in range(1, split + 1):
        m = int(prefix[e] - prefix[e - 1])
        if m:
            total += m * function(n // (e * e))
    if split < e_max:
        table = tables(max(n // ((split + 1) ** 2), 15))
        e = split + 1
        while e <= e_max:
            z = n // (e * e)
            e_hi = min(isqrt(n // z), e_max)
            d = int(prefix[e_hi] - prefix[e - 1])
            if d and z >= 15:
                total += d * int(table[z])
            e = e_hi + 1
    return total


def primitive_count(n, threshold=2_000_000):
    if n < 15:
        return 0
    return mobius_sum(n, count_pairs, lambda t: pair_tables(t)[0], threshold)


def all_triangles_count(n, threshold=20_000):
    if n < 15:
        return 0
    return mobius_sum(n, weighted_pairs, lambda t: pair_tables(t)[1], threshold)

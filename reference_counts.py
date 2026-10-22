from math import gcd, isqrt

import numpy as np


def count_by_sides(n):
    total = 0
    primitive = 0
    for a in range(1, n):
        if 3 * a + 1 > n and 2 * a + isqrt(3 * a * a) > n:
            break
        for b in range(a + 1, n):
            c2 = a * a + a * b + b * b
            c = isqrt(c2)
            if a + b + c > n:
                break
            if c * c == c2:
                total += 1
                if gcd(gcd(a, b), c) == 1:
                    primitive += 1
    return total, primitive


def count_by_parameters(n):
    total = 0
    primitive = 0
    for x in range(1, n + 1):
        for y in range(x + 1, 2 * x):
            if 2 * y > 3 * x and y % 3 and gcd(x, y) == 1 and x * y <= n:
                primitive += 1
                total += n // (x * y)
    return total, primitive


def count_vectorized(n):
    total = 0
    primitive = 0
    x = 1
    while x <= isqrt(n) + 2:
        y_lo = 3 * x // 2 + 1
        y_hi = min(2 * x - 1, n // x)
        if y_hi >= y_lo:
            y = np.arange(y_lo, y_hi + 1, dtype=np.int64)
            y = y[(y % 3 != 0) & (np.gcd(y, x) == 1)]
            primitive += y.size
            total += int((n // (x * y)).sum())
        x += 1
    return total, primitive

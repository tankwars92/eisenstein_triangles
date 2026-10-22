import argparse
import json
import platform
import random
import sys
import time
from math import log, pi

import numpy as np

import eisenstein_triangles as fast
import reference_counts as ref


def report(name, passed, results):
    results.append({"check": name, "passed": bool(passed)})
    print("  [%s] %s" % ("PASS" if passed else "FAIL", name), flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-primitive", type=int, default=15)
    parser.add_argument("--max-all", type=int, default=12)
    parser.add_argument("--compare-up-to", type=int, default=8)
    arguments = parser.parse_args()
    print("Python", sys.version.split()[0], platform.platform(), "numpy", np.__version__)
    results = []
    ok = True
    ok &= report("perimeters 15, 30, 50 give 1, 3, 5 triangles",
                 [fast.all_triangles_count(n) for n in (15, 30, 50)] == [1, 3, 5], results)
    ok &= report("N < 700 equals enumeration by sides",
                 all((fast.all_triangles_count(n), fast.primitive_count(n)) == ref.count_by_sides(n) for n in range(1, 700)), results)
    ok &= report("parametrized enumeration equals enumeration by sides, N < 300",
                 all(ref.count_by_parameters(n) == ref.count_by_sides(n) for n in range(1, 300)), results)
    values = [10 ** 5, 10 ** 6, 3 * 10 ** 6 + 1] + random.Random(5).sample(range(10 ** 5, 3 * 10 ** 7), 12)
    ok &= report("%d values up to 3e7 equal the vectorized baseline" % len(values),
                 all((fast.all_triangles_count(n), fast.primitive_count(n)) == ref.count_vectorized(n) for n in values), results)
    if not ok:
        sys.exit(1)
    constant = 9 * log(4 / 3) / (4 * pi ** 2)
    print("\nlimit of Q(N)/N: 9 ln(4/3) / (4 pi^2) = %.12f\n" % constant)
    rows = []
    print("%3s %20s %9s %20s %9s %12s" % ("n", "primitive(10^n)", "time,s", "all(10^n)", "time,s", "baseline,s"))
    for n in range(1, arguments.max_primitive + 1):
        row = {"n": n}
        start = time.perf_counter()
        row["primitive"] = fast.primitive_count(10 ** n)
        row["primitive_time"] = time.perf_counter() - start
        if n <= arguments.max_all:
            start = time.perf_counter()
            row["all"] = fast.all_triangles_count(10 ** n)
            row["all_time"] = time.perf_counter() - start
        if n <= arguments.compare_up_to:
            start = time.perf_counter()
            baseline = ref.count_vectorized(10 ** n)
            row["baseline_time"] = time.perf_counter() - start
            if baseline != (row["all"], row["primitive"]):
                print("MISMATCH at n = %d" % n)
                sys.exit(1)
        rows.append(row)
        print("%3d %20d %9.2f %20s %9s %12s" % (
            n, row["primitive"], row["primitive_time"],
            row.get("all", "-"), "%.2f" % row["all_time"] if "all_time" in row else "-",
            "%.2f" % row["baseline_time"] if "baseline_time" in row else "-"), flush=True)
    with open("results.json", "w") as handle:
        json.dump({"python": sys.version, "platform": platform.platform(), "checks": results, "rows": rows}, handle, indent=2)
    print("wrote results.json")


if __name__ == "__main__":
    main()

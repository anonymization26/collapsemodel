"""Verify the two-dimensional A-opt/MMD counterexample in exact arithmetic."""
from fractions import Fraction as F


def gram(rows):
    return [[sum(row[i] * row[j] for row in rows) for j in range(2)] for i in range(2)]


e1, e2 = (1, 0), (0, 1)
target = [e1] * 3 + [e2]
pools = {"A": [e1] * 6 + [e2] * 2, "B": [e1] * 5 + [e2] * 3}
target_gram = gram(target)
moment = [[F(value, len(target)) for value in row] for row in target_gram]
results = {}
for name, rows in pools.items():
    g = gram(rows)
    trace = g[0][0] + g[1][1]
    assert len(rows) == trace == 8 and g[0][1] == g[1][0] == 0
    mmd_squared = sum((F(g[i][j], len(rows)) - moment[i][j]) ** 2
                      for i in range(2) for j in range(2))
    variance = sum(moment[i][i] / (1 + g[i][i]) for i in range(2))
    results[name] = mmd_squared, variance
    print(f"{name}: n={len(rows)}, Gram={g}, trace={trace}, "
          f"MMD²={mmd_squared}, A-opt={variance} ({float(variance):.9f}), "
          f"predictive risk={1 + variance}")

assert results["A"] == (0, F(4, 21))
assert results["B"] == (F(1, 32), F(3, 16))
assert results["A"][0] < results["B"][0]
gap = results["A"][1] - results["B"][1]
assert gap == F(1, 336) > 0
print(f"PASS: MMD² prefers A, while A has higher predictive risk by {gap}.")

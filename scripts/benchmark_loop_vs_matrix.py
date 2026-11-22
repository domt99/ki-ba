"""4a/4d: time for one forward + backward step of 784-30-10 with one random input,
loop vs. matrix, extrapolated to one epoch with 60 000 instances.

Run: uv run python scripts/benchmark_loop_vs_matrix.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ki_ba.activation_function import ReLU

from tests.helpers import build_mlp

TOPOLOGY = [784, 30, 10]
INSTANCES_PER_EPOCH = 60_000


def time_step(calculation, repeats):
    rng = np.random.default_rng(0)
    mlp = build_mlp(TOPOLOGY, ReLU(), calculation)
    inputs = rng.random((1, TOPOLOGY[0]))
    targets = np.eye(TOPOLOGY[-1])[[rng.integers(TOPOLOGY[-1])]]

    start = time.perf_counter()
    for _ in range(repeats):
        mlp.forward_propagation(inputs)
        mlp.backward_propagation(targets)
    return (time.perf_counter() - start) / repeats


def main():
    results = {"loop": time_step("loop", repeats=20), "matrix": time_step("matrix", repeats=2000)}
    print(f"Forward + backward, {'-'.join(map(str, TOPOLOGY))}, one instance")
    for name, seconds in results.items():
        epoch = seconds * INSTANCES_PER_EPOCH
        print(f"  {name:<6} {seconds * 1000:8.3f} ms/step → {epoch:8.1f} s/epoch ({epoch / 60:.1f} min)")
    print(f"  matrix is {results['loop'] / results['matrix']:.0f}× faster")


if __name__ == "__main__":
    main()

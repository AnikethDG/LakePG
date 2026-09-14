"""Micro-benchmarks for the LakePG storage engine.

These are deliberately dependency-free and run under the standard library
``timeit`` so they can execute in CI without a benchmarking framework. They
report throughput, not wall-clock regressions -- CI runners are far too noisy
to gate on absolute numbers.

Run with::

    python -m benchmarks.bench_page
"""

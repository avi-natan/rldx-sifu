# WAY 1 — NOT USED ❌ (baseline only)

These way-1 results are the **Taxi-style ambiguous baseline** and are **NOT part of the benchmark
we use**. We use **way 2** (see `../BENCHMARK_WAYS.md`).

Why dropped: way 1 corrupts the *rarest* action and adds *near-twins*; on a slippery env their
outcomes overlap, so the true fault stays ambiguous **even at full observability**
(best top-1 ~29% at fault rate 0.8 + 100% vis, vs way2's 91.8%). Retained only to document this.

# WAY 2 — CHOSEN ✅ (the benchmark we use)

These way-2 results are the **FrozenLake fault-diagnosis benchmark we use going forward**.
Construction and evidence: `../BENCHMARK_WAYS.md`.

Distinguishable by design: a\* = most-used action; 10 candidates = healthy + all 3 redirects of the
3 most-used actions. Every candidate corrupts a visible action, so hardness comes from partial
observability, not ambiguity. Headline: fault rate 0.8 + 100% vis -> rank 1.13 / top-1 91.8%.

Way 1 is NOT used (see `../frozenlake_way1_known_fr_epsweep/DO_NOT_USE_way1.md`).

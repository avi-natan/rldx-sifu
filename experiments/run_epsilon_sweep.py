"""
Autonomous epsilon-sweep over the v2 hard-Taxi benchmark (classes 2,3,4).

Runs UNATTENDED to completion. Every diagnosed cell is appended to a CSV immediately
and each phase's analysis is logged, so partial results survive an interruption.

Plan (user's tasks 1-5):
  epsilons   = [0.035, 0.07, 0.1]
  visibility = [50, 80]
  faultrates = [0.3, 0.2, 0.1, 0.05, 0.02]
  Phase A : 10 seeds/class x 3 eps              -> analyze
  Phase B : +10 more seeds/class (=> 20/class)  -> analyze combined (reuses Phase A cells)
  Phase C : if a class shows  smaller-eps -> better avg rank, rerun 50 seeds of it x 3 eps
            (prefers the hardest class 2; reuses the 20 cells already computed)

Outputs: runs/epsilon_sweep/cells.csv  +  runs/epsilon_sweep/report.txt
"""
import os, io, csv, time, contextlib, traceback

EPSILONS = [0.035, 0.07, 0.1]
VIS = [50, 80]
FRATES = [0.3, 0.2, 0.1, 0.05, 0.02]
LOW_FR = (0.05, 0.02)               # where epsilon should matter most
OUT_DIR = "runs/epsilon_sweep"
RESULTS_CSV = os.path.join(OUT_DIR, "cells.csv")
REPORT_TXT = os.path.join(OUT_DIR, "report.txt")

os.makedirs(OUT_DIR, exist_ok=True)

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on sys.path (script lives in experiments/)

from hard_taxi_benchmark_v2 import build_benchmark, CLASSES
from p_pipeline import run_NON_DETERMINSTIC_single_experiment_PO

_done = set()      # (class, seed, eps, vis, fr) already run -> skip recompute
_rows = []         # all result dicts


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(REPORT_TXT, "a") as f:
        f.write(line + "\n")


def run_cell(inst, eps, vis, fr):
    main_seed, base, cid, a_star, E_str, cands = inst
    key = (cid, main_seed, eps, vis, fr)
    if key in _done:
        return
    rank, dropped = None, True
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            out = run_NON_DETERMINSTIC_single_experiment_PO(
                domain_name="Taxi_v4", ml_model_name="PPO", render_mode="rgb_array",
                max_exec_len=200, debug_print=False,
                execution_fault_mode_name=E_str, instance_seed=base,
                fault_probability=fr, percent_visible_states=vis,
                possible_fault_mode_names=cands, num_candidate_fault_modes=10,
                epsilon=eps, unknown_fault_rate=False, fault_rate_candidates=None,
                fixed_candidate_fault_modes=cands)
        if out is not None:
            rank, dropped = out.get("real_fault_rank"), False
    except Exception:
        rank, dropped = None, True
    _rows.append({"class": cid, "seed": main_seed, "a_star": a_star, "eps": eps,
                  "vis": vis, "fr": fr, "rank": rank, "dropped": dropped})
    _done.add(key)
    with open(RESULTS_CSV, "a", newline="") as f:
        csv.writer(f).writerow([cid, main_seed, a_star, eps, vis, fr,
                                "" if rank is None else rank, int(dropped)])


def run_instances(instances, epsilons, label):
    t0 = time.time()
    total = len(instances) * len(epsilons) * len(VIS) * len(FRATES)
    n = 0
    for inst in instances:
        for eps in epsilons:
            for vis in VIS:
                for fr in FRATES:
                    run_cell(inst, eps, vis, fr)
                    n += 1
                    if n % 60 == 0:
                        log(f"  {label}: {n}/{total} cells, {time.time()-t0:.0f}s elapsed")
    log(f"{label}: {total} cells done in {time.time()-t0:.0f}s")


def avg_rank(rows, cls, eps, frs=None):
    sel = [r for r in rows if r["class"] == cls and r["eps"] == eps
           and r["rank"] is not None and (frs is None or r["fr"] in frs)]
    if not sel:
        return None, 0
    return sum(r["rank"] for r in sel) / len(sel), len(sel)


def report(rows, classes, title):
    log("")
    log(f"=== {title}: avg rank by (class, epsilon) [lower=better] ===")
    for c in classes:
        parts = [f"class {c}:"]
        for eps in EPSILONS:
            a, n = avg_rank(rows, c, eps)
            parts.append(f"eps{eps}={a:.2f}(n{n})" if a is not None else f"eps{eps}=NA")
        log("  " + "  ".join(parts))
    log(f"--- {title}: same, restricted to low fault rates {LOW_FR} (where eps matters) ---")
    for c in classes:
        parts = [f"class {c}:"]
        for eps in EPSILONS:
            a, n = avg_rank(rows, c, eps, frs=LOW_FR)
            parts.append(f"eps{eps}={a:.2f}" if a is not None else f"eps{eps}=NA")
        log("  " + "  ".join(parts))


def qualifies(rows, c):
    a035, _ = avg_rank(rows, c, 0.035)
    a07, _ = avg_rank(rows, c, 0.07)
    a10, _ = avg_rank(rows, c, 0.1)
    if None in (a035, a07, a10):
        return None
    return (a10 - a035), (a035, a07, a10)   # positive improvement => smaller eps better


def main():
    open(RESULTS_CSV, "w").write("class,seed,a_star,eps,vis,fr,rank,dropped\n")
    open(REPORT_TXT, "w").write("")
    log("building benchmark: up to 50 seeds/class (classes 2,3,4)...")
    with contextlib.redirect_stdout(io.StringIO()):
        bench = build_benchmark(seeds_per_class=50)
    by_class = {c: [i for i in bench if i[2] == c] for c in CLASSES}
    log("built: " + ", ".join(f"class {c}={len(by_class[c])}" for c in CLASSES))

    # ---- Phase A: 10/class ----
    instA = [i for c in CLASSES for i in by_class[c][:10]]
    log(f"PHASE A: 10/class x {len(EPSILONS)}eps x {len(VIS)}vis x {len(FRATES)}fr "
        f"= {len(instA)*len(EPSILONS)*len(VIS)*len(FRATES)} cells")
    run_instances(instA, EPSILONS, "PhaseA(10/class)")
    report(_rows, CLASSES, "PHASE A (10/class)")

    # ---- Phase B: +10 more (=> 20/class) ----
    instB = [i for c in CLASSES for i in by_class[c][10:20]]
    log("PHASE B: +10/class -> 20/class")
    run_instances(instB, EPSILONS, "PhaseB(+10)")
    report(_rows, CLASSES, "PHASE A+B (20/class)")

    # ---- qualification (task 5 trigger) ----
    log("")
    log("=== qualification: does smaller epsilon give a better (lower) avg rank? ===")
    qual = []
    for c in CLASSES:
        q = qualifies(_rows, c)
        if q is None:
            continue
        imp, (a035, a07, a10) = q
        mono = a035 <= a07 <= a10
        log(f"  class {c}: eps0.035={a035:.2f} eps0.07={a07:.2f} eps0.1={a10:.2f} "
            f"| improvement(0.1-0.035)={imp:+.2f} monotone={mono}")
        if imp > 0.05:
            qual.append((imp, c))

    if not qual:
        log("No class shows smaller-eps -> better rank (>0.05). Skipping Phase C.")
    else:
        qual.sort(reverse=True)
        chosen = 2 if any(c == 2 for _, c in qual) else qual[0][1]   # prefer hardest class 2
        log(f"PHASE C: class {chosen} qualifies -> 50 seeds x 3 eps (reusing the 20 already run)")
        instC = by_class[chosen][:50]
        run_instances(instC, EPSILONS, f"PhaseC(50/class={chosen})")
        report([r for r in _rows if r["class"] == chosen], [chosen],
               f"PHASE C (50/class, class {chosen})")

    log("")
    log("ALL DONE. results -> runs/epsilon_sweep/cells.csv , report -> runs/epsilon_sweep/report.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL:\n" + traceback.format_exc())

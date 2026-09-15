"""Offline analysis of ufr (unknown-fault-rate) diagnosis: where can we cut Monte-Carlo compute
WITHOUT hurting the true-fault rank? Reads the stored `log_prob_total_per_fault_and_rate` grid from
every ufr run -- no cluster, no re-simulation. Produces:
  - true-fault winning-rate histogram (which rates ever matter)
  - top1-top2 log-likelihood margin (how much MC precision is actually needed per instance)
  - rate-pruning rank cost (full 10 rates vs coarse subsets vs coarse-to-fine)
  - candidate-elimination potential (# contenders within 5 logL of the top)

Run from repo root:
  ./.venv_domains/Scripts/python.exe experiments_scripts/analyze_ufr_speedup.py
"""
import glob, json, numpy as np, pandas as pd

RATES = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']

EXPERIMENTS = {
 'FrozenLake': ('experimental results/FrozenLake_v1/UNknown_fr_experiments/exper*/xlsx/*.xlsx', 'execution_fault'),
 'Taxi':       ('experimental results/Taxi_v4/UNknown_fr_experiments_eps_sweep/xlsx/*.xlsx', None),
 'Empty0.5':   ('experimental results/MiniGrid_Empty_16x16_v0/unknown/minigrid_bench_noise0_5_ufr/xlsx/*.xlsx', 'execution_fault'),
 'Cross0.5':   ('experimental results/MiniGrid_SimpleCrossing_S11N2_v0/unknown/minigrid_bench_noise0_5_ufr/xlsx/*.xlsx', 'execution_fault'),
 'Cross0.7':   ('experimental results/MiniGrid_SimpleCrossing_S11N2_v0/unknown/minigrid_bench_noise0_7_ufr/xlsx/*.xlsx', 'execution_fault'),
}

SUBSETS = {
 'full(10)': RATES,
 'no-extremes(7)': ['0.2','0.3','0.4','0.5','0.6','0.7','0.8'],
 'five':     ['0.1','0.3','0.5','0.7','0.9'],
 'four':     ['0.2','0.4','0.6','0.8'],
 'three':    ['0.2','0.5','0.8'],
}


def rank_of(scores, tf):
    st = scores[tf]
    return 1 + sum(1 for v in scores.values() if v > st)


def true_key(row, grid, truecol):
    """The true fault's grid key: execution_fault when present, else the fault whose best
    (max-over-rate) score matches the recorded real_fault_score."""
    if truecol and truecol in row and str(row[truecol]) in grid:
        return str(row[truecol])
    return min(grid.keys(), key=lambda f: abs(max(grid[f].values()) - row.real_fault_score))


def score_subset(grid, subset):
    return {f: max(d.get(r, -1e18) for r in subset) for f, d in grid.items()}


def coarse_to_fine(grid):
    """Stage 1: best rate among {.2,.4,.6,.8}; Stage 2: refine to the 2 neighbours of the winner.
    ~6 rate-evaluations per fault. Final score = max over (coarse union its neighbours)."""
    s1 = ['0.2','0.4','0.6','0.8']
    out = {}
    for f, d in grid.items():
        b = max(s1, key=lambda r: d.get(r, -1e18)); bi = RATES.index(b)
        nbrs = [RATES[j] for j in (bi-1, bi+1) if 0 <= j < len(RATES)]
        out[f] = max(d.get(r, -1e18) for r in (s1 + nbrs))
    return out


def analyze(name, patt, truecol):
    files = glob.glob(patt)
    if not files:
        print(f'\n### {name}: NO FILES'); return
    df = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
    win_rates, top_margin, contenders = [], [], []
    ranks = {k: [] for k in list(SUBSETS) + ['ctf(~6)']}
    unchanged = {k: 0 for k in list(SUBSETS) + ['ctf(~6)']}
    n = 0
    for _, row in df.iterrows():
        try:
            grid = json.loads(row.log_prob_total_per_fault_and_rate)
        except Exception:
            continue
        tf = true_key(row, grid, truecol)
        if tf not in grid:
            continue
        n += 1
        d = grid[tf]
        win_rates.append(float(max(RATES, key=lambda r: d.get(r, -1e18))))
        best = {f: max(v.values()) for f, v in grid.items()}
        top = max(best.values())
        vals = sorted(best.values(), reverse=True)
        top_margin.append(vals[0] - vals[1] if len(vals) > 1 else np.nan)
        contenders.append(sum(1 for v in best.values() if v >= top - 5))
        r_full = rank_of(score_subset(grid, RATES), tf)
        for k, sub in SUBSETS.items():
            r = rank_of(score_subset(grid, sub), tf); ranks[k].append(r)
            unchanged[k] += (r == r_full)
        rc = rank_of(coarse_to_fine(grid), tf); ranks['ctf(~6)'].append(rc)
        unchanged['ctf(~6)'] += (rc == r_full)

    wr = np.array(win_rates); tm = np.array(top_margin); tm = tm[np.isfinite(tm)]
    c = np.array(contenders)
    print(f'\n### {name}  (n={n}) ###')
    print('  winning-rate: median=%.2f mean=%.2f  histogram=%s' % (
        np.median(wr), wr.mean(), {r: int((np.abs(wr-float(r))<1e-6).sum()) for r in RATES}))
    print('  top1-top2 margin: median=%.2f  near-tie(<1)=%.0f%%  easy(>10)=%.0f%%' % (
        np.median(tm), 100*np.mean(tm<1), 100*np.mean(tm>10)))
    print('  contenders(<=5 of top): mean=%.1f median=%d  %%(<=3)=%.0f%%' % (
        c.mean(), int(np.median(c)), 100*np.mean(c<=3)))
    base = np.mean(ranks['full(10)'])
    print('  %-16s %8s %8s %10s' % ('rate-subset', 'avgRank', 'vs full', '%rank-kept'))
    for k in ['full(10)', 'no-extremes(7)', 'five', 'four', 'three', 'ctf(~6)']:
        a = np.mean(ranks[k])
        print('  %-16s %8.3f %+8.3f %9.0f%%' % (k, a, a-base, 100*unchanged[k]/n))


if __name__ == '__main__':
    for name, (patt, truecol) in EXPERIMENTS.items():
        analyze(name, patt, truecol)

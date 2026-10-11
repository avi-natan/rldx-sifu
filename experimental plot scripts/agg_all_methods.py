"""Full aggregation: brute-force vs v2b vs racing-1 vs racing-3, both domains.
One row per xlsx. Prints rank/sims tables and writes CSVs. Numbers come straight from the files."""
import openpyxl, glob, os, csv
from statistics import mean

REPO = r'C:\Users\User\Desktop\rldx_repo\rldx-sifu\bruteforce_unknown_eps_experiments'
OUTDIR = os.path.join(REPO, 'summary_tables')
NS = [50, 100, 200, 400, 800]

def agg(base, folder, simcol):
    fs = glob.glob(os.path.join(base, folder, 'xlsx', '*.xlsx'))
    R, S = [], []
    for f in fs:
        try:
            wb = openpyxl.load_workbook(f, read_only=True, data_only=True); ws = wb.active
            it = ws.iter_rows(values_only=True); h = list(next(it)); r = dict(zip(h, list(next(it)))); wb.close()
            if r.get('real_fault_rank') is not None: R.append(r['real_fault_rank'])
            if r.get(simcol) is not None: S.append(r[simcol])
        except Exception:
            pass
    return len(R), (mean(R) if R else None), (mean(S) if S else None)

for dom, base in [('FrozenLake_v1', os.path.join(REPO, 'FrozenLake_v1')),
                  ('Taxi_v4', os.path.join(REPO, 'Taxi_v4'))]:
    print(f'\n========== {dom} ==========')
    print(f'{"N":>4} | {"brute r/sims":>20} | {"v2b r/sims":>20} | {"racing1 r/sims":>20} | {"racing3 r/sims":>20}')
    rows = []
    for N in NS:
        nb, rb, sb = agg(base, f'fixedtries_{N}', 'total_simulations')
        nv, rv, sv = agg(base, f'v2b_N{N}', 'total_simulations')
        n1, r1, s1 = agg(base, f'racing_N{N}_c095', 'racing_total_traces')
        n3, r3, s3 = agg(base, f'racing3_N{N}_c095', 'racing_total_traces')
        def fmt(r, s, n):
            return f'{r:>7.3f} {s:>11.0f}' if r is not None else f'{"-":>7} {"("+str(n)+")":>11}'
        print(f'{N:>4} | {fmt(rb,sb,nb)} | {fmt(rv,sv,nv)} | {fmt(r1,s1,n1)} | {fmt(r3,s3,n3)}')
        rows.append({'N': N, 'brute_rank': rb, 'brute_sims': sb, 'v2b_rank': rv, 'v2b_sims': sv,
                     'racing1_rank': r1, 'racing1_sims': s1, 'racing3_rank': r3, 'racing3_sims': s3,
                     'n_brute': nb, 'n_v2b': nv, 'n_racing1': n1, 'n_racing3': n3})
    out = os.path.join(OUTDIR, f'table8_all_methods_{dom}.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'wrote {out}')
print('\nDONE')

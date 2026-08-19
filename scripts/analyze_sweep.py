"""Sliced analysis of runs/epsilon_sweep/cells.csv (tasks 1-4) + deep-dive candidate finder.
All ranks; lower = better. SE = std/sqrt(n) so we can judge significance.
"""
import csv, statistics as st

EPS = [0.035, 0.07, 0.1]
FR = [0.3, 0.2, 0.1, 0.05, 0.02]
VIS = [50, 80]

rows = []
with open("runs/epsilon_sweep/cells.csv") as f:
    for r in csv.DictReader(f):
        if r["rank"] == "":
            continue
        rows.append(dict(cls=int(r["class"]), seed=int(r["seed"]), a=int(r["a_star"]),
                         eps=float(r["eps"]), vis=int(r["vis"]), fr=float(r["fr"]),
                         rank=int(r["rank"])))


def stat(sel):
    if not sel:
        return None
    m = st.mean(sel); n = len(sel); se = (st.pstdev(sel) / n ** 0.5) if n > 1 else 0.0
    return m, n, se


def line(sel_by_eps):
    out = []
    for e in EPS:
        s = stat(sel_by_eps(e))
        out.append(f"eps{e}={s[0]:.2f}+/-{s[2]:.2f}(n{s[1]})" if s else f"eps{e}=NA")
    return "  ".join(out)


print("======================================================================")
print("TASK 1 - avg rank per epsilon (ALL data, pooled over class/vis/fr)")
print("======================================================================")
print("  " + line(lambda e: [r["rank"] for r in rows if r["eps"] == e]))

print("\n======================================================================")
print("TASK 2 - per epsilon, EACH fault rate alone (pooled class/vis)")
print("======================================================================")
for fr in FR:
    print(f"  fr={fr:<4} " + line(lambda e, fr=fr: [r["rank"] for r in rows if r["eps"] == e and r["fr"] == fr]))

print("\n======================================================================")
print("TASK 3 - per epsilon, ONLY the 2 smallest fault rates (0.05, 0.02)")
print("======================================================================")
print("  " + line(lambda e: [r["rank"] for r in rows if r["eps"] == e and r["fr"] in (0.05, 0.02)]))

print("\n======================================================================")
print("TASK 4 - per epsilon, EACH visibility alone (pooled class/fr)")
print("======================================================================")
for v in VIS:
    print(f"  vis={v} " + line(lambda e, v=v: [r["rank"] for r in rows if r["eps"] == e and r["vis"] == v]))
print("  --- visibility x low fault rates (0.05,0.02) ---")
for v in VIS:
    print(f"  vis={v} " + line(lambda e, v=v: [r["rank"] for r in rows if r["eps"] == e and r["vis"] == v and r["fr"] in (0.05, 0.02)]))

# ---- deep-dive candidate finder: class 2, vis 80, fr 0.02 (and 0.05) ----
def rank_of(cls, seed, vis, fr, eps):
    for r in rows:
        if r["cls"] == cls and r["seed"] == seed and r["vis"] == vis and r["fr"] == fr and r["eps"] == eps:
            return r["rank"]
    return None

for fr in (0.02, 0.05):
    print("\n======================================================================")
    print(f"DEEP-DIVE CANDIDATES - class 2, vis 80, fr {fr}: rank @ each epsilon")
    print("  (looking for: eps0.035 detects (rank1) but bigger eps don't)")
    print("======================================================================")
    seeds = sorted({r["seed"] for r in rows if r["cls"] == 2})
    print(f"  {'seed':>5}  {'a*':>3}  e0.035  e0.07  e0.10   pattern")
    for s in seeds:
        a = next((r["a"] for r in rows if r["cls"] == 2 and r["seed"] == s), None)
        r035 = rank_of(2, s, 80, fr, 0.035)
        r07 = rank_of(2, s, 80, fr, 0.07)
        r10 = rank_of(2, s, 80, fr, 0.1)
        flag = ""
        if r035 == 1 and (r07 != 1 or r10 != 1):
            flag = "<<< 0.035 detects, bigger eps weaker"
        elif r035 == 1 and r07 == 1 and r10 != 1:
            flag = "<<< 0.035&0.07 detect, 0.1 fails"
        elif r035 is not None and r10 is not None and r035 < r10:
            flag = "(0.035 better)"
        print(f"  {s:>5}  {a:>3}  {str(r035):>6} {str(r07):>6} {str(r10):>6}   {flag}")

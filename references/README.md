# References

Reference material for the RL fault-diagnosis project. Drop PDFs (papers, and later
cluster/HPC docs) into this folder; the text is extracted for reading (see note below).

## How to add a paper
1. Copy the PDF into this folder.
2. Add a row to the table below with a short note on its relevance.
3. Tell Claude the filename when you want it read.

> Note: the Read tool needs `pdftoppm` (poppler), which isn't installed here. Instead we
> extract text with `pypdf` into a sibling `.txt` (gitignored). To (re)extract:
> `python -c "from pypdf import PdfReader; open('X.txt','w',encoding='utf-8').write('\n'.join((p.extract_text() or '') for p in PdfReader('X.pdf').pages))"`

## Papers

| File | Title / Authors | Relevance |
|------|-----------------|-----------|
| `recent_paper_1.pdf` | **"What Went Wrong? Diagnosing Anomalies in RL Policy Executions"** (defines RLDX; algorithms **SIF** + **SIFU**) | **Core / definitely relevant.** Source of `SIF`, `SIFU`, and the `SIFU2…SIFU8` ablation variants in `p_diagnosers.py`. Defines the RLDX problem, fault modes, observation gaps & **conflicts**. **Explicitly assumes deterministic transitions** — the assumption Ahmad relaxes to stochastic. Lists "ranking methods for diagnoses" as future work — which Ahmad's likelihood ranking implements. |
| `recent_paper_2.pdf` | **"Diagnosing Non-Intermittent Anomalies in RL Policy Executions"** — Natan, Stern, Kalech (DX 2024) | **Relevant (earlier work).** The previous student's published paper; `W`/`SN` baseline diagnosers and the non-intermittent→intermittent fault framing. Already cited in `README.md`/`CLAUDE.md`. |

## Cluster / HPC docs (later)

| File | Topic | Notes |
|------|-------|-------|
| _(none yet)_ | | |

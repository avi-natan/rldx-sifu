"""Per-experiment plots for the FrozenLake way2 UNKNOWN-fault-rate experiments.

Each experiment (exper1_fr05 / exper2_fr03 / exper3_fr08) is a single epsilon (0.04)
and a single injected fault rate, run over 100 maps and split across many per-map-group
xlsx files. There is no epsilon axis here, so Taxi's 6-figure epsilon-sweep template does
not apply; within one experiment only VISIBILITY varies. This script therefore makes the
two meaningful figures per experiment:

  1. avg real-fault rank  vs visibility (%)
  2. avg diagnosis time   vs visibility (%)

Before plotting it UNIFIES each experiment's many per-map xlsx into one merged xlsx at the
experiment root (so the split is collapsed to a single file) and drops a MERGE_NOTE.txt
recording that the merge happened, by which script, from how many files. Each plots folder
also gets a PLOT_PROVENANCE.txt (standing rule; see CLAUDE.md).

Run from repo root:
  ./.venv_domains/Scripts/python.exe scripts/plot_frozenlake_unknown.py
"""
import os
import glob
import datetime

import pandas as pd

# reuse the exact aggregation + single-line styling used for the Taxi plots
from plot_epsilon_sweep import _agg, _line_plot, RANK_COL, TIME_COL, VIS_COL
from plot_provenance import write_plot_provenance

# (folder name, human label, filename fault-rate token)
EXPERIMENTS = [
    ("exper1_fr05_eps_004", "fr=0.5", "05"),
    ("exper2_fr03_eps_004", "fr=0.3", "03"),
    ("exper3_fr08_eps_004", "fr=0.8", "08"),
]


def load_folder(xlsx_dir):
    files = sorted(glob.glob(os.path.join(xlsx_dir, "*.xlsx")))
    if not files:
        raise SystemExit(f"No .xlsx files found in {xlsx_dir}")
    frames = [pd.read_excel(f) for f in files]
    print(f"  loaded {len(files)} files, {sum(len(f) for f in frames)} rows from {xlsx_dir}")
    return pd.concat(frames, ignore_index=True), files


def write_merge_note(exper_dir, script_path, source_dir, source_files, merged_path, n_rows):
    note_path = os.path.join(exper_dir, "MERGE_NOTE.txt")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "XLSX MERGE NOTE",
        "===============",
        f"Merged by  : {os.path.abspath(script_path)}",
        f"Generated  : {now}",
        f"Source dir : {os.path.abspath(source_dir)}",
        f"Merged     : {len(source_files)} per-map-group xlsx files -> 1 file, {n_rows} rows total",
        f"Output file: {os.path.abspath(merged_path)}",
        "",
        "The individual per-map-group files under xlsx/ are unchanged; this merged file is a",
        "single-file convenience copy of the same rows. Plots are built from these rows.",
        "",
        "Source files:",
    ]
    lines += [f"  - {os.path.basename(f)}" for f in source_files]
    lines.append("")
    with open(note_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  wrote merge note {note_path}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = os.path.join(repo_root, "experimental results", "FrozenLake_v1",
                        "UNknown_fr_experiments")
    script_path = os.path.abspath(__file__)

    for folder, label, fr_token in EXPERIMENTS:
        exper_dir = os.path.join(base, folder)
        xlsx_dir = os.path.join(exper_dir, "xlsx")
        print(f"\n=== {folder} ({label}) ===")
        df, source_files = load_folder(xlsx_dir)

        # 1) unify the many per-map-group xlsx into one merged file at the experiment root
        merged_path = os.path.join(exper_dir, f"{folder}_MERGED.xlsx")
        df.to_excel(merged_path, index=False)
        print(f"  merged -> {merged_path}")
        write_merge_note(exper_dir, script_path, xlsx_dir, source_files, merged_path, len(df))

        # 2) the two meaningful figures (visibility is the only within-experiment axis)
        plots_dir = os.path.join(exper_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        tag = f"fl_way2_unknown_fr{fr_token}"
        created = []

        xs, ys, se, ns = _agg(df, VIS_COL, RANK_COL)
        created.append(_line_plot(
            xs, ys, se, ns, "Visibility (% observed states)", "Avg real-fault rank",
            f"FrozenLake way2 (unknown {label}): rank vs visibility",
            os.path.join(plots_dir, f"{tag}_rank_vs_visibility.png")))

        xs, ys, se, ns = _agg(df, VIS_COL, TIME_COL)
        created.append(_line_plot(
            xs, ys, se, ns, "Visibility (% observed states)", "Avg diagnosis time (sec)",
            f"FrozenLake way2 (unknown {label}): time vs visibility",
            os.path.join(plots_dir, f"{tag}_time_vs_visibility.png")))

        write_plot_provenance(plots_dir, created, input_sources=[xlsx_dir, merged_path],
                              script_path=script_path)

    print("\nDone.")


if __name__ == "__main__":
    main()

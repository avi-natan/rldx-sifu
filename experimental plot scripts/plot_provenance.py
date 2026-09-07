"""Plot provenance marker.

Standing convention (see CLAUDE.md / project rule): every script that writes plots
must drop a `PLOT_PROVENANCE.txt` into each plots folder recording WHICH script
produced those plots, when, and from what input data. Anyone opening a plots folder
can then trace it straight back to the exact script.

Usage from any plotting script, after saving its figures:

    from plot_provenance import write_plot_provenance
    write_plot_provenance(out_dir, created_files, input_sources=[...])

`created_files` and `input_sources` may be absolute or relative paths; only their
basenames/paths are recorded. `script_path` is auto-detected from the caller if not
given, so a script normally just passes the folder and the list of plots it wrote.
"""
import os
import sys
import inspect
import datetime

PROVENANCE_FILENAME = "PLOT_PROVENANCE.txt"


def write_plot_provenance(plots_dir, created_files, input_sources=None,
                          script_path=None, filename=PROVENANCE_FILENAME):
    """Write a provenance file into `plots_dir`.

    plots_dir      folder the plots were written to (the file lands here).
    created_files  iterable of the plot paths just created (basenames are recorded).
    input_sources  optional iterable of the input data path(s) read (xlsx files/folders).
    script_path    full path of the producing script; auto-detected from the caller
                   if None (falls back to the calling frame's filename).
    filename       provenance filename (default PLOT_PROVENANCE.txt).

    Returns the full path of the provenance file written.
    """
    # Resolve the producing script's full path (caller's file unless explicitly given).
    if script_path is None:
        caller = inspect.stack()[1]
        script_path = caller.filename
    script_path = os.path.abspath(script_path)

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, filename)

    created = [os.path.basename(str(p)) for p in (created_files or [])]
    sources = [str(p) for p in (input_sources or [])]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "PLOT PROVENANCE",
        "===============",
        f"These plots were created by : {script_path}",
        f"Generated                   : {now}",
        f"Command                     : {' '.join(sys.argv)}",
        "",
        f"Plots created in this folder ({os.path.abspath(plots_dir)}):",
    ]
    lines += [f"  - {c}" for c in created] or ["  (none reported)"]
    if sources:
        lines += ["", "Input data read from:"]
        lines += [f"  - {s}" for s in sources]
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  wrote provenance {out_path}")
    return out_path


import os
import pandas as pd

# ----------------------------
# Helper: find row by (Solver, Strategy, n=64) + optional extra conditions
# ----------------------------
def pick_row(df, solver, strategy, n_target=64.0, extra_cond=None):
    if "Solver" not in df.columns or "Strategy" not in df.columns:
        return None

    mask = (df["Solver"] == solver) & (df["Strategy"] == strategy)

    # Apply extra condition(s), e.g., {"warm_up": -1}
    if extra_cond is not None:
        for k, v in extra_cond.items():
            if k in df.columns:
                mask = mask & (df[k] == v)

    sub = df[mask].copy()
    if sub.empty:
        return None

    # Try to locate the "n" column used by this method
    # Priority: num_chains (for Parallel*), solver_n (for SC/ASC/ESC), then n
    candidate_cols = [c for c in ["num_chains", "solver_n", "n"] if c in sub.columns]
    if candidate_cols:
        ncol = candidate_cols[0]
        # some cols are empty for certain solvers; fall back if all NaN
        for c in candidate_cols:
            if sub[c].notna().any():
                ncol = c
                break

        # filter by n=64 if possible
        try:
            sub2 = sub[sub[ncol].astype(float) == float(n_target)]
            if not sub2.empty:
                sub = sub2
        except Exception:
            pass

    # If multiple configs match (e.g., different prune_patience), take the first
    return sub.iloc[0]


def get_metrics(row):
    """Return (acc, seq_k, tok_k) from a selected row."""
    if row is None:
        return None, None, None
    acc = float(row["Acc"])
    tok = float(row["Cost"]) / 1000.0
    # If Seq_Cost missing, treat seq = total
    seq = float(row["Seq_Cost"]) / 1000.0 if "Seq_Cost" in row.index else tok
    return acc, seq, tok


def delta_sup(current, baseline):
    """
    For token metrics only. Baseline is Full method.
    Negative delta => fewer tokens (good).
    """
    if current is None or baseline is None or baseline == 0:
        return ""
    diff = (current - baseline) / baseline * 100.0
    if abs(diff) < 0.01:
        return ""
    # negative is better -> red (as you used before)
    color = "red" if diff < 0 else "green!80!black"
    sign = "+" if diff > 0 else ""
    return f"\\textsuperscript{{\\scriptsize(\\textcolor{{{color}}}{{{sign}{diff:.1f}\\%}})}}"


def fmt_cell_acc(x):
    return "--" if x is None else f"{x:.1f}"


def fmt_cell_tok(x):
    return "--" if x is None else f"{x:.1f}k"


# ----------------------------
# Main: build ablation table
# ----------------------------
def main():
    # ----- configure -----
    models = ["Qwen3-0.6B-128"]
    datasets = ["aime24", "aime25"]   # add more benchmarks if needed
    root_dir = "./"
    out_tex = "ablation_table.tex"

    # You define: Full + ablations mapping here
    # Now supports extra_cond (e.g., warm_up=-1)
    ABLATIONS = [
        # (latex_name, solver, strategy, extra_cond)
        ("\\textbf{\\method{} (Full)}",
         "GreedySolver", "ParallelESTPruningBurstMajorityOnActiveBranches", {"warm_up": 15}),

        ("\\quad w/o Probe-guided Pruning",
         "GreedySolver", "ParallelConvergenceProbeStrategy", None),

        ("\\quad w/o majority consensus early-stop",
         "GreedySolver", "ParallelESTPruningBurstMajorityOnActiveBranchesWOES", None),

        ("\\quad w/o warmup stage",
         "GreedySolver", "ParallelESTPruningBurstMajorityOnActiveBranches",
         {"warm_up": -1}),  # ⭐ this row

        ("\\quad w/o leveraging 2d probing information",
         "MajorityVoteSolver", "ConvergenceProbeStrategy", None),
    ]

    # ----- latex header -----
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{")
    lines.append(r"Ablation study of \method{} on four benchmarks.")
    lines.append(r"We report Accuracy, sequential tokens (SeqTok; lower is better), and total generated tokens (TotTok; lower is better).")
    lines.append(r"$\Delta$ reports the relative change compared to \method{} (negative means fewer tokens / lower cost).")
    lines.append(r"}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.12}")
    lines.append(r"\begin{adjustbox}{width=\textwidth}")

    # Column layout depends on number of datasets
    # Each dataset -> 3 cols, plus Avg -> 3 cols, plus first "Ablation" column
    col_spec = "l " + "ccc " * (len(datasets) + 1)
    lines.append(rf"\begin{{tabular}}{{{col_spec.strip()}}}")
    lines.append(r"\toprule")

    # Header row 1
    ds_titles = {"aime24": "AIME24", "aime25": "AIME25", "hmmt25": "HMMT25"}
    hdr1 = [r"\multirow{2}{*}{\textbf{Ablation}}"]
    for ds in datasets:
        hdr1.append(rf"\multicolumn{{3}}{{c}}{{\textbf{{{ds_titles.get(ds, ds.upper())}}}}}")
    hdr1.append(r"\multicolumn{3}{c}{\textbf{Avg.}}")
    lines.append(" & ".join(hdr1) + r" \\")
    # cmidrules
    start = 2
    cm = []
    for _ in datasets:
        cm.append(rf"\cmidrule(lr){{{start}-{start+2}}}")
        start += 3
    cm.append(rf"\cmidrule(lr){{{start}-{start+2}}}")
    lines.append("".join(cm))

    # Header row 2
    hdr2 = [""]
    for _ in datasets + ["avg"]:
        hdr2 += [r"\textbf{Acc.}", r"\textbf{SeqTok}", r"\textbf{TotTok}"]
    lines.append(" & ".join(hdr2) + r" \\")
    lines.append(r"\midrule")

    # ----- per model block -----
    for model in models:
        cand_dirs = [
            os.path.join(root_dir, f"matrix_results_{model}"),
            os.path.join(root_dir, f"matrix_results_{model}-128"),
            os.path.join(root_dir, f"matrix_results_{model}-Final"),
        ]
        base_dir = None
        for d in cand_dirs:
            if os.path.isdir(d):
                base_dir = d
                break

        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1+3*(len(datasets)+1)}}}{{l}}{{\textit{{Base Model: {model}}}}} \\")
        lines.append(r"\midrule")

        # First, collect Full metrics per dataset (for deltas)
        full_per_ds = {}
        for ds in datasets:
            fp = None if base_dir is None else os.path.join(base_dir, f"{ds}_raw.csv")
            if fp and os.path.exists(fp):
                df = pd.read_csv(fp)
                full_row = pick_row(
                    df,
                    ABLATIONS[0][1],
                    ABLATIONS[0][2],
                    n_target=64.0,
                    extra_cond=ABLATIONS[0][3],
                )
                full_per_ds[ds] = get_metrics(full_row)
            else:
                full_per_ds[ds] = (None, None, None)

        # Also compute Full avg baseline
        full_accs = [full_per_ds[ds][0] for ds in datasets if full_per_ds[ds][0] is not None]
        full_seqs = [full_per_ds[ds][1] for ds in datasets if full_per_ds[ds][1] is not None]
        full_toks = [full_per_ds[ds][2] for ds in datasets if full_per_ds[ds][2] is not None]
        full_avg = (
            sum(full_accs) / len(full_accs) if full_accs else None,
            sum(full_seqs) / len(full_seqs) if full_seqs else None,
            sum(full_toks) / len(full_toks) if full_toks else None,
        )

        # Now write each ablation row
        for latex_name, solver, strategy, extra_cond in ABLATIONS:
            acc_list, seq_list, tok_list = [], [], []
            row_cells = [latex_name]

            for ds in datasets:
                fp = None if base_dir is None else os.path.join(base_dir, f"{ds}_raw.csv")
                if fp and os.path.exists(fp):
                    df = pd.read_csv(fp)
                    r = pick_row(df, solver, strategy, n_target=64.0, extra_cond=extra_cond)
                    acc, seq, tok = get_metrics(r)
                else:
                    acc, seq, tok = None, None, None

                # per-dataset delta w.r.t. Full (only for token cols)
                f_acc, f_seq, f_tok = full_per_ds[ds]
                seq_delta = "" if latex_name.startswith("\\textbf") else delta_sup(seq, f_seq)
                tok_delta = "" if latex_name.startswith("\\textbf") else delta_sup(tok, f_tok)

                row_cells += [
                    fmt_cell_acc(acc),
                    (fmt_cell_tok(seq) + ("" if seq is None else f"{seq_delta}")),
                    (fmt_cell_tok(tok) + ("" if tok is None else f"{tok_delta}")),
                ]

                if acc is not None:
                    acc_list.append(acc)
                if seq is not None:
                    seq_list.append(seq)
                if tok is not None:
                    tok_list.append(tok)

            # Avg columns (delta w.r.t Full avg baseline)
            avg_acc = sum(acc_list) / len(acc_list) if acc_list else None
            avg_seq = sum(seq_list) / len(seq_list) if seq_list else None
            avg_tok = sum(tok_list) / len(tok_list) if tok_list else None

            avg_seq_delta = "" if latex_name.startswith("\\textbf") else delta_sup(avg_seq, full_avg[1])
            avg_tok_delta = "" if latex_name.startswith("\\textbf") else delta_sup(avg_tok, full_avg[2])

            row_cells += [
                fmt_cell_acc(avg_acc),
                (fmt_cell_tok(avg_seq) + ("" if avg_seq is None else f"{avg_seq_delta}")),
                (fmt_cell_tok(avg_tok) + ("" if avg_tok is None else f"{avg_tok_delta}")),
            ]

            lines.append(" & ".join(row_cells) + r" \\")

            # spacing after full row
            if latex_name.startswith("\\textbf{\\method{} (Full)}"):
                lines.append(r"\addlinespace[1.5pt]")

    # ----- latex footer -----
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table*}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Wrote: {out_tex}")


if __name__ == "__main__":
    main()

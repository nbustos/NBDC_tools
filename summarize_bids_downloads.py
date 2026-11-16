#!/usr/bin/env python3
"""
Summarize subjects/sessions downloaded from BIDS-style open datasets
(e.g. /nbdc/hbcd_2.1/hbcd) across rawdata and each derivative pipeline.

Expected layout per dataset root:
    <root>/rawdata/sub-*/ses-*
    <root>/derivatives/<pipeline>-<hash>/sub-*/ses-*
        e.g. freesurfer-0f306a2f, freesurfer-2afa9081, xcp_d-2afa9081+0ef9c88a

Outputs (in --outdir):
    subject_sessions_long.csv    one row per dataset/source/subject/session
    subject_summary.csv          one row per dataset/source/subject (n_sessions, session list)
    source_summary.csv           counts per source (rawdata, each derivative version)
    pipeline_summary.csv         counts per pipeline type (versions merged, e.g. all freesurfer-*)
    subject_coverage.csv         subject x source matrix (1 = present)
    (source/pipeline summaries also include n_expected_subjects and pct_subjects_downloaded,
     relative to the full release's participants.tsv -- see EXPECTED below)
    dataset_summary.csv          estimated total size (GB/TB) per dataset = sum over rawdata + all derivatives
    parameters.csv               one row per derivative source: [environment] version + all [workflow]
                                 parameters, read from sub-*/log/<run>/*.toml (e.g. xcp_d.toml)
    plots/*.png

Usage:
    python summarize_bids_downloads.py /nbdc/hbcd_2.1/hbcd -o hbcd_summary
    python summarize_bids_downloads.py /nbdc/hbcd_2.1/hbcd /nbdc/abcd_6.0/abcd -o all_summary
"""

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import tomllib  # Python >= 3.11
except ImportError:  # pragma: no cover
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ----------------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------------
def list_dirs(path, prefix):
    """Fast listing of subdirectories starting with prefix (works well on network FS)."""
    try:
        with os.scandir(path) as it:
            return sorted(e.name for e in it if e.name.startswith(prefix) and e.is_dir())
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        return []


def find_subject_root(path, max_depth=2):
    """Return the directory that directly contains sub-* folders (search a little deeper if nested)."""
    path = Path(path)
    if list_dirs(path, "sub-"):
        return path
    if max_depth == 0:
        return None
    for child in list_dirs(path, ""):
        found = find_subject_root(path / child, max_depth - 1)
        if found is not None:
            return found
    return None


def pipeline_type(source_name):
    """'freesurfer-0f306a2f' -> 'freesurfer'; 'xcp_d-2afa9081+0ef9c88a' -> 'xcp_d'; 'rawdata' -> 'rawdata'."""
    return re.split(r"-(?=[0-9a-f]{6,}|v?\d)", source_name, maxsplit=1)[0]


SUBJECT_ROOTS = {}  # (dataset, source) -> Path containing sub-* folders
EXPECTED = {}  # dataset -> (set of sub-* ids in the full release, basis string)


def load_expected_subjects(root):
    """
    Full-release subject list = participants.tsv (BIDS) at <root>/ or <root>/rawdata/.
    Returns a set of 'sub-XXXX' ids, or None if no participants file is found.
    """
    for cand in (Path(root) / "participants.tsv", Path(root) / "rawdata" / "participants.tsv"):
        if cand.is_file():
            try:
                df = pd.read_csv(cand, sep="\t", dtype=str)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] could not read {cand}: {e}", file=sys.stderr)
                continue
            col = "participant_id" if "participant_id" in df.columns else df.columns[0]
            ids = df[col].dropna().astype(str).str.strip()
            ids = ids.where(ids.str.startswith("sub-"), "sub-" + ids)
            return set(ids), f"participants.tsv ({cand.parent.name}/)"
    return None


def add_download_fraction(summary, subj_df, group_col):
    """Add n_expected_subjects / pct_subjects_downloaded / expected_basis to a per-source or per-pipeline summary."""
    exp_n, pct, basis = [], [], []
    for _, r in summary.iterrows():
        ds = r["dataset"]
        exp, how = EXPECTED.get(ds, (None, None))
        have = set(subj_df[(subj_df["dataset"] == ds) & (subj_df[group_col] == r[group_col])]["subject"])
        if exp:
            n_have, n_exp = len(have & exp), len(exp)
        else:  # no participants.tsv: fall back to everything seen in any source of this dataset
            exp = set(subj_df[subj_df["dataset"] == ds]["subject"])
            n_have, n_exp, how = len(have), len(exp), "union of downloaded subjects (no participants.tsv)"
        exp_n.append(n_exp)
        pct.append(round(100 * n_have / n_exp, 1) if n_exp else 0.0)
        basis.append(how)
    out = summary.copy()
    i = out.columns.get_loc("n_subjects") + 1
    out.insert(i, "n_expected_subjects", exp_n)
    out.insert(i + 1, "pct_subjects_downloaded", pct)
    out["expected_basis"] = basis
    return out


def dir_size_bytes(path):
    """Total size of files under path. Follows file symlinks (e.g. git-annex/datalad) but not dir symlinks."""
    total = 0
    stack = [str(path)]
    while stack:
        p = stack.pop()
        try:
            with os.scandir(p) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=True):
                            total += e.stat(follow_symlinks=True).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def estimate_sizes(sub_src, n_sample=5, seed=0):
    """
    Estimate total size per source by measuring a sample of subjects.
    Sampling is stratified by session count, and the estimate is
        sum over session-count groups k:  mean_size(k) * n_subjects(k)
    Groups that were not sampled fall back to (GB per session) * k.
    """
    rows = []
    for (ds, src), d in sub_src.groupby(["dataset", "source"]):
        root = SUBJECT_ROOTS.get((ds, src))
        if root is None:
            continue
        # stratified sample: round-robin across session-count groups
        groups = {k: g.sample(frac=1, random_state=seed)["subject"].tolist()
                  for k, g in d.groupby("n_sessions")}
        picked = []
        while len(picked) < min(n_sample, len(d)):
            for k in sorted(groups):
                if groups[k] and len(picked) < n_sample:
                    picked.append((groups[k].pop(), k))
        sizes = pd.DataFrame(
            [(s, k, dir_size_bytes(root / s) / 1e9) for s, k in picked],
            columns=["subject", "n_sessions", "gb"])

        gb_per_session = sizes["gb"].sum() / max(sizes["n_sessions"].clip(lower=1).sum(), 1)
        mean_by_k = sizes.groupby("n_sessions")["gb"].mean()
        counts = d["n_sessions"].value_counts()
        est = sum(n * mean_by_k.get(k, gb_per_session * max(k, 1)) for k, n in counts.items())
        rows.append(dict(
            dataset=ds, source=src,
            n_sampled=len(sizes),
            sampled_gb=round(sizes["gb"].sum(), 3),
            gb_per_session=round(gb_per_session, 3),
            gb_per_subject_min=round(sizes["gb"].min(), 3),
            gb_per_subject_max=round(sizes["gb"].max(), 3),
            est_total_gb=round(est, 1),
        ))
        print(f"  {src:<35} sampled {len(sizes)} subj -> ~{gb_per_session:.2f} GB/session, est {est:,.1f} GB")
    return pd.DataFrame(rows)


def _parse_toml_simple(text):
    """Minimal fallback TOML reader (sections + scalar/array values) for Pythons without tomllib."""
    import ast
    out, cur = {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = out.setdefault(line[1:-1].strip(), {})
        elif "=" in line and cur is not None:
            k, v = (x.strip() for x in line.split("=", 1))
            v = v.replace("true", "True").replace("false", "False")
            try:
                cur[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                cur[k] = v.strip('"')
    return out


def read_toml(path):
    text = Path(path).read_text()
    return tomllib.loads(text) if tomllib else _parse_toml_simple(text)


def find_log_toml(sub_dir):
    """Newest <sub>/log/<run>/*.toml, or None."""
    log = Path(sub_dir) / "log"
    for run in reversed(list_dirs(log, "")):
        try:
            tomls = sorted(f for f in os.listdir(log / run) if f.endswith(".toml"))
        except OSError:
            continue
        if tomls:
            return log / run / tomls[0]
    return None


def collect_parameters(sub_src, n_check=3):
    """
    For each derivative source, read the run-config TOML from the first subjects that have one and
    return one row: [environment] version + every [workflow] parameter.
    If the first n_check subjects disagree, the first is reported and 'params_consistent' is False.
    """
    rows = []
    for (ds, src), d in sub_src.groupby(["dataset", "source"]):
        root = SUBJECT_ROOTS.get((ds, src))
        if root is None or src == "rawdata":
            continue
        found = []
        for sub in d["subject"]:
            t = find_log_toml(root / sub)
            if t is not None:
                try:
                    cfg = read_toml(t)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] could not parse {t}: {e}", file=sys.stderr)
                    continue
                found.append((sub, t, cfg))
                if len(found) >= n_check:
                    break
        if not found:
            print(f"  {src:<35} no log/*.toml found")
            continue
        sub, t, cfg = found[0]
        wf = cfg.get("workflow", {})
        consistent = all(c.get("workflow", {}) == wf and
                         c.get("environment", {}).get("version") == cfg.get("environment", {}).get("version")
                         for _, _, c in found[1:])
        row = dict(dataset=ds, source=src, pipeline=pipeline_type(src),
                   env_version=cfg.get("environment", {}).get("version"),
                   params_consistent=consistent, config_file=t.name, example_subject=sub)
        row.update({k: str(v) if isinstance(v, (list, dict)) else v for k, v in wf.items()})
        rows.append(row)
        if not consistent:
            print(f"  [warn] {src}: parameters differ between sampled subjects; reporting {sub}")
    return pd.DataFrame(rows)


def scan_source(dataset, source, source_type, path):
    rows = []
    sub_root = find_subject_root(path)
    if sub_root is None:
        print(f"  [warn] no sub-* folders under {path}", file=sys.stderr)
        return rows
    SUBJECT_ROOTS[(dataset, source)] = sub_root
    subjects = list_dirs(sub_root, "sub-")
    for sub in subjects:
        sessions = list_dirs(sub_root / sub, "ses-")
        if not sessions:  # subject folder without session level
            sessions = [None]
        for ses in sessions:
            rows.append(
                dict(dataset=dataset, source=source, pipeline=source_type,
                     subject=sub, session=ses)
            )
    print(f"  {source:<35} {len(subjects):>6} subjects")
    return rows


def scan_dataset(root, include_rawdata=True):
    root = Path(root).resolve()
    # e.g. /nbdc/hbcd_2.1/hbcd -> "hbcd_2.1/hbcd"
    dataset =f"{root.parent.name}/{root.name}" if root.parent.name else root.name
    print(f"\nScanning {root}  (dataset = {dataset})")
    rows = []
    exp = load_expected_subjects(root)
    if exp:
        EXPECTED[dataset] = exp
        print(f"  full release: {len(exp[0])} subjects ({exp[1]})")
    else:
        print("  [warn] no participants.tsv found; % downloaded will be relative to subjects seen on disk",
              file=sys.stderr)
    if include_rawdata and (root / "rawdata").is_dir():
        rows += scan_source(dataset, "rawdata", "rawdata", root / "rawdata")
    deriv_dir = root / "derivatives"
    for d in list_dirs(deriv_dir, ""):
        rows += scan_source(dataset, d, pipeline_type(d), deriv_dir / d)
    return rows


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------
def subject_table(long_df, group_col):
    """One row per dataset/<group_col>/subject with session info."""
    g = long_df.groupby(["dataset", group_col, "subject"])["session"]
    out = g.agg(
        n_sessions=lambda s: s.dropna().nunique(),
        sessions=lambda s: ";".join(sorted(s.dropna().unique())),
    ).reset_index()
    return out


def count_summary(sub_df, long_df, group_col):
    rows = []
    for (ds, grp), d in sub_df.groupby(["dataset", group_col]):
        ns = d["n_sessions"]
        row = {
            "dataset": ds,
            group_col: grp,
            "n_subjects": len(d),
            "n_single_session": int((ns <= 1).sum()),
            "n_multi_session": int((ns >= 2).sum()),
            "pct_multi_session": round(100 * (ns >= 2).mean(), 1),
            "n_2_sessions": int((ns == 2).sum()),
            "n_3plus_sessions": int((ns >= 3).sum()),
            "total_subject_sessions": int(ns.sum()),
            "max_sessions": int(ns.max()) if len(ns) else 0,
        }
        # per-session-label subject counts (ses-V02, ses-V03, ...)
        ld = long_df[(long_df["dataset"] == ds) & (long_df[group_col] == grp)]
        for ses, n in ld.dropna(subset=["session"]).groupby("session")["subject"].nunique().items():
            row[f"n_subj_{ses}"] = int(n)
        rows.append(row)
    df = pd.DataFrame(rows)
    ses_cols = sorted(c for c in df.columns if c.startswith("n_subj_ses-"))
    base = [c for c in df.columns if c not in ses_cols]
    return df[base + ses_cols].fillna(0)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def _label(df, col):
    multi_ds = df["dataset"].nunique() > 1
    return (df["dataset"] + " | " + df[col]) if multi_ds else df[col]


def plot_subject_counts(summary, col, title, out):
    df = summary.sort_values(["dataset", col])
    labels = _label(df, col)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(df)), 5))
    ax.bar(labels, df["n_single_session"], label="1 session", color="#9ecae1")
    ax.bar(labels, df["n_multi_session"], bottom=df["n_single_session"],
           label="2+ sessions", color="#3182bd")
    for x, (tot, m, p) in enumerate(zip(df["n_subjects"], df["n_multi_session"], df["pct_multi_session"])):
        ax.text(x, tot, f"{tot}\n({m} multi, {p}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Subjects")
    ax.set_title(title)
    ax.set_ylim(0, df["n_subjects"].max() * 1.18)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sessions_distribution(sub_df, col, out):
    tab = (sub_df.assign(label=_label(sub_df, col))
           .pivot_table(index="label", columns="n_sessions", values="subject",
                        aggfunc="count", fill_value=0))
    ax = tab.plot(kind="bar", figsize=(max(6, 1.2 * len(tab)), 5), colormap="Blues", edgecolor="k")
    ax.set_ylabel("Subjects")
    ax.set_xlabel("")
    ax.set_title("Number of sessions per subject")
    ax.legend(title="# sessions")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_session_labels(long_df, col, out):
    d = long_df.dropna(subset=["session"]).assign(label=_label(long_df.dropna(subset=["session"]), col))
    tab = d.groupby(["label", "session"])["subject"].nunique().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(1.2 * tab.shape[1] + 4, 0.6 * tab.shape[0] + 2))
    im = ax.imshow(tab.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns)
    ax.set_yticks(range(tab.shape[0]), tab.index)
    vmax = tab.values.max() if tab.size else 0
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            ax.text(j, i, v, ha="center", va="center",
                    color="white" if v > 0.6 * vmax else "black", fontsize=9)
    ax.set_title("Subjects per session label")
    fig.colorbar(im, ax=ax, label="Subjects")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sizes(summary, out):
    df = summary.dropna(subset=["est_total_gb"]).sort_values(["dataset", "source"])
    labels = _label(df, "source")
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(df)), 5))
    ax.bar(labels, df["est_total_gb"], color="#31a354")
    for x, (gb, per) in enumerate(zip(df["est_total_gb"], df["gb_per_session"])):
        ax.text(x, gb, f"{gb:,.0f} GB\n(~{per:.2f} GB/ses)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Estimated size (GB)")
    ax.set_title("Estimated disk usage per source (from sampled subjects)")
    ax.set_ylim(0, df["est_total_gb"].max() * 1.18 or 1)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_coverage(coverage, out):
    """How many subjects have each combination of rawdata/derivatives (UpSet-style bars)."""
    combos = coverage.drop(columns=["dataset", "subject"]).astype(bool)
    keys = combos.apply(lambda r: " + ".join(c for c, v in r.items() if v) or "(none)", axis=1)
    counts = (pd.DataFrame({"dataset": coverage["dataset"], "combo": keys})
              .value_counts().reset_index(name="n").sort_values("n"))
    labels = _label(counts, "combo")
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(counts) + 1.5))
    ax.barh(labels, counts["n"], color="#3182bd")
    for y, n in enumerate(counts["n"]):
        ax.text(n, y, f" {n}", va="center", fontsize=8)
    ax.set_xlabel("Subjects")
    ax.set_title("Subject coverage across rawdata / derivatives")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="Dataset root(s) containing rawdata/ and/or derivatives/")
    ap.add_argument("-o", "--outdir", default="download_summary", help="Output directory")
    ap.add_argument("--no-rawdata", action="store_true", help="Only summarize derivatives")
    ap.add_argument("--size-sample", type=int, default=5,
                    help="Subjects to measure per source for the size estimate (0 = skip sizes)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for subject sampling")
    args = ap.parse_args()

    rows = []
    for r in args.roots:
        rows += scan_dataset(r, include_rawdata=not args.no_rawdata)
    if not rows:
        sys.exit("No subjects found. Check the paths.")

    long_df = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    # Per source (each derivative version separately)
    sub_src = subject_table(long_df, "source")
    src_summary = count_summary(sub_src, long_df, "source")
    src_summary.insert(2, "pipeline", src_summary["source"].map(pipeline_type))

    # Size estimate (sample a few subjects per source, scale by session counts)
    if args.size_sample > 0:
        print(f"\nEstimating sizes ({args.size_sample} subjects per source)...")
        sizes = estimate_sizes(sub_src, args.size_sample, args.seed)
        src_summary = src_summary.merge(sizes, on=["dataset", "source"], how="left")

    # Per pipeline type (freesurfer-* versions merged -> union of subjects/sessions)
    long_pipe = long_df.drop_duplicates(["dataset", "pipeline", "subject", "session"])
    sub_pipe = subject_table(long_pipe, "pipeline")
    pipe_summary = count_summary(sub_pipe, long_pipe, "pipeline")
    n_versions = long_df.groupby(["dataset", "pipeline"])["source"].nunique().rename("n_versions")
    pipe_summary = pipe_summary.merge(n_versions.reset_index(), on=["dataset", "pipeline"])
    if "est_total_gb" in src_summary:
        # disk usage of a pipeline type = sum over its versions (each version is stored separately)
        gb = src_summary.groupby(["dataset", "pipeline"])["est_total_gb"].sum().round(1).reset_index()
        pipe_summary = pipe_summary.merge(gb, on=["dataset", "pipeline"], how="left")

    # Fraction of the full release downloaded (subjects)
    src_summary = add_download_fraction(src_summary, sub_src, "source")
    pipe_summary = add_download_fraction(pipe_summary, sub_pipe, "pipeline")

    # Whole-dataset size estimate = sum over all sources (rawdata + each derivative version)
    ds_summary = None
    if "est_total_gb" in src_summary:
        ds_summary = (src_summary.groupby("dataset")
                      .agg(n_sources=("source", "nunique"), est_total_gb=("est_total_gb", "sum"))
                      .round(1).reset_index())
        ds_summary["est_total_tb"] = (ds_summary["est_total_gb"] / 1000).round(2)

    # Subject x source coverage
    coverage = (long_df.drop_duplicates(["dataset", "source", "subject"])
                .assign(present=1)
                .pivot_table(index=["dataset", "subject"], columns="source",
                             values="present", fill_value=0)
                .reset_index())
    coverage.columns.name = None

    # Pipeline versions + parameters from log/*.toml
    print("\nReading pipeline parameters from sub-*/log/*/*.toml ...")
    params = collect_parameters(sub_src)

    # CSVs
    long_df.to_csv(outdir / "subject_sessions_long.csv", index=False)
    sub_src.to_csv(outdir / "subject_summary.csv", index=False)
    src_summary.to_csv(outdir / "source_summary.csv", index=False)
    pipe_summary.to_csv(outdir / "pipeline_summary.csv", index=False)
    coverage.to_csv(outdir / "subject_coverage.csv", index=False)
    if ds_summary is not None:
        ds_summary.to_csv(outdir / "dataset_summary.csv", index=False)
    if not params.empty:
        params.to_csv(outdir / "parameters.csv", index=False)

    # Plots
    p = outdir / "plots"
    plot_subject_counts(src_summary, "source", "Subjects per source (single vs multi-session)",
                        p / "subjects_per_source.png")
    plot_subject_counts(pipe_summary, "pipeline", "Subjects per pipeline type (versions merged)",
                        p / "subjects_per_pipeline.png")
    plot_sessions_distribution(sub_src, "source", p / "sessions_per_subject.png")
    plot_session_labels(long_df, "source", p / "session_labels_heatmap.png")
    plot_coverage(coverage, p / "subject_coverage.png")
    if "est_total_gb" in src_summary:
        plot_sizes(src_summary, p / "estimated_size_per_source.png")

    pd.set_option("display.width", 200)
    print("\n=== Per source ===")
    cols = ["dataset", "source", "n_subjects", "n_expected_subjects", "pct_subjects_downloaded",
            "n_single_session", "n_multi_session", "pct_multi_session"]
    cols += [c for c in ["gb_per_session", "est_total_gb"] if c in src_summary]
    print(src_summary[cols].to_string(index=False))
    print("\n=== Per pipeline type ===")
    cols = ["dataset", "pipeline", "n_versions", "n_subjects", "n_expected_subjects",
            "pct_subjects_downloaded", "n_multi_session", "pct_multi_session"]
    cols += [c for c in ["est_total_gb"] if c in pipe_summary]
    print(pipe_summary[cols].to_string(index=False))
    if ds_summary is not None:
        print("\n=== Estimated total size per dataset (rawdata + all derivatives) ===")
        print(ds_summary.to_string(index=False))
        if len(ds_summary) > 1:
            print(f"All datasets: ~{ds_summary['est_total_gb'].sum():,.1f} GB "
                  f"({ds_summary['est_total_gb'].sum() / 1000:,.2f} TB)")
    if not params.empty:
        print("\n=== Pipeline versions & parameters ([environment] version, [workflow]) ===")
        meta = ["dataset", "pipeline", "source", "env_version", "params_consistent"]
        wf_cols = [c for c in params.columns if c not in meta + ["config_file", "example_subject"]]
        for _, r in params.iterrows():
            print(f"\n{r['source']}  (dataset {r['dataset']}, version {r['env_version']}, "
                  f"from {r['config_file']} in {r['example_subject']})")
            for c in wf_cols:
                if pd.notna(r[c]):
                    print(f"    {c:<22} {r[c]}")
    print(f"\nWrote CSVs and plots to {outdir.resolve()}")


if __name__ == "__main__":
    main()

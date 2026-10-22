#!/usr/bin/env python3
"""
Plot MPI scaling results from timings.csv (written by scaling_timer.hpp).

CSV columns: nranks,ndofs,section,calls,max_s,avg_s,min_s

Strong scaling (default): for each problem size (ndofs) run at 2+ rank
counts, plots time, speedup and parallel efficiency against ranks.

Weak scaling (--weak): plots time and efficiency against ranks for runs
with (roughly) the same DOFs per rank. Strong-scaling runs in the same CSV
are ignored automatically; use --dofs-per-rank to choose a specific series.

Repeated runs with the same (nranks, ndofs) are averaged.

By default reads timings.csv and saves PNGs to plots/scaling/.

Usage:
    python plot_scaling.py                       # strong scaling, all sections
    python plot_scaling.py --sections solve total
    python plot_scaling.py --weak
    python plot_scaling.py my_timings.csv --metric avg_s --outdir results --out run1
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # no display needed (works on cluster login nodes)
import matplotlib.pyplot as plt

class Tee:
    """Write to several streams at once (e.g. terminal + file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()

def load(csv_file, metric):
    """Return {(ndofs, section): {nranks: mean_time}} and the section order."""
    raw = defaultdict(lambda: defaultdict(list))
    section_order = []
    with open(csv_file, newline="") as f:
        for row in csv.DictReader(f):
            sec = row["section"]
            if sec not in section_order:
                section_order.append(sec)
            key = (int(row["ndofs"]), sec)
            raw[key][int(row["nranks"])].append(float(row[metric]))

    data = {}
    for key, by_ranks in raw.items():
        data[key] = {p: sum(ts) / len(ts) for p, ts in by_ranks.items()}
    return data, section_order


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'ranks':>7} {'time [s]':>11} {'speedup':>9} {'efficiency':>11}")
    for p, t, s, e in rows:
        print(f"{p:>7d} {t:>11.4f} {s:>9.2f} {e:>10.1%}")


def strong_scaling(data, sections, metric, out):
    sizes = sorted({nd for nd, _ in data})
    made_any = False

    for ndofs in sizes:
        secs = [s for s in sections if len(data.get((ndofs, s), {})) >= 2]
        if not secs:
            continue
        made_any = True

        fig, (ax_t, ax_s, ax_e) = plt.subplots(1, 3, figsize=(16, 4.8))
        all_ranks = set()

        for sec in secs:
            times = data[(ndofs, sec)]
            ranks = sorted(times)
            all_ranks.update(ranks)
            p0, t0 = ranks[0], times[ranks[0]]
            t = [times[p] for p in ranks]
            speedup = [t0 / times[p] for p in ranks]
            eff = [t0 / times[p] / (p / p0) for p in ranks]

            line, = ax_t.plot(ranks, t, "o-", label=sec)
            ax_t.plot(ranks, [t0 * p0 / p for p in ranks], "--",
                      color=line.get_color(), alpha=0.4)
            ax_s.plot(ranks, speedup, "o-", label=sec)
            ax_e.plot(ranks, eff, "o-", label=sec)

            print_table(f"[strong] ndofs={ndofs:,}  section={sec}",
                        zip(ranks, t, speedup, eff))

        r = sorted(all_ranks)
        ax_s.plot(r, [p / r[0] for p in r], "k--", alpha=0.5, label="ideal")
        ax_e.axhline(1.0, color="k", ls="--", alpha=0.5, label="ideal")

        ax_t.set(xscale="log", yscale="log", xlabel="MPI ranks",
                 ylabel=f"time [s] ({metric})",
                 title="Wall time (dashed = ideal)")
        ax_s.set(xscale="log", yscale="log", xlabel="MPI ranks",
                 ylabel=f"speedup vs {r[0]} ranks", title="Speedup")
        ax_e.set(xscale="log", xlabel="MPI ranks", ylabel="efficiency",
                 title="Parallel efficiency", ylim=(0, 1.2))

        for ax in (ax_t, ax_s, ax_e):
            ax.set_xticks(r)
            ax.set_xticklabels([str(p) for p in r])
            ax.minorticks_off()
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)

        fig.suptitle(f"Strong scaling, {ndofs:,} DOFs")
        fig.tight_layout()
        fname = f"{out}_strong_{ndofs}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"-> saved {fname}")

    if not made_any:
        print("No problem size has been run at 2+ rank counts yet; "
              "nothing to plot for strong scaling.")


def weak_series(data, sec, target_dpr, tol):
    """Pick one run per rank count whose DOFs/rank is within tol of target.

    If target_dpr is None, use the DOFs/rank value shared by the most
    rank counts (so strong-scaling runs in the same CSV are ignored).
    """
    pts = [(p, nd, t) for (nd, s), times in data.items() if s == sec
           for p, t in times.items()]
    if not pts:
        return [], None

    def matches(dpr, target):
        return abs(dpr - target) <= tol * target

    if target_dpr is None:
        best, best_n = None, 0
        for p0, nd0, _t0 in pts:
            cand = nd0 / p0
            n = len({p for p, nd2, _t in pts if matches(nd2 / p, cand)})
            if n > best_n:
                best, best_n = cand, n
        target_dpr = best

    chosen = {}
    for p, nd, t in pts:
        dpr = nd / p
        if matches(dpr, target_dpr):
            # if several runs match at one rank count, keep the closest
            if p not in chosen or abs(dpr - target_dpr) < abs(chosen[p][0] / p - target_dpr):
                chosen[p] = (nd, t)
    return sorted((p, nd, t) for p, (nd, t) in chosen.items()), target_dpr


def weak_scaling(data, sections, metric, out, target_dpr=None, tol=0.1):
    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(12, 4.8))
    all_ranks = set()
    used_dpr = None

    for sec in sections:
        pts, dpr = weak_series(data, sec, target_dpr, tol)
        if len(pts) < 2:
            continue
        used_dpr = used_dpr or dpr
        ranks = [p for p, _, _ in pts]
        t = [tt for _, _, tt in pts]
        all_ranks.update(ranks)
        eff = [t[0] / tt for tt in t]

        ax_t.plot(ranks, t, "o-", label=sec)
        ax_e.plot(ranks, eff, "o-", label=sec)
        if not all_ranks - set(ranks) and len(ax_t.lines) == 1:  # first series
            for p, nd, tt in pts:
                ax_t.annotate(f"{nd // p:,}/rank", (p, tt),
                              textcoords="offset points", xytext=(4, 6),
                              fontsize=7, alpha=0.7)

        print(f"\n[weak] section={sec}")
        print(f"{'ranks':>7} {'ndofs':>12} {'dofs/rank':>10} "
              f"{'time [s]':>11} {'efficiency':>11}")
        for (p, nd, tt), e in zip(pts, eff):
            print(f"{p:>7d} {nd:>12,} {nd // p:>10,} {tt:>11.4f} {e:>10.1%}")

    if not all_ranks:
        print("Not enough data for weak scaling.")
        plt.close(fig)
        return

    r = sorted(all_ranks)
    ax_e.axhline(1.0, color="k", ls="--", alpha=0.5, label="ideal")
    ax_t.set(xscale="log", xlabel="MPI ranks",
             ylabel=f"time [s] ({metric})", title="Wall time (ideal = flat)")
    ax_e.set(xscale="log", xlabel="MPI ranks", ylabel="efficiency  T1/Tn",
             title="Weak-scaling efficiency", ylim=(0, 1.2))
    for ax in (ax_t, ax_e):
        ax.set_xticks(r)
        ax.set_xticklabels([str(p) for p in r])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Weak scaling, ~{used_dpr:,.0f} DOFs/rank")
    fig.tight_layout()
    fname = f"{out}_weak.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"-> saved {fname}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="timings.csv")
    ap.add_argument("--sections", nargs="+",
                    help="sections to plot (default: all)")
    ap.add_argument("--metric", default="max_s",
                    choices=["max_s", "avg_s", "min_s"],
                    help="which timing column to use (default: max_s)")
    ap.add_argument("--weak", action="store_true",
                    help="weak-scaling plot instead of strong scaling")
    ap.add_argument("--dofs-per-rank", type=float, default=None,
                    help="weak scaling: DOFs/rank to plot (default: the "
                         "value shared by the most rank counts)")
    ap.add_argument("--tol", type=float, default=0.1,
                    help="weak scaling: relative tolerance on DOFs/rank "
                         "(default: 0.1 = 10%%)")
    ap.add_argument("--outdir", default="plots/scaling",
                    help="directory for the PNGs (default: plots/scaling)")
    ap.add_argument("--out", default="scaling",
                    help="output filename prefix (default: scaling)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"File not found: {args.csv}")

    os.makedirs(args.outdir, exist_ok=True)
    args.out = os.path.join(args.outdir, args.out)

    data, order = load(args.csv, args.metric)
    sections = args.sections or order
    missing = [s for s in sections if s not in order]
    if missing:
        sys.exit(f"Unknown section(s): {missing}. Available: {order}")

    mode = "weak" if args.weak else "strong"
    txt_file = f"{args.out}_{mode}.txt"

    with open(txt_file, "w") as f:
        sys.stdout = Tee(sys.__stdout__, f)
        try:
            if args.weak:
                weak_scaling(data, sections, args.metric, args.out,
                             args.dofs_per_rank, args.tol)
            else:
                strong_scaling(data, sections, args.metric, args.out)
        finally:
            sys.stdout = sys.__stdout__

    print(f"-> saved {txt_file}")



if __name__ == "__main__":
    main()
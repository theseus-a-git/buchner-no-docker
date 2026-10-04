#!/usr/bin/env python3
"""Progress of the benchmark from the .done markers; with --sacct also Slurm's view of every submitted task.
Python >= 3.6, standard library only."""
import argparse
import collections
import glob
import os
import re
import subprocess
import sys

ROOT = os.environ.get("INFENV_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLURM = os.path.join(ROOT, "slurm")


def run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, **kw)


def have(cmd):
    return any(os.access(os.path.join(p, cmd), os.X_OK) for p in os.environ.get("PATH", "").split(os.pathsep))


def rss_mb(s):
    m = re.match(r"^([0-9.]+)([KMGT]?)$", s.strip())
    return 0 if not m else float(m.group(1)) * {"": 1 / 1024.0, "K": 1 / 1024.0, "M": 1, "G": 1024, "T": 1048576}[m.group(2)]


def secs(s):
    m = re.match(r"^(?:(\d+)-)?(\d+):(\d+):(\d+)", s)
    return 0 if not m else int(m.group(1) or 0) * 86400 + int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems", nargs="+", default=["real", "synthetic"])
    ap.add_argument("--samplers", nargs="+", default=os.environ.get("IE_SAMPLERS", "ultranest dynesty nestle emcee pso").split())
    ap.add_argument("--sacct", action="store_true", help="add Slurm accounting (state, elapsed, peak memory) per submitted task")
    ap.add_argument("--all", action="store_true", help="with --sacct: also list COMPLETED tasks")
    ap.add_argument("--since", default="", help="with --sacct: only jobs started after this date, e.g. 2026-10-05")
    a = ap.parse_args()

    mm = run(["bash", os.path.join(SLURM, "make_manifest.sh")] + a.problems, env=dict(os.environ, IE_SAMPLERS=" ".join(a.samplers)))
    tasks = [tuple(l.split("\t")) for l in mm.stdout.splitlines() if l.strip()]
    state = {}
    for d, c, s in tasks:
        base = os.path.join(ROOT, d, "runs", c, s)
        state[(d, c, s)] = "done" if os.path.exists(os.path.join(base, ".done")) else ("part" if os.path.exists(os.path.join(base, "log.txt")) else "-")

    cases = collections.OrderedDict()
    for d, c, s in tasks:
        cases.setdefault((d, c), {})[s] = state[(d, c, s)]
    w = max([len(d + ":" + c) for d, c in cases] + [10])
    print("%-*s  %s" % (w, "problem:case", "  ".join("%-9s" % s for s in a.samplers)))
    for (d, c), row in cases.items():
        print("%-*s  %s" % (w, d + ":" + c, "  ".join("%-9s" % row.get(s, "") for s in a.samplers)))
    print("\ndone = finished OK;  part = started (running, failed or timed out: see runs/<case>/<sampler>/log.txt);  - = not started")
    n_done = sum(1 for v in state.values() if v == "done")
    print("TOTAL: %d / %d done, %d started but not done, %d not started" % (
        n_done, len(state), sum(1 for v in state.values() if v == "part"), sum(1 for v in state.values() if v == "-")))
    for s in a.samplers:
        print("  %-10s %3d / %d" % (s, sum(1 for (d, c, ss), v in state.items() if ss == s and v == "done"), sum(1 for k in state if k[2] == s)))

    if have("squeue"):
        out = run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%j %T"]).stdout.split("\n")
        cnt = collections.Counter(l.split()[1] for l in out if l.startswith("ie-") and len(l.split()) == 2)
        if cnt:
            print("\nin the queue now: " + ", ".join("%d %s" % (v, k) for k, v in sorted(cnt.items())))

    if not a.sacct:
        return
    if not have("sacct"):
        sys.exit("\nsacct not found (run this on the login node)")
    ledger = {}
    for f in sorted(glob.glob(os.path.join(SLURM, "jobs", "*", "ledger.tsv"))):
        for line in open(f):
            p = line.rstrip("\n").split("\t")
            if len(p) == 6:
                ledger[p[0]] = p[1:]                                  # later submissions win
    if not ledger:
        sys.exit("\nno submissions recorded in slurm/jobs/*/ledger.tsv yet")
    jobids = sorted({k.split("_")[0] for k in ledger})
    cmd = ["sacct", "-n", "-P", "-j", ",".join(jobids), "--format=JobID,State,Elapsed,Timelimit,MaxRSS,ExitCode"]
    if a.since:
        cmd += ["-S", a.since]
    r = run(cmd)
    if r.returncode != 0:
        sys.exit("\nsacct failed: " + r.stderr.strip())
    info = {}
    for line in r.stdout.splitlines():
        f = line.split("|")
        if len(f) < 6:
            continue
        jid = f[0]
        if re.match(r"^\d+_\d+$", jid):
            info.setdefault(jid, {}).update(state=f[1].split()[0], elapsed=f[2], limit=f[3], exit=f[5])
        elif re.match(r"^\d+_\d+\.(batch|0)$", jid) and f[4]:
            d = info.setdefault(jid.split(".")[0], {})
            d["rss"] = max(d.get("rss", 0), rss_mb(f[4]))
    bad = []
    stats = collections.defaultdict(lambda: [0, 0, 0.0])               # dir -> n completed, max elapsed s, max rss MB
    for jid, (d, c, s, lim, mem) in ledger.items():
        i = info.get(jid)
        if not i:
            continue
        if i["state"] == "COMPLETED":
            st = stats[d]
            st[0] += 1
            st[1] = max(st[1], secs(i["elapsed"]))
            st[2] = max(st[2], i.get("rss", 0))
        if a.all or i["state"] != "COMPLETED":
            bad.append((d, c, s, i["state"], i["elapsed"], i["limit"], "%.0fM" % i.get("rss", 0), mem, i["exit"]))
    if bad:
        print("\nSlurm tasks (not COMPLETED unless --all):")
        print("%-24s %-22s %-10s %-14s %-11s %-11s %-8s %-7s %s" % ("problem", "case", "sampler", "state", "elapsed", "limit", "maxrss", "req", "exit"))
        for b in sorted(bad):
            print("%-24s %-22s %-10s %-14s %-11s %-11s %-8s %-7s %s" % b)
        if any(b[3] == "TIMEOUT" for b in bad):
            print("  TIMEOUT -> slurm/submit.sh --time-mult 2   (or lower max_ncalls for that problem in resources.tsv)")
        if any(b[3] == "OUT_OF_MEMORY" for b in bad):
            print("  OUT_OF_MEMORY -> slurm/submit.sh --mem-mult 2   (or raise mem in resources.tsv)")
    if stats:
        print("\nwhat finished runs actually used (use this to tune resources.tsv):")
        print("%-28s %9s %12s %10s" % ("problem", "completed", "max elapsed", "max rss"))
        for d, (n, el, rs) in sorted(stats.items()):
            print("%-28s %9d %12s %9.0fM" % (d, n, "%d:%02d:%02d" % (el // 3600, el % 3600 // 60, el % 60), rs))


if __name__ == "__main__":
    main()

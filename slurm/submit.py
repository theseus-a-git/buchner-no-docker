#!/usr/bin/env python3
"""Submit (problem, case, sampler) runs to Slurm as job arrays: one array task = one single-core run.

Runs are grouped by their row in resources.tsv, so every group gets its own time/memory request.  Groups whose
problem directory is marked mutex (shared work files) become arrays limited to 1 concurrent task.  Finished runs
(runs/<case>/<sampler>/.done) are skipped, so calling this again only submits what is still missing.
Python >= 3.6, standard library only.
"""
import argparse
import collections
import datetime
import fnmatch
import os
import re
import subprocess
import sys

ROOT = os.environ.get("INFENV_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLURM = os.path.join(ROOT, "slurm")
Res = collections.namedtuple("Res", "pattern time mem cpus partition mutex net max_ncalls")


def parse_time(s):
    s = s.strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
        p = [int(x) for x in s.split(":")]
        h, m, sec = p[0], (p[1] if len(p) > 1 else 0), (p[2] if len(p) > 2 else 0)
    else:
        p = [int(x) for x in s.split(":")]
        if len(p) == 1:
            h, m, sec = 0, p[0], 0
        elif len(p) == 2:
            h, m, sec = 0, p[0], p[1]
        else:
            h, m, sec = p
    return days * 86400 + h * 3600 + m * 60 + sec


def fmt_time(sec):
    d, r = divmod(int(sec), 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return "%d-%02d:%02d:%02d" % (d, h, m, s)


def parse_mem_mb(s):
    m = re.match(r"^\s*([0-9.]+)\s*([KMGT]?)B?\s*$", s, re.I)
    if not m:
        raise ValueError("bad memory value %r" % s)
    return int(float(m.group(1)) * {"": 1, "K": 1 / 1024.0, "M": 1, "G": 1024, "T": 1024 * 1024}[m.group(2).upper()])


def load_resources(path):
    rows = []
    for ln, line in enumerate(open(path), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        f = line.split()
        if len(f) != 8:
            sys.exit("%s:%d: expected 8 columns, got %d: %s" % (path, ln, len(f), line))
        rows.append(Res(f[0], f[1], f[2], int(f[3]), f[4], f[5] == "1", f[6] == "1", f[7]))
    if not rows or rows[-1].pattern != "*":
        sys.exit("%s: the last row must be the catch-all '*'" % path)
    return rows


def lookup(rows, d):
    for r in rows:
        if fnmatch.fnmatchcase(d, r.pattern):
            return r


def run(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, **kw)


def have(cmd):
    return any(os.access(os.path.join(p, cmd), os.X_OK) for p in os.environ.get("PATH", "").split(os.pathsep))


def max_array_size():
    if have("scontrol"):
        m = re.search(r"MaxArraySize\s*=\s*(\d+)", run(["scontrol", "show", "config"]).stdout)
        if m:
            return int(m.group(1))
    return 1001


def active_ie_jobs():
    if not have("squeue"):
        return []
    out = run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%j"]).stdout.split()
    return [j for j in out if j.startswith("ie-")]


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-") or "default"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="+", default=["real", "synthetic"], metavar="DIR",
                    help="problem directories or prefixes (default: real synthetic)")
    ap.add_argument("--samplers", nargs="+", default=os.environ.get("IE_SAMPLERS", "ultranest dynesty nestle emcee pso").split())
    ap.add_argument("--cases", metavar="REGEX", help="only cases matching this regular expression")
    ap.add_argument("--all", action="store_true", help="also resubmit runs that are already done (adds FORCE=1)")
    ap.add_argument("--no-net", action="store_true", help="leave out problems that need internet at run time (resources.tsv: net=1)")
    ap.add_argument("--only-net", action="store_true", help="only the problems that need internet at run time")
    ap.add_argument("--max-ncalls", type=int, help="likelihood budget for every run (default: $MAX_NCALLS or 100000; per-problem value in resources.tsv wins)")
    ap.add_argument("--time-mult", type=float, default=1.0, help="multiply every wall limit (e.g. 2 to retry timeouts)")
    ap.add_argument("--mem-mult", type=float, default=1.0, help="multiply every memory request")
    ap.add_argument("--partition", default=os.environ.get("IE_PARTITION", "cpu"))
    ap.add_argument("--throttle", type=int, default=int(os.environ.get("IE_THROTTLE", "48")), help="max concurrent tasks per array")
    ap.add_argument("--resources", default=os.path.join(SLURM, "resources.tsv"))
    ap.add_argument("--no-compare", action="store_true", help="do not queue the final comparison job")
    ap.add_argument("--allow-active", action="store_true", help="submit even though ie-* jobs are already queued/running")
    ap.add_argument("--dry-run", "-n", action="store_true", help="show the plan, submit nothing, write nothing")
    a = ap.parse_args()

    rows = load_resources(a.resources)
    env = dict(os.environ, IE_SAMPLERS=" ".join(a.samplers))
    mm = run(["bash", os.path.join(SLURM, "make_manifest.sh")] + a.problems, env=env)
    if mm.returncode != 0:
        sys.exit(mm.stderr.strip() or "make_manifest.sh failed")
    for line in mm.stderr.strip().splitlines():
        print(line)
    tasks = [tuple(l.split("\t")) for l in mm.stdout.splitlines() if l.strip()]
    total = len(tasks)
    if a.cases:
        rx = re.compile(a.cases)
        tasks = [t for t in tasks if rx.search(t[1])]
    if a.no_net:
        tasks = [t for t in tasks if not lookup(rows, t[0]).net]
    if a.only_net:
        tasks = [t for t in tasks if lookup(rows, t[0]).net]
    selected = len(tasks)
    if not a.all:
        tasks = [t for t in tasks if not os.path.exists(os.path.join(ROOT, t[0], "runs", t[1], t[2], ".done"))]
    print("%d runs in the manifest, %d selected, %d already done, %d to submit" % (total, selected, selected - len(tasks), len(tasks)))
    if not tasks:
        return

    busy = active_ie_jobs()
    if busy and not a.allow_active and not a.dry_run:
        sys.exit("%d ie-* job(s) are still queued/running (%s ...). Wait for them, or use --allow-active "
                 "(runs that are still queued would then be submitted twice)." % (len(busy), busy[0]))

    max_wall = parse_time(os.environ.get("IE_MAX_TIME", "4-00:00:00"))
    default_ncalls = a.max_ncalls or int(os.environ.get("MAX_NCALLS", "100000"))
    groups = collections.OrderedDict()
    for t in tasks:
        r = lookup(rows, t[0])
        key = (r.pattern, t[0]) if r.mutex else (r.pattern, None)
        groups.setdefault(key, (r, []))[1].append(t)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    jobdir = os.path.join(SLURM, "jobs", stamp)
    maxarr = max_array_size()
    chunk = max(1, min(maxarr - 1, 1000))
    plan = []
    for (pattern, d), (r, ts) in groups.items():
        wall = parse_time(r.time) * a.time_mult
        clamped = wall > max_wall
        wall = min(wall, max_wall)
        mem = int(parse_mem_mb(r.mem) * a.mem_mult)
        name = "ie-" + slug(d if d else pattern)
        part = a.partition if r.partition == "-" else r.partition
        ncalls = default_ncalls if r.max_ncalls == "-" else int(r.max_ncalls)
        thr = 1 if r.mutex else a.throttle
        for ci in range(0, len(ts), chunk):
            sub = ts[ci:ci + chunk]
            plan.append(dict(name=name, tasks=sub, wall=wall, clamped=clamped, mem=mem, cpus=r.cpus, part=part,
                             ncalls=ncalls, thr=thr, mutex=r.mutex, net=r.net,
                             file="%s%s.tsv" % (name, "" if len(ts) <= chunk else "_c%d" % (ci // chunk + 1))))

    hdr = "%-34s %6s %-12s %6s %-9s %5s %4s %s" % ("array job", "tasks", "time/task", "mem", "partition", "limit", "net", "note")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    core_s = 0
    for p in plan:
        n = len(p["tasks"])
        core_s += n * p["wall"] * p["cpus"]
        print("%-34s %6d %-12s %5dM %-9s %5s %4s %s" % (p["name"], n, fmt_time(p["wall"]), p["mem"], p["part"],
              "1" if p["thr"] == 1 else str(min(p["thr"], n)), "yes" if p["net"] else "-",
              ("serial (shared work files); " if p["mutex"] else "") + ("time clamped to IE_MAX_TIME" if p["clamped"] else "")))
    print("\nworst case if every run uses its whole limit: %.0f core-hours (the real figure is usually far lower)" % (core_s / 3600.0))
    if a.dry_run:
        print("dry run: nothing submitted")
        return
    if not have("sbatch"):
        sys.exit("sbatch not found: run this on the PARAM Seva login node")

    os.makedirs(os.path.join(SLURM, "logs"), exist_ok=True)
    os.makedirs(jobdir, exist_ok=True)
    ledger = open(os.path.join(jobdir, "ledger.tsv"), "a")
    ids = []
    for p in plan:
        tf = os.path.join(jobdir, p["file"])
        with open(tf, "w") as f:
            for (d, c, s) in p["tasks"]:
                f.write("%s\t%s\t%s\t%d\n" % (d, c, s, p["ncalls"]))
        n = len(p["tasks"])
        spec = "1-%d" % n + ("%%%d" % p["thr"] if p["thr"] < n else "")
        cmd = ["sbatch", "--parsable", "--job-name=" + p["name"], "--array=" + spec, "--partition=" + p["part"],
               "--nodes=1", "--ntasks=1", "--cpus-per-task=%d" % p["cpus"], "--time=" + fmt_time(p["wall"]),
               "--mem=%dM" % p["mem"], "--requeue", "--output=%s/logs/%%x_%%A_%%a.out" % SLURM,
               "--export=ALL,INFENV_ROOT=%s,IE_TASKFILE=%s,IE_FORCE=%d" % (ROOT, tf, 1 if a.all else 0)]
        if os.environ.get("IE_ACCOUNT"):
            cmd.append("--account=" + os.environ["IE_ACCOUNT"])
        if os.environ.get("IE_QOS"):
            cmd.append("--qos=" + os.environ["IE_QOS"])
        if os.environ.get("IE_MAIL"):
            cmd += ["--mail-user=" + os.environ["IE_MAIL"], "--mail-type=END,FAIL"]
        cmd.append(os.path.join(SLURM, "task.sbatch"))
        r = run(cmd)
        if r.returncode != 0:
            ledger.close()
            sys.exit("sbatch failed for %s:\n%s\nalready submitted: %s (their task files are in %s)" %
                     (p["name"], (r.stderr or r.stdout).strip(), ",".join(ids) or "none", jobdir))
        jid = r.stdout.strip().split(";")[0]
        ids.append(jid)
        for i, (d, c, s) in enumerate(p["tasks"], 1):
            ledger.write("%s_%d\t%s\t%s\t%s\t%s\t%dM\n" % (jid, i, d, c, s, fmt_time(p["wall"]), p["mem"]))
        print("submitted %-34s job %s  (%d tasks, array %s)" % (p["name"], jid, n, spec))
    ledger.close()

    if not a.no_compare:
        cmd = ["sbatch", "--parsable", "--job-name=ie-compare", "--dependency=afterany:" + ":".join(ids),
               "--partition=" + a.partition, "--nodes=1", "--ntasks=1", "--time=01:00:00", "--mem=8G",
               "--output=%s/logs/%%x_%%j.out" % SLURM, "--export=ALL,INFENV_ROOT=" + ROOT]
        if os.environ.get("IE_ACCOUNT"):
            cmd.append("--account=" + os.environ["IE_ACCOUNT"])
        if os.environ.get("IE_QOS"):
            cmd.append("--qos=" + os.environ["IE_QOS"])
        cmd.append(os.path.join(SLURM, "compare.sbatch"))
        r = run(cmd)
        print("comparison job queued after all arrays: %s" % (r.stdout.strip() if r.returncode == 0 else "FAILED: " + r.stderr.strip()))
    print("\nmonitor: slurm/status.sh      per-task Slurm info: slurm/status.sh --sacct      logs: slurm/logs/")


if __name__ == "__main__":
    main()

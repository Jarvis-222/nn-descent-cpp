"""Shared parsers for build/search reports."""
import os
import re
from glob import glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
SEARCH = os.path.join(RESULTS, "search")


def parse_build(path):
    """Return dict with summary + per-iter CSV rows."""
    out = {"path": path, "iters": []}
    with open(path) as f:
        text = f.read()

    def grab(pattern, cast=float, default=None):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default

    out["n"] = grab(r"n=(\d+)", int)
    out["dim"] = grab(r"dim=(\d+)", int)
    out["init"] = grab(r"init=(\S+)", str)
    out["init_dist"] = grab(r"init_dist_comps=(\d+)", int, 0)
    out["init_time"] = grab(r"init_time=([\d.]+)s", float, 0.0)
    out["total_dist"] = grab(r"total_dist=(\d+)", int, 0)
    out["total_time"] = grab(r"total_time=([\d.]+)s", float, 0.0)
    out["recall"] = grab(r"final_recall=([\d.]+)", float, 0.0)

    # Fallback: old log format (no structured summary block)
    if out["total_dist"] == 0:
        out["total_dist"] = grab(r"\[final\] Total dist_comps = (\d+)", int, 0)
    if out["total_time"] == 0.0:
        out["total_time"] = grab(r"\[final\] Total time = ([\d.]+)s", float, 0.0)
    if out["recall"] == 0.0:
        out["recall"] = grab(r"\[final\] Recall = ([\d.]+)", float, 0.0)
    if out["init_dist"] == 0:
        out["init_dist"] = grab(r"\[init\] Done\. dist_comps=(\d+)", int, 0)
    if out["init_time"] == 0.0:
        out["init_time"] = grab(r"\[init\] Done\. dist_comps=\d+ time=([\d.]+)s", float, 0.0)

    if "filter=OFF" in text:
        out["pt"] = None
    else:
        out["pt"] = grab(r"pτ=([\d.]+)", float)

    csv_block = re.search(r"--- CSV ---\n(.+?)\n=", text, re.S)
    if csv_block:
        for line in csv_block.group(1).strip().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 7:
                continue
            out["iters"].append({
                "iter": int(parts[0]),
                "dist": int(parts[1]),
                "filtered": int(parts[2]),
                "rate": float(parts[3]),
                "updates": int(parts[4]),
                "cumul": int(parts[5]),
                "recall": float(parts[6]),
            })

    # Fallback: parse per-iteration lines from old log format
    if not out["iters"]:
        for m in re.finditer(
            r"\[iter (\d+)\] updates=(\d+) dist=(\d+) filtered=(\d+) "
            r"rate=(\d+)% cumul=(\d+) recall=([\d.e+-]+)",
            text,
        ):
            out["iters"].append({
                "iter": int(m.group(1)),
                "dist": int(m.group(3)),
                "filtered": int(m.group(4)),
                "rate": float(m.group(5)),
                "updates": int(m.group(2)),
                "cumul": int(m.group(6)),
                "recall": float(m.group(7)),
            })

    return out


def parse_search(path):
    """Parse single-graph search file -> list of (ef, recall, qps, dist, time)."""
    rows = []
    with open(path) as f:
        in_csv = False
        for line in f:
            line = line.strip()
            if line.startswith("ef,recall"):
                in_csv = True
                continue
            if in_csv:
                if not line or not line[0].isdigit():
                    break
                p = line.split(",")
                rows.append((int(p[0]), float(p[1]), float(p[2]),
                             int(p[3]), float(p[4])))
    return rows


def collect_filter_sweep(tag, init):
    """Return list of build dicts sorted by pτ (None first).

    If the *_build.txt run was done with --no-gt (recall=0), we fall back
    to the *_search.txt sibling — those runs include a full build with GT
    loaded and thus report a real final_recall.
    """
    build_paths = glob(os.path.join(RESULTS, f"{tag}_{init}_*_build.txt"))
    # Also pick up full-build-with-GT runs that only exist as _search.txt
    # (files living in RESULTS/, not RESULTS/search/)
    search_paths = glob(os.path.join(RESULTS, f"{tag}_{init}_*_search.txt"))
    by_suffix = {}
    for fp in build_paths:
        key = os.path.basename(fp).replace("_build.txt", "")
        by_suffix[key] = fp
    for fp in search_paths:
        key = os.path.basename(fp).replace("_search.txt", "")
        if key not in by_suffix:
            by_suffix[key] = fp

    out = []
    for key, fp in by_suffix.items():
        d = parse_build(fp)
        # If the primary run was --no-gt (recall=0), pull recall *only* from
        # the sibling. Keep this run's timing/dist so bar charts stay
        # comparable across pτ (both _build.txt and _search.txt re-run the
        # build, but at slightly different wall-clock times).
        if d["recall"] == 0.0 and fp.endswith("_build.txt"):
            alt = fp.replace("_build.txt", "_search.txt")
            if os.path.exists(alt):
                da = parse_build(alt)
                if da["recall"] > 0.0:
                    d["recall"] = da["recall"]
                    if not d["iters"] or d["iters"][-1]["recall"] == 0.0:
                        d["iters"] = da["iters"]
        out.append(d)
    out.sort(key=lambda d: (-1 if d["pt"] is None else d["pt"]))
    return out


def collect_mc_sweep(tag):
    """Return list of build dicts sorted by mc, from *_mc*_nofilter_r1.txt.

    Also includes the mc=40 baseline from the regular nofilter build.
    Each returned dict has an extra 'mc' key.
    """
    pat = os.path.join(RESULTS, f"{tag}_mc*_nofilter_r1.txt")
    out = []
    for fp in glob(pat):
        m = re.search(r"_mc(\d+)_", os.path.basename(fp))
        if m:
            mc = int(m.group(1))
            d = parse_build(fp)
            d["mc"] = mc
            out.append(d)
    # Add mc=40 baseline from the regular nofilter run
    for suffix in ("_build.txt", "_search.txt"):
        bp = os.path.join(RESULTS, f"{tag}_random_nofilter{suffix}")
        if os.path.exists(bp):
            d = parse_build(bp)
            d["mc"] = 40
            out.append(d)
            break
    out.sort(key=lambda d: d["mc"])
    return out


def collect_search_sweep(tag, init):
    pat = os.path.join(SEARCH, f"{tag}_{init}_*_search.txt")
    out = []
    for fp in glob(pat):
        name = os.path.basename(fp).replace("_search.txt", "")
        m = re.search(r"pt(\d+)", name)
        pt = int(m.group(1)) / 100.0 if m else None
        if "nofilter" in name:
            pt = None
        rows = parse_search(fp)
        if rows:
            out.append({"pt": pt, "rows": rows, "name": name})
    out.sort(key=lambda d: (-1 if d["pt"] is None else d["pt"]))
    return out

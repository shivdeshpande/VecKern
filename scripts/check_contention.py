#!/usr/bin/env python3
"""Flag benchmark commands that overlapped another job's GPU work.

Reads run_all.log (commands bracketed by "+ start=<epoch> ..." / "+ end=<epoch>", plus the
"[wait_idle_gpu] GPU <i> idle ..." line that says which GPU each command used) and gpu_monitor.log
(one line per 2 s from `nvidia-smi pmon`: "<epoch> <gpu> <pid> <sm%> <name>;..."). A command is CONTENDED if any
sample inside its time window shows another process (not this project's tooling) using SM time on its GPU.
Processes that are resident but idle (sm "-" or 0) do not count.

usage: scripts/check_contention.py results/final
"""
import re
import sys
from datetime import datetime
from pathlib import Path

OWN = re.compile(r"veckern_bench|test_gpu|ncu|python")  # this project's own processes (pmon shows short names)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/final")
    samples = []  # (epoch, [(gpu, name, sm%)])
    for line in (out / "gpu_monitor.log").read_text().splitlines():
        ts, _, rest = line.partition(" ")
        procs = []
        for p in rest.split(";"):
            f = p.split()
            if len(f) >= 4 and f[2].isdigit():
                procs.append((f[0], f[3], int(f[2])))
        samples.append((int(ts), procs))

    lines = (out / "run_all.log").read_text().splitlines()
    # Older logs print the idle moment only as local server time. Calibrate the server's UTC offset from a
    # "=== <local time>" step header followed by a "+ start=<epoch>" line (written within the same second).
    offset = None
    for a, b in zip(lines, lines[1:]):
        if a.startswith("=== ") and b.startswith("+ start="):
            local = datetime.strptime(a[4:23], "%Y-%m-%d %H:%M:%S")
            offset = int(b[len("+ start="):].split()[0]) - int(local.replace(tzinfo=None).timestamp())
            break

    commands, current, gpu = [], None, None
    for line in lines:
        if line.startswith("+ start="):
            ts, cmd = line[len("+ start="):].split(" ", 1)
            current, gpu = {"start": int(ts), "cmd": cmd}, None
        elif "[wait_idle_gpu] GPU" in line and current is not None:
            gpu = line.split("GPU", 1)[1].split()[0]
            if "NOT checked" in line:
                current["unchecked"] = True
            # The measurement starts when the GPU was found idle, not when the command was queued.
            m = re.search(r"epoch=(\d+)", line)
            if m:
                current["start"] = int(m.group(1))
            else:
                m = re.search(r"at (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
                if m and offset is not None:
                    local = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    current["start"] = int(local.timestamp()) + offset
        elif line.startswith("+ end=") and current is not None:
            current["end"], current["gpu"] = int(line[len("+ end="):]), gpu
            commands.append(current)
            current = None

    bad = 0
    for c in commands:
        if c["gpu"] is None:  # CPU-only command
            continue
        hits = {name for ts, procs in samples if c["start"] <= ts <= c["end"]
                for g, name, sm in procs if g == c["gpu"] and sm > 0 and not OWN.search(name)}
        status = "CONTENDED by " + ", ".join(sorted(hits)) if hits else "clean"
        if c.get("unchecked"):
            status += ", idle check skipped"
        bad += bool(hits)
        print(f"[{status}] gpu{c['gpu']} ran {c['end'] - c['start']:5d}s  {c['cmd'][:140]}")
    print(f"\n{len(commands)} commands, {bad} contended")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

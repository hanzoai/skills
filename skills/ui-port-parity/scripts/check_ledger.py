#!/usr/bin/env python3
"""Fail a UI parity run when unresolved blocking findings remain.

Input JSON may be either a list of findings or an object with a `findings` list.
Each finding should contain: id, class, priority, status. Optional: screen, region.

Blocking rule:
- unresolved PARITY findings with priority P0 or P1 fail
- unresolved topology findings fail at any priority when `gate` == "topology"
- UNMEASURED/ABSENCE/EXEMPT/SHARED-BUG do not fail parity by themselves
"""

import json
import sys
from pathlib import Path

DONE = {"done", "fixed", "resolved", "pass", "passed", "closed"}


def load(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    return data.get("findings", [])


def main():
    if len(sys.argv) != 2:
        print("usage: check_ledger.py AUDIT.json", file=sys.stderr)
        return 2

    findings = load(sys.argv[1])
    blockers = []
    for item in findings:
        cls = str(item.get("class", "PARITY")).upper()
        priority = str(item.get("priority", "P2")).upper()
        status = str(item.get("status", "open")).lower()
        gate = str(item.get("gate", "")).lower()
        if status in DONE:
            continue
        if cls == "PARITY" and (priority in {"P0", "P1"} or gate == "topology"):
            blockers.append(item)

    if not blockers:
        print("PASS: no unresolved blocking parity findings")
        return 0

    print("FAIL: unresolved blocking parity findings")
    for item in blockers:
        ident = item.get("id", "?")
        screen = item.get("screen", "global")
        region = item.get("region", "")
        priority = item.get("priority", "")
        print(f"- {ident} {priority} {screen} {region}".rstrip())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

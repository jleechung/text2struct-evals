#!/usr/bin/env python3
"""
Build a small cache mapping CATH codes -> human-readable names (and example domains)
restricted to the C/A/T label space used by GearNet (from cath_label_mapping.pt).

Inputs:
  --label_mapping_pt  : proteina_additional_files/pdb_raw/cath_label_mapping.pt
  --names_file        : cath-b-newest-names.gz (or an uncompressed names file)
Output:
  --out_json          : eval-gearnet/cache/cath_node_names.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

CODE_RE = re.compile(r"^\d+(?:\.\d+)*$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _open_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def load_gearnet_label_space(label_mapping_pt: Path) -> Dict[str, set[str]]:
    import torch

    obj = torch.load(str(label_mapping_pt), map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {label_mapping_pt}, got {type(obj).__name__}")

    out: Dict[str, set[str]] = {}
    for level in ("C", "A", "T"):
        m = obj.get(level)
        if not isinstance(m, dict):
            raise ValueError(f"Expected dict for mapping[{level}], got {type(m).__name__}")
        out[level] = set(str(k) for k in m.keys())  # label strings
    return out


def parse_names_lines(lines: Iterable[str]) -> Dict[str, Tuple[str, str]]:
    """
    Return mapping: code -> (name, example_domain)

    We don't assume strict format; we accept:
      - tab-separated: code <tab> name <tab> example ...
      - whitespace: code <space> name...
    """
    names: Dict[str, Tuple[str, str]] = {}

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        # Many CATH flat files include header-ish fields; skip them
        if line.startswith(("NAME:", "FILE", "CATH", "VERSION")):
            continue

        # First token should be the CATH node code
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip() != ""]
        else:
            parts = line.split()

        if not parts:
            continue

        code = parts[0]
        if not CODE_RE.match(code):
            continue

        # Heuristic: if tabbed, second col is usually name; third is example
        name = parts[1] if len(parts) >= 2 else ""
        example = parts[2] if len(parts) >= 3 else ""

        # If whitespace-split and the "name" is actually multiple tokens, we lose spaces.
        # Try to recover by slicing from the original line after the first token.
        if "\t" not in line and len(parts) > 2:
            # Keep everything after code as "name (maybe plus example)"; we can't reliably split.
            tail = line[len(code) :].strip()
            # If the tail contains something like an example domain token at end, we can't be sure.
            # Keep it as name and leave example blank.
            name = tail
            example = ""

        names[code] = (name, example)

    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_mapping_pt", required=True)
    ap.add_argument("--names_file", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    label_mapping_pt = Path(args.label_mapping_pt).expanduser().resolve()
    names_file = Path(args.names_file).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()

    if not label_mapping_pt.exists():
        raise FileNotFoundError(label_mapping_pt)
    if not names_file.exists():
        raise FileNotFoundError(names_file)

    label_space = load_gearnet_label_space(label_mapping_pt)

    with _open_maybe_gz(names_file) as f:
        names_all = parse_names_lines(f)

    # Restrict to GearNet label space
    out: Dict[str, Dict[str, Dict[str, str]]] = {"C": {}, "A": {}, "T": {}}
    for level in ("C", "A", "T"):
        for code in sorted(label_space[level]):
            if code in names_all:
                name, example = names_all[code]
                out[level][code] = {
                    "name": name,
                    "example_domain": example,
                }

    payload = {
        "meta": {
            "created_at": utc_now_iso(),
            "names_file": str(names_file),
            "label_mapping_pt": str(label_mapping_pt),
        },
        "levels": out,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_json} (C={len(out['C'])} A={len(out['A'])} T={len(out['T'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

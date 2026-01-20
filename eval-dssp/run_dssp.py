#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is on sys.path so `import utils` works everywhere (cluster, SLURM, etc.)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.batch_utils import create_batches
from utils.io_utils import (
    ensure_dir,
    iter_structures,
    load_manifest,
    output_json_path,
    write_json_atomic,
)
from utils.schema import validate_output_json


@dataclass(frozen=True)
class StructureRec:
    structure_id: str
    structure_path: str
    accession: str
    sample_num: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_mkdssp_version(mkdssp_bin: str) -> Optional[str]:
    try:
        r = subprocess.run(
            [mkdssp_bin, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        v = out if out else err
        return v if v else None
    except Exception:
        return None


def run_mkdssp(
    mkdssp_bin: str,
    pdb_path: str,
    dssp_out_path: Path,
    *,
    verbose: bool = False,
) -> Tuple[int, str, str]:
    """Run mkdssp and write classic DSSP output to dssp_out_path."""
    ensure_dir(dssp_out_path.parent)

    cmd = [mkdssp_bin, "--output-format", "dssp"]
    if verbose:
        cmd.append("--verbose")
    cmd.extend([pdb_path, str(dssp_out_path)])

    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def _segment_lengths(ss: str, allowed: set[str], *, min_len: int = 1) -> list[int]:
    lens: list[int] = []
    cur = 0
    for ch in ss:
        if ch in allowed:
            cur += 1
        else:
            if cur >= min_len:
                lens.append(cur)
            cur = 0
    if cur >= min_len:
        lens.append(cur)
    return lens


def compute_dssp_summary(ss: str, residues: list[dict]) -> dict:
    """Compute small, stable derived metrics from DSSP assignments.

    Includes both raw segment counts (min_len=1) and filtered segment counts:
      - Helix (H) segments counted with min_len >= 4
      - Strand (E) segments counted with min_len >= 2

    Intended for fast aggregation; prompt-specific scoring belongs in Step 2.
    """
    n_res = len(ss)

    # Counts for all symbols present
    counts: dict[str, int] = {}
    for ch in ss:
        counts[ch] = counts.get(ch, 0) + 1

    # Fractions for common DSSP symbols
    symbols = ["H", "E", "C", "T", "S", "G", "I", "B"]
    fracs = {f"frac_{sym}": (counts.get(sym, 0) / n_res if n_res else 0.0) for sym in symbols}

    # Helix fractions: strict alpha (H) and any helix (H/G/I)
    frac_helix_any = (
        (counts.get("H", 0) + counts.get("G", 0) + counts.get("I", 0)) / n_res
        if n_res
        else 0.0
    )
    frac_strand = (counts.get("E", 0) / n_res) if n_res else 0.0

    # Segment/run stats
    H_lens_raw = _segment_lengths(ss, {"H"}, min_len=1)
    H_lens_ge4 = _segment_lengths(ss, {"H"}, min_len=4)

    E_lens_raw = _segment_lengths(ss, {"E"}, min_len=1)
    E_lens_ge2 = _segment_lengths(ss, {"E"}, min_len=2)

    helix_any_lens_raw = _segment_lengths(ss, {"H", "G", "I"}, min_len=1)
    helix_any_lens_ge3 = _segment_lengths(ss, {"H", "G", "I"}, min_len=3)

    def seg_stats(lens: list[int]) -> dict:
        if not lens:
            return {"n_segments": 0, "mean_len": 0.0, "max_len": 0}
        return {
            "n_segments": len(lens),
            "mean_len": float(statistics.mean(lens)),
            "max_len": int(max(lens)),
        }

    # Transitions (how "fragmented" the assignment is)
    n_transitions = 0
    for i in range(1, n_res):
        if ss[i] != ss[i - 1]:
            n_transitions += 1

    # Accessibility stats (if present)
    acc_vals = [r.get("acc") for r in residues if r.get("acc") is not None]
    acc_vals = [int(a) for a in acc_vals]  # type: ignore[arg-type]
    if acc_vals:
        acc_mean = float(statistics.mean(acc_vals))
        acc_median = float(statistics.median(acc_vals))
        frac_exposed_ge_30 = float(sum(a >= 30 for a in acc_vals) / len(acc_vals))
    else:
        acc_mean = None
        acc_median = None
        frac_exposed_ge_30 = None

    # Convenience ratios
    helix_to_strand_ratio = (frac_helix_any / frac_strand) if frac_strand > 0 else None

    return {
        "n_res": n_res,
        **fracs,
        "frac_helix_any": frac_helix_any,
        "frac_strand": frac_strand,
        "helix_to_strand_ratio": helix_to_strand_ratio,
        # Segment stats: raw (min_len=1) and filtered (common thresholds)
        "H_segments_raw": seg_stats(H_lens_raw),
        "H_segments_ge_4": seg_stats(H_lens_ge4),
        "E_segments_raw": seg_stats(E_lens_raw),
        "E_segments_ge_2": seg_stats(E_lens_ge2),
        "helix_segments_any_raw": seg_stats(helix_any_lens_raw),
        "helix_segments_any_ge_3": seg_stats(helix_any_lens_ge3),
        # Long-helix counters (computed from raw H runs)
        "n_long_H_ge_10": int(sum(l >= 10 for l in H_lens_raw)),
        "n_long_H_ge_15": int(sum(l >= 15 for l in H_lens_raw)),
        "n_transitions": n_transitions,
        "acc_mean": acc_mean,
        "acc_median": acc_median,
        "frac_exposed_ge_30": frac_exposed_ge_30,
    }


def parse_classic_dssp(dssp_path: Path) -> Dict:
    """Parse classic DSSP text output into a compact JSON-friendly structure."""
    lines = dssp_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Find start of residue table (line beginning with '  #')
    start_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("  #"):
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError("Could not find DSSP residue table header (line starting with '  #').")

    residues: List[Dict] = []
    ss_chars: List[str] = []
    counts: Dict[str, int] = {}

    for ln in lines[start_idx:]:
        if not ln.strip():
            continue
        if len(ln) < 40:
            continue

        try:
            dssp_idx = int(ln[0:5].strip())
        except Exception:
            continue

        resseq = ln[5:10].strip()
        icode = ln[10].strip()  # insertion code
        chain = ln[11].strip()
        aa = ln[13].strip()
        ss = ln[16].strip()  # blank means coil

        # Skip chain breaks / missing residues markers
        if aa == "!" or aa == "":
            continue

        # Normalize coil: DSSP uses blank for coil; represent as 'C'
        if ss == "":
            ss = "C"

        # Accessibility column (best effort)
        acc_val = None
        acc_str = ln[34:38].strip()
        if acc_str:
            try:
                acc_val = int(acc_str)
            except Exception:
                acc_val = None

        residues.append(
            {
                "dssp_index": dssp_idx,
                "resseq": resseq,
                "icode": icode if icode else None,
                "chain": chain if chain else None,
                "aa": aa,
                "ss": ss,
                "acc": acc_val,
            }
        )

        ss_chars.append(ss)
        counts[ss] = counts.get(ss, 0) + 1

    ss_str = "".join(ss_chars)
    summary = compute_dssp_summary(ss_str, residues)

    return {
        "secondary_structure": ss_str,
        "counts": counts,
        "residues": residues,
        "summary": summary,
    }


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("eval-dssp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def main() -> int:
    ap = argparse.ArgumentParser(description="Run DSSP (mkdssp) over a manifest of PDB paths.")
    ap.add_argument("--manifest", required=True, help="Path to manifest CSV (with header).")
    ap.add_argument("--run_name", required=True, help="Run name (e.g., baseline, model).")
    ap.add_argument("--output_root", default="results", help="Root output dir (default: results).")
    ap.add_argument("--log_root", default="logs", help="Root log dir (default: logs).")
    ap.add_argument("--batch_size", type=int, default=50, help="Batch size (default: 50).")

    ap.add_argument("--mkdssp_bin", default="mkdssp", help="Path/name of mkdssp binary (default: mkdssp).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON outputs.")
    ap.add_argument("--allow_missing_files", action="store_true", help="Do not fail if some structure files are missing.")
    ap.add_argument("--verbose_mkdssp", action="store_true", help="Pass --verbose to mkdssp.")

    args = ap.parse_args()

    run_name = args.run_name.strip()
    if not run_name:
        raise ValueError("--run_name must be non-empty")

    eval_name = "dssp"

    log_path = Path(args.log_root) / run_name / eval_name / "run.log"
    logger = setup_logger(log_path)

    logger.info("Starting DSSP eval")
    logger.info("manifest=%s run_name=%s output_root=%s", args.manifest, run_name, args.output_root)

    mkdssp_version = get_mkdssp_version(args.mkdssp_bin)
    if mkdssp_version:
        logger.info("mkdssp version: %s", mkdssp_version)

    df = load_manifest(
        args.manifest,
        resolve_paths=True,
        allow_missing_files=args.allow_missing_files,
    )

    recs: List[StructureRec] = [
        StructureRec(structure_id=sid, structure_path=spath, accession=acc, sample_num=sn)
        for (sid, spath, acc, sn) in iter_structures(df)
    ]
    logger.info("Loaded %d structures", len(recs))

    batches = create_batches(recs, args.batch_size)
    logger.info("Created %d batches (batch_size=%d)", len(batches), args.batch_size)

    n_ok = 0
    n_fail = 0
    n_skip = 0

    for bi, batch in enumerate(batches, start=1):
        logger.info("Batch %d/%d (%d structures)", bi, len(batches), len(batch))

        for rec in batch:
            out_json = output_json_path(args.output_root, run_name, eval_name, rec.accession, rec.sample_num)
            raw_out = out_json.with_suffix(".dssp")

            if out_json.exists() and not args.overwrite:
                n_skip += 1
                continue

            t0 = time.perf_counter()
            status = "success"
            err_msg: Optional[str] = None
            predictions = None
            raw_path_str: Optional[str] = None

            try:
                pdb_path = rec.structure_path
                if not Path(pdb_path).exists():
                    raise FileNotFoundError(f"Structure file not found: {pdb_path}")

                rc, so, se = run_mkdssp(
                    args.mkdssp_bin,
                    pdb_path,
                    raw_out,
                    verbose=args.verbose_mkdssp,
                )

                if raw_out.exists():
                    raw_path_str = str(raw_out)

                if rc != 0:
                    raise RuntimeError(f"mkdssp failed (rc={rc}). stderr: {se.strip() or '<empty>'}")

                predictions = parse_classic_dssp(raw_out)

            except Exception as e:
                status = "failed"
                err_msg = str(e)

            runtime = time.perf_counter() - t0

            out_obj = {
                "structure_id": rec.structure_id,
                "structure_path": rec.structure_path,
                "accession": rec.accession,
                "sample_num": rec.sample_num,
                "eval_type": "dssp",
                "status": status,
                "error": err_msg,
                "predictions": predictions,
                "raw_output_path": raw_path_str,
                "timestamp": utc_now_iso(),
                "runtime_seconds": float(runtime),
            }

            # Helpful extras (schema allows extras)
            if mkdssp_version:
                out_obj["tool_version"] = mkdssp_version
            out_obj["tool"] = args.mkdssp_bin

            try:
                validate_output_json(out_obj)
            except Exception as ve:
                out_obj["status"] = "failed"
                out_obj["error"] = f"Schema validation failed: {ve}"
                out_obj["predictions"] = None

            write_json_atomic(out_json, out_obj)

            if out_obj["status"] == "success":
                n_ok += 1
            else:
                n_fail += 1
                logger.warning(
                    "FAIL %s (%s sample=%s): %s",
                    rec.accession,
                    rec.structure_id,
                    rec.sample_num,
                    out_obj["error"],
                )

    logger.info("Done. success=%d failed=%d skipped=%d", n_ok, n_fail, n_skip)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

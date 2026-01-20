#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Ensure repo root is on sys.path so `import utils` works everywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.batch_utils import create_batches
from utils.io_utils import ensure_dir, iter_structures, load_manifest, output_json_path, write_json_atomic
from utils.schema import validate_output_json


@dataclass(frozen=True)
class StructureRec:
    structure_id: str
    structure_path: str
    accession: str
    sample_num: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("eval-p2rank")
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


def run_prank_version(prank_bin: str) -> Optional[str]:
    try:
        r = subprocess.run([prank_bin, "-v"], check=False, capture_output=True, text=True)
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        s = out if out else err
        return s if s else None
    except Exception:
        return None


def run_p2rank_predict(
    prank_bin: str,
    input_pdb: Path,
    out_dir: Path,
    *,
    threads: int = 1,
    profile: str = "alphafold",
    visualizations: bool = False,
) -> Tuple[int, str, str]:
    """Run `prank predict -f <pdb> -o <outdir> ...`."""
    ensure_dir(out_dir)

    cmd = [
        prank_bin,
        "predict",
        "-f",
        str(input_pdb),
        "-o",
        str(out_dir),
        "-threads",
        str(int(threads)),
    ]

    if profile:
        cmd.extend(["-c", profile])

    if not visualizations:
        cmd.extend(["-visualizations", "0"])

    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def _find_single(glob_pat: str, root: Path) -> Path:
    hits = sorted(root.glob(glob_pat))
    if not hits:
        raise FileNotFoundError(f"Expected file matching {glob_pat} in {root}, found none.")
    return hits[0]


def parse_p2rank_outputs(pred_csv: Path, res_csv: Path, *, top_k: int = 5, top_res_k: int = 50) -> Dict[str, Any]:
    pockets_df = pd.read_csv(pred_csv)
    residues_df = pd.read_csv(res_csv)
    pockets_df.columns = [c.strip() for c in pockets_df.columns]
    residues_df.columns = [c.strip() for c in residues_df.columns]

    pockets_top = pockets_df.head(top_k).copy()

    # Choose a reasonable sort column for residues (probability/score-like)
    sort_col = None
    for c in residues_df.columns:
        if "prob" in c.lower():
            sort_col = c
            break
    if sort_col is None:
        for c in residues_df.columns:
            lc = c.lower()
            if lc == "score" or lc.endswith("score"):
                sort_col = c
                break

    if sort_col:
        residues_top = residues_df.sort_values(by=sort_col, ascending=False).head(top_res_k).copy()
    else:
        residues_top = residues_df.head(min(top_res_k, len(residues_df))).copy()

    # Summary fields (best effort)
    top_score = None
    for c in pockets_df.columns:
        if c.lower() == "score" and len(pockets_df) > 0:
            try:
                top_score = float(pockets_df.iloc[0][c])
            except Exception:
                top_score = None
            break

    top_prob = None
    for c in pockets_df.columns:
        if "prob" in c.lower() and len(pockets_df) > 0:
            try:
                top_prob = float(pockets_df.iloc[0][c])
            except Exception:
                top_prob = None
            break

    return {
        "summary": {
            "num_pockets": int(len(pockets_df)),
            "top_score": top_score,
            "top_probability": top_prob,
            "top_k": int(top_k),
            "top_res_k": int(top_res_k),
        },
        "pockets_columns": pockets_df.columns.tolist(),
        "residues_columns": residues_df.columns.tolist(),
        "pockets_topk": pockets_top.to_dict(orient="records"),
        "residues_topk": residues_top.to_dict(orient="records"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run P2Rank (prank) over a manifest of PDB paths.")
    ap.add_argument("--manifest", required=True, help="Path to manifest CSV (with header).")
    ap.add_argument("--run_name", required=True, help="Run name (e.g., baseline, model).")
    ap.add_argument("--output_root", default="results", help="Root output dir.")
    ap.add_argument("--log_root", default="logs", help="Root log dir.")
    ap.add_argument("--batch_size", type=int, default=50, help="Batch size.")

    ap.add_argument(
        "--prank_bin",
        default=str(Path("eval-p2rank/vendor/p2rank/prank")),
        help="Path to prank executable (default points to repo vendor path).",
    )
    ap.add_argument("--threads", type=int, default=1, help="P2Rank threads per structure.")
    ap.add_argument("--profile", default="alphafold", help="P2Rank config profile (e.g. alphafold).")
    ap.add_argument("--visualizations", action="store_true", help="Enable visualization outputs (default off).")

    ap.add_argument("--top_k", type=int, default=5, help="Number of top pockets to store in JSON.")
    ap.add_argument("--top_res_k", type=int, default=50, help="Number of top residues to store in JSON.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON outputs.")
    ap.add_argument("--allow_missing_files", action="store_true", help="Do not fail if some structure files are missing.")

    args = ap.parse_args()

    run_name = args.run_name.strip()
    if not run_name:
        raise ValueError("--run_name must be non-empty")

    eval_name = "p2rank"
    log_path = Path(args.log_root) / run_name / eval_name / "run.log"
    logger = setup_logger(log_path)

    prank_bin = Path(args.prank_bin).expanduser()
    if not prank_bin.exists():
        raise FileNotFoundError(f"prank executable not found: {prank_bin}")

    prank_version = run_prank_version(str(prank_bin))
    logger.info("Starting P2Rank eval")
    logger.info("manifest=%s run_name=%s output_root=%s", args.manifest, run_name, args.output_root)
    if prank_version:
        logger.info("prank -v:\n%s", prank_version)

    df = load_manifest(args.manifest, resolve_paths=True, allow_missing_files=args.allow_missing_files)
    recs: List[StructureRec] = [
        StructureRec(structure_id=sid, structure_path=spath, accession=acc, sample_num=sn)
        for (sid, spath, acc, sn) in iter_structures(df)
    ]

    batches = create_batches(recs, args.batch_size)
    logger.info("Loaded %d structures; %d batches (batch_size=%d)", len(recs), len(batches), args.batch_size)

    n_ok = 0
    n_fail = 0
    n_skip = 0

    for bi, batch in enumerate(batches, start=1):
        logger.info("Batch %d/%d (%d structures)", bi, len(batches), len(batch))

        for rec in batch:
            out_json = output_json_path(args.output_root, run_name, eval_name, rec.accession, rec.sample_num)

            raw_pred = out_json.with_suffix(".p2rank_predictions.csv")
            raw_res = out_json.with_suffix(".p2rank_residues.csv")

            if out_json.exists() and not args.overwrite:
                n_skip += 1
                continue

            t0 = time.perf_counter()
            status = "success"
            err_msg: Optional[str] = None
            predictions: Optional[Dict[str, Any]] = None

            work_dir = Path("eval-p2rank/temp") / rec.accession / f"sample_{rec.sample_num}"
            tmp_out = work_dir / "out"
            ensure_dir(tmp_out)

            try:
                pdb_path = Path(rec.structure_path)
                if not pdb_path.exists():
                    raise FileNotFoundError(f"Structure file not found: {pdb_path}")

                rc, so, se = run_p2rank_predict(
                    str(prank_bin),
                    pdb_path,
                    tmp_out,
                    threads=args.threads,
                    profile=args.profile,
                    visualizations=args.visualizations,
                )
                if rc != 0:
                    raise RuntimeError(f"prank predict failed (rc={rc}). stderr: {se.strip() or '<empty>'}")

                pred_csv = _find_single("*_predictions.csv", tmp_out)
                res_csv = _find_single("*_residues.csv", tmp_out)

                ensure_dir(out_json.parent)
                shutil.copy2(pred_csv, raw_pred)
                shutil.copy2(res_csv, raw_res)

                predictions = parse_p2rank_outputs(raw_pred, raw_res, top_k=args.top_k, top_res_k=args.top_res_k)

            except Exception as e:
                status = "failed"
                err_msg = str(e)

            runtime = time.perf_counter() - t0

            out_obj = {
                "structure_id": rec.structure_id,
                "structure_path": rec.structure_path,
                "accession": rec.accession,
                "sample_num": rec.sample_num,
                "eval_type": "p2rank",
                "status": status,
                "error": err_msg,
                "predictions": predictions,
                "raw_output_path": str(raw_pred) if raw_pred.exists() else None,
                "timestamp": utc_now_iso(),
                "runtime_seconds": float(runtime),
            }

            out_obj["tool"] = str(prank_bin)
            if prank_version:
                out_obj["tool_version"] = prank_version
            out_obj["raw_output_paths"] = {
                "predictions_csv": str(raw_pred) if raw_pred.exists() else None,
                "residues_csv": str(raw_res) if raw_res.exists() else None,
            }

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

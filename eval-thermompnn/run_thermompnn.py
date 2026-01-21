#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import argparse
import json
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


def _write_local_yaml_from_template(
    template_path: Path,
    dst_path: Path,
    *,
    thermompnn_dir: Path,
    cache_dir: Path,
    accel: str = "gpu",
) -> None:
    ensure_dir(dst_path.parent)
    ensure_dir(cache_dir)

    thermo_str = str(thermompnn_dir)
    cache_str = str(cache_dir)

    if template_path.exists():
        text = template_path.read_text(encoding="utf-8")

        def repl(key: str, value: str, s: str) -> str:
            pat = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*)(\"[^\"]*\"|'[^']*'|.*)\s*$", re.M)
            return pat.sub(rf'\1"{value}"', s) if pat.search(s) else s

        text = repl("accel", accel, text)
        text = repl("cache_dir", cache_str, text)
        text = repl("thermompnn_dir", thermo_str, text)

        dst_path.write_text(text, encoding="utf-8")
        return

    # fallback if template missing
    dst_path.write_text(
        f"""platform:
  accel: "{accel}"
  cache_dir: "{cache_str}"
  thermompnn_dir: "{thermo_str}"
data_loc: {{}}
""",
        encoding="utf-8",
    )


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
    logger = logging.getLogger("eval-thermompnn")
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


def infer_chain_from_accession(accession: str) -> Optional[str]:
    """
    Heuristic: if accession looks like '5fo5_A' or 'protein_001_A', return 'A'.
    Otherwise return None and let caller decide.
    """
    acc = accession.strip()
    if "_" not in acc:
        return None
    last = acc.split("_")[-1]
    if len(last) == 1 and last.isalnum():
        return last
    return None


def run_custom_inference(
    thermompnn_dir: Path,
    pdb_path: Path,
    work_dir: Path,
    *,
    chain: Optional[str] = None,
    model_path: Optional[Path] = None,
    python_bin: str = sys.executable,
    extra_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    """
    Runs ThermoMPNN analysis/custom_inference.py.
    
    Copies local.yaml to parent of work_dir so the script's ../local.yaml reference works.
    """
    ensure_dir(work_dir)
    
    # Copy local.yaml to parent of work_dir since script does ../local.yaml
    local_yaml_src = thermompnn_dir / "local.yaml"
    local_yaml_dst = work_dir.parent / "local.yaml"

    # default: repo-local cache
    default_cache = (REPO_ROOT / "eval-thermompnn" / "cache").resolve()

    # optional override (still nice for scratch)
    env0 = env or os.environ
    cache_dir = Path(env0.get("THERMOMPNN_CACHE_DIR", str(default_cache))).expanduser().resolve()
    accel = env0.get("THERMOMPNN_ACCEL", "gpu").strip() or "gpu"

    _write_local_yaml_from_template(
        local_yaml_src,
        local_yaml_dst,
        thermompnn_dir=thermompnn_dir,
        cache_dir=cache_dir,
        accel=accel,
    )

    script = thermompnn_dir / "analysis" / "custom_inference.py"
    if not script.exists():
        raise FileNotFoundError(f"ThermoMPNN custom_inference.py not found: {script}")

    cmd = [python_bin, str(script), "--pdb", str(pdb_path)]
    if chain:
        cmd += ["--chain", chain]
    if model_path:
        cmd += ["--model_path", str(model_path)]
    if extra_args:
        cmd += extra_args

    r = subprocess.run(
        cmd,
        cwd=str(work_dir),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def _find_ddg_column(df: pd.DataFrame) -> Optional[str]:
    # pick a column that looks like ddG / ddg
    for c in df.columns:
        if "ddg" in c.lower():
            return c
    return None


def _find_mutation_column(df: pd.DataFrame) -> Optional[str]:
    # try common names
    candidates = ["mutation", "mut", "variant", "mutant"]
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    # fallback: any column containing "mut"
    for c in df.columns:
        if "mut" in c.lower():
            return c
    return None


def summarize_ddg_csv(csv_path: Path, *, top_k: int = 10) -> Dict[str, Any]:
    """
    Reads a ThermoMPNN output CSV (if present) and computes a small summary.
    Assumes more negative ddG is more stabilizing (common convention).
    """
    df = pd.read_csv(csv_path)
    # normalize whitespace in headers
    df.columns = [str(c).strip() for c in df.columns]

    ddg_col = _find_ddg_column(df)
    mut_col = _find_mutation_column(df)

    if not ddg_col:
        return {
            "csv_path": str(csv_path),
            "num_rows": int(len(df)),
            "ddg_column": None,
            "mutation_column": mut_col,
            "note": "No ddG-like column found; stored columns only.",
            "columns": df.columns.tolist(),
        }

    ddg = pd.to_numeric(df[ddg_col], errors="coerce")
    ddg_valid = ddg.dropna()

    def top_rows(sort_ascending: bool) -> List[Dict[str, Any]]:
        d2 = df.copy()
        d2["_ddg_"] = pd.to_numeric(d2[ddg_col], errors="coerce")
        d2 = d2.dropna(subset=["_ddg_"]).sort_values("_ddg_", ascending=sort_ascending).head(top_k)
        cols = []
        if mut_col and mut_col in d2.columns:
            cols.append(mut_col)
        cols.append(ddg_col)
        # include a couple extra columns if present
        for c in d2.columns:
            if c in cols or c == "_ddg_":
                continue
            # keep it small
            if len(cols) >= 4:
                break
            cols.append(c)
        return d2[cols].to_dict(orient="records")

    summary = {
        "csv_path": str(csv_path),
        "num_rows": int(len(df)),
        "ddg_column": ddg_col,
        "mutation_column": mut_col,
        "ddg_mean": float(ddg_valid.mean()) if len(ddg_valid) else None,
        "ddg_median": float(ddg_valid.median()) if len(ddg_valid) else None,
        "ddg_min": float(ddg_valid.min()) if len(ddg_valid) else None,
        "ddg_max": float(ddg_valid.max()) if len(ddg_valid) else None,
        "num_stabilizing_ddg_lt_0": int((ddg_valid < 0).sum()) if len(ddg_valid) else 0,
        "num_destabilizing_ddg_gt_0": int((ddg_valid > 0).sum()) if len(ddg_valid) else 0,
        "top_stabilizing": top_rows(sort_ascending=True),   # most negative ddG
        "top_destabilizing": top_rows(sort_ascending=False),
        "columns": df.columns.tolist(),
    }
    return summary


def collect_outputs_to_results(work_out_dir: Path, results_raw_dir: Path) -> List[Path]:
    """
    Copy whatever ThermoMPNN wrote into a stable raw-output directory next to JSON.
    Returns list of copied file paths (in results_raw_dir).
    """
    ensure_dir(results_raw_dir)

    copied: List[Path] = []
    if not work_out_dir.exists():
        return copied

    for p in sorted(work_out_dir.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(work_out_dir)
        dst = results_raw_dir / rel
        ensure_dir(dst.parent)
        shutil.copy2(p, dst)
        copied.append(dst)

    return copied


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ThermoMPNN custom_inference over a manifest of PDBs.")
    ap.add_argument("--manifest", required=True, help="Path to manifest CSV (with header).")
    ap.add_argument("--run_name", required=True, help="Run name (e.g., baseline, model).")
    ap.add_argument("--output_root", default="results", help="Root output dir.")
    ap.add_argument("--log_root", default="logs", help="Root log dir.")
    ap.add_argument("--batch_size", type=int, default=10, help="Batch size (ThermoMPNN is heavier).")

    ap.add_argument(
        "--thermompnn_dir",
        default=str(Path("eval-thermompnn/vendor/ThermoMPNN")),
        help="Path to ThermoMPNN repo directory (vendored).",
    )
    ap.add_argument(
        "--model_path",
        default="",
        help="Path to model weights (.pt). Default empty -> use ThermoMPNN default models/thermoMPNN_default.pt.",
    )
    ap.add_argument("--chain", default="", help="Chain ID to use (default: infer from accession; else omit).")
    ap.add_argument("--gpu", default="", help="If set, exports CUDA_VISIBLE_DEVICES=<gpu> for ThermoMPNN call.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON outputs.")
    ap.add_argument("--allow_missing_files", action="store_true", help="Do not fail if some structure files are missing.")

    # parsing knobs
    ap.add_argument("--top_k", type=int, default=10, help="Top K stabilizing/destabilizing mutations to store if CSV found.")

    args = ap.parse_args()

    run_name = args.run_name.strip()
    if not run_name:
        raise ValueError("--run_name must be non-empty")

    eval_name = "thermompnn"
    log_path = Path(args.log_root) / run_name / eval_name / "run.log"
    logger = setup_logger(log_path)

    thermompnn_dir = Path(args.thermompnn_dir).expanduser().resolve()
    if not thermompnn_dir.exists():
        raise FileNotFoundError(f"ThermoMPNN dir not found: {thermompnn_dir}")

    # model path default
    model_path: Optional[Path]
    if args.model_path.strip():
        model_path = Path(args.model_path).expanduser().resolve()
    else:
        model_path = (thermompnn_dir / "models" / "thermoMPNN_default.pt").resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"ThermoMPNN model weights not found: {model_path}")

    logger.info("Starting ThermoMPNN eval")
    logger.info("manifest=%s run_name=%s output_root=%s", args.manifest, run_name, args.output_root)
    logger.info("thermompnn_dir=%s", thermompnn_dir)
    logger.info("model_path=%s", model_path)

    # Build env for ThermoMPNN subprocess
    import os

    env_sub = dict(os.environ)

    # Make ThermoMPNN imports like `from datasets import Mutation` work by adding
    # the ThermoMPNN repo root to PYTHONPATH.
    thermo_root = str(thermompnn_dir)
    old_pp = env_sub.get("PYTHONPATH", "")
    env_sub["PYTHONPATH"] = thermo_root if not old_pp else f"{thermo_root}:{old_pp}"
    env_sub["WANDB_DISABLED"] = "true"
    env_sub["WANDB_MODE"] = "disabled"


    # Optional: pin GPU
    if args.gpu.strip():
        env_sub["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()
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

            # raw artifacts directory next to JSON:
            # results/<run>/thermompnn/<acc>/sample_<N>.thermompnn/...
            raw_dir = out_json.with_suffix("")  # drop .json
            results_raw_dir = Path(str(raw_dir) + ".thermompnn")

            if out_json.exists() and not args.overwrite:
                n_skip += 1
                continue

            t0 = time.perf_counter()
            status = "success"
            err_msg: Optional[str] = None
            predictions: Optional[Dict[str, Any]] = None
            raw_output_path: Optional[str] = None

            # temp working output dir
            work_dir = Path("eval-thermompnn/temp") / rec.accession / f"sample_{rec.sample_num}"
            work_out = work_dir / "out"
            ensure_dir(work_out)

            try:
                pdb_path = Path(rec.structure_path)
                if not pdb_path.exists():
                    raise FileNotFoundError(f"Structure file not found: {pdb_path}")

                chain = args.chain.strip() or infer_chain_from_accession(rec.accession)

                rc, so, se = run_custom_inference(
                    thermompnn_dir=thermompnn_dir,
                    pdb_path=pdb_path,
                    work_dir=work_out,
                    chain=chain,
                    model_path=model_path,
                    python_bin=sys.executable,
                    extra_args=None,
                    env=env_sub,
                )
                if rc != 0:
                    raise RuntimeError(f"ThermoMPNN custom_inference failed (rc={rc}). stderr: {se.strip() or '<empty>'}")

                # copy raw artifacts to results
                copied = collect_outputs_to_results(work_out, results_raw_dir)
                raw_output_path = str(results_raw_dir)

                # Try to parse a CSV (if any) to create a compact summary
                csvs = [p for p in copied if p.suffix.lower() == ".csv"]
                csv_summary = None
                if csvs:
                    # choose largest CSV as "primary"
                    primary = max(csvs, key=lambda p: p.stat().st_size)
                    csv_summary = summarize_ddg_csv(primary, top_k=args.top_k)

                predictions = {
                    "summary": csv_summary,
                    "raw_files": [str(p) for p in copied],
                }

            except Exception as e:
                status = "failed"
                err_msg = str(e)

            runtime = time.perf_counter() - t0

            out_obj = {
                "structure_id": rec.structure_id,
                "structure_path": rec.structure_path,
                "accession": rec.accession,
                "sample_num": rec.sample_num,
                "eval_type": "thermompnn",
                "status": status,
                "error": err_msg,
                "predictions": predictions,
                "raw_output_path": raw_output_path,
                "timestamp": utc_now_iso(),
                "runtime_seconds": float(runtime),
            }

            out_obj["tool"] = str(thermompnn_dir / "analysis" / "custom_inference.py")
            out_obj["model_path"] = str(model_path)
            if args.gpu.strip():
                out_obj["cuda_visible_devices"] = args.gpu.strip()

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
                logger.warning("FAIL %s (%s sample=%s): %s", rec.accession, rec.structure_id, rec.sample_num, out_obj["error"])

    logger.info("Done. success=%d failed=%d skipped=%d", n_ok, n_fail, n_skip)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

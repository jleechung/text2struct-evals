#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path so `import utils` works everywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.batch_utils import create_batches
from utils.io_utils import ensure_dir, iter_structures, load_manifest, output_json_path, write_json_atomic
from utils.schema import validate_output_json


@dataclass(frozen=True)
class Rec:
    structure_id: str
    structure_path: str
    accession: str
    sample_num: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("eval-clean")
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


# ----------------------------
# PDB -> sequence extraction
# ----------------------------

def _extract_chain_sequences(pdb_path: Path) -> Dict[str, str]:
    """
    Extract AA sequences per chain using BioPython PPBuilder (handles gaps reasonably).
    Returns: {chain_id: sequence} for chains with non-empty sequences.
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import PPBuilder
    except Exception as e:
        raise ImportError(
            "BioPython is required for CLEAN runner (pip/conda install biopython)."
        ) from e

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", str(pdb_path))

    ppb = PPBuilder()
    out: Dict[str, str] = {}

    # take first model
    models = list(structure.get_models())
    if not models:
        return out
    model = models[0]

    for chain in model.get_chains():
        chain_id = str(chain.id).strip()
        if not chain_id:
            continue
        peptides = ppb.build_peptides(chain)
        if not peptides:
            continue
        seq = "".join(str(pp.get_sequence()) for pp in peptides).strip()
        if seq:
            out[chain_id] = seq

    return out


def _select_chains(
    chain_to_seq: Dict[str, str],
    *,
    chain_policy: str,
    explicit_chain: str = "",
) -> List[Tuple[str, str]]:
    """
    Returns list of (chain_id, seq) to evaluate for this structure.
    chain_policy in {"longest","first","all"}.
    explicit_chain (if provided) overrides policy and selects only that chain id.
    """
    if explicit_chain.strip():
        c = explicit_chain.strip()
        if c not in chain_to_seq:
            raise ValueError(f"Requested chain '{c}' not found in PDB chains: {sorted(chain_to_seq.keys())}")
        return [(c, chain_to_seq[c])]

    items = [(c, s) for c, s in chain_to_seq.items() if s]
    if not items:
        return []

    if chain_policy == "all":
        # stable ordering
        return sorted(items, key=lambda x: x[0])

    if chain_policy == "first":
        return [sorted(items, key=lambda x: x[0])[0]]

    # default: longest
    # tie-break on chain id for determinism
    items_sorted = sorted(items, key=lambda x: (-len(x[1]), x[0]))
    return [items_sorted[0]]


def _apply_length_policy(
    seq: str,
    *,
    max_len: int,
    long_seq_policy: str,
) -> Tuple[str, bool, int]:
    """
    Returns (seq_used, truncated_flag, original_len).
    """
    orig = len(seq)
    if max_len <= 0:
        raise ValueError(f"max_len must be > 0, got {max_len}")
    if orig <= max_len:
        return seq, False, orig

    if long_seq_policy == "fail":
        raise ValueError(f"Sequence length {orig} exceeds max_len={max_len} (long_seq_policy=fail)")
    # truncate
    return seq[:max_len], True, orig


# ----------------------------
# CLEAN invocation + parsing
# ----------------------------

def run_clean_infer(
    clean_dir: Path,
    fasta_data: str,
    *,
    python_bin: str,
    env: Dict[str, str],
) -> Tuple[int, str, str]:
    """
    Runs CLEAN_infer_fasta.py in clean_dir.
    Assumes input FASTA is at clean_dir/data/<fasta_data>.fasta
    Produces results/<fasta_data>_maxsep.csv (usually).
    """
    script = clean_dir / "CLEAN_infer_fasta.py"
    if not script.exists():
        raise FileNotFoundError(f"CLEAN_infer_fasta.py not found: {script}")

    cmd = [python_bin, str(script), "--fasta_data", fasta_data]
    r = subprocess.run(
        cmd,
        cwd=str(clean_dir),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def find_maxsep_csv(clean_dir: Path, fasta_data: str) -> Path:
    """
    Preferred: results/<fasta_data>_maxsep.csv
    Fallback: any results/*<fasta_data>*maxsep*.csv (choose newest by mtime)
    """
    results_dir = clean_dir / "results"
    cand = results_dir / f"{fasta_data}_maxsep.csv"
    if cand.exists():
        return cand

    globbed = sorted(results_dir.glob(f"*{fasta_data}*maxsep*.csv"))
    if not globbed:
        globbed = sorted(results_dir.glob(f"*{fasta_data}*.csv"))

    if not globbed:
        raise FileNotFoundError(f"No maxsep CSV found under {results_dir} for fasta_data={fasta_data}")

    # pick most recently modified
    return max(globbed, key=lambda p: p.stat().st_mtime)


def parse_maxsep_csv(csv_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse CLEAN max-separation CSV rows of the form:
      <id>,EC:1.2.3.4/7.4553,EC:.../6.12,...
    Returns mapping: id -> list of {ec, maxsep}
    Sorted by maxsep descending (higher = better).
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 2:
            # might happen for "no prediction" rows
            continue
        qid = parts[0]
        preds: List[Dict[str, Any]] = []
        for tok in parts[1:]:
            # expected "EC:.../score"
            if "/" not in tok:
                continue
            left, right = tok.split("/", 1)
            left = left.strip()
            right = right.strip()
            if left.startswith("EC:"):
                left = left[len("EC:") :].strip()
            if not left:
                continue
            try:
                score = float(right)
            except Exception:
                continue
            preds.append({"ec": left, "maxsep": score})
        preds.sort(key=lambda d: float(d.get("maxsep", float("-inf"))), reverse=True)
        out[qid] = preds
    return out


def write_fasta(path: Path, records: List[Tuple[str, str]]) -> None:
    """
    records: list of (id, seq)
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rid, seq in records:
            f.write(f">{rid}\n")
            # wrap at 60 for readability
            for i in range(0, len(seq), 60):
                f.write(seq[i : i + 60] + "\n")


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--log_root", required=True)
    ap.add_argument("--batch_size", type=int, default=16)

    ap.add_argument(
        "--clean_dir",
        default=str(Path("eval-clean/vendor/CLEAN")),
        help="Path to CLEAN repo directory (vendored).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON outputs.")
    ap.add_argument("--allow_missing_files", action="store_true", help="Do not fail if some structure files are missing.")

    # GPU pinning (optional; CLEAN uses torch)
    ap.add_argument("--gpu", default="", help="If set, exports CUDA_VISIBLE_DEVICES=<gpu> for CLEAN call.")

    # Sequence selection / sizing
    ap.add_argument("--chain", default="", help="If set, use this chain ID only (overrides chain_policy).")
    ap.add_argument(
        "--chain_policy",
        default="longest",
        choices=["longest", "first", "all"],
        help="How to choose chain(s) if --chain is not provided.",
    )
    ap.add_argument("--max_len", type=int, default=1022, help="Max sequence length (ESM-1b limit is ~1022).")
    ap.add_argument(
        "--long_seq_policy",
        default="truncate",
        choices=["truncate", "fail"],
        help="What to do if sequence exceeds --max_len.",
    )

    # Output shaping
    ap.add_argument("--max_ec", type=int, default=10, help="Keep top-K EC predictions per chain.")
    ap.add_argument(
        "--keep_batch_files",
        action="store_true",
        help="Keep temporary batch FASTA + batch CSV inside CLEAN repo (otherwise cleaned up).",
    )

    args = ap.parse_args()

    run_name = args.run_name.strip()
    if not run_name:
        raise ValueError("--run_name must be non-empty")

    eval_name = "clean"
    log_path = Path(args.log_root) / run_name / eval_name / "run.log"
    logger = setup_logger(log_path)

    clean_dir = Path(args.clean_dir).expanduser().resolve()
    if not clean_dir.exists():
        raise FileNotFoundError(f"CLEAN dir not found: {clean_dir}")

    # Build env for subprocess
    env_sub = dict(os.environ)

    # Make imports robust for the CLEAN script:
    # - repo root (clean_dir)
    # - src/ (where CLEAN package lives)
    # - app/esm (local clone; harmless if unused because conda has fair-esm too)
    add_paths = [
        str(clean_dir),
        str(clean_dir / "src"),
        str(clean_dir / "app" / "esm"),
    ]
    old_pp = env_sub.get("PYTHONPATH", "")
    env_sub["PYTHONPATH"] = ":".join(add_paths + ([old_pp] if old_pp else []))

    if args.gpu.strip():
        env_sub["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()

    logger.info("Starting CLEAN eval")
    logger.info("manifest=%s run_name=%s output_root=%s", args.manifest, run_name, args.output_root)
    logger.info("clean_dir=%s", str(clean_dir))
    logger.info("batch_size=%d chain_policy=%s chain=%s", args.batch_size, args.chain_policy, args.chain or "<auto>")
    logger.info("max_len=%d long_seq_policy=%s max_ec=%d", args.max_len, args.long_seq_policy, args.max_ec)
    logger.info("CUDA_VISIBLE_DEVICES=%s", env_sub.get("CUDA_VISIBLE_DEVICES", ""))

    df = load_manifest(args.manifest, resolve_paths=True, allow_missing_files=args.allow_missing_files)
    all_recs: List[Rec] = [
        Rec(structure_id=sid, structure_path=spath, accession=acc, sample_num=sn)
        for (sid, spath, acc, sn) in iter_structures(df)
    ]

    # Skip existing outputs up front
    pending: List[Rec] = []
    n_skip = 0
    for rec in all_recs:
        out_json = output_json_path(args.output_root, run_name, eval_name, rec.accession, rec.sample_num)
        if out_json.exists() and not args.overwrite:
            n_skip += 1
            continue
        pending.append(rec)

    logger.info("Total=%d pending=%d skipped=%d", len(all_recs), len(pending), n_skip)

    batches = create_batches(pending, args.batch_size)
    logger.info("Created %d batches (batch_size=%d)", len(batches), args.batch_size)

    n_ok = 0
    n_fail = 0

    for bi, batch in enumerate(batches, start=1):
        logger.info("Batch %d/%d (%d structures)", bi, len(batches), len(batch))

        # Build per-batch FASTA records
        # We’ll use unique IDs so we can map predictions back.
        # For chain_policy=all, each structure can produce multiple query IDs.
        batch_uuid = uuid.uuid4().hex[:10]
        fasta_data = f"t2s_{run_name}_b{bi:04d}_{batch_uuid}"
        data_dir = clean_dir / "data"
        results_dir = clean_dir / "results"
        ensure_dir(data_dir)
        ensure_dir(results_dir)

        batch_fasta_path = data_dir / f"{fasta_data}.fasta"

        # Maps structure_id -> list of query ids in this batch
        struct_to_qids: Dict[str, List[str]] = {}
        # Metadata per query id for output shaping
        qmeta: Dict[str, Dict[str, Any]] = {}

        fasta_records: List[Tuple[str, str]] = []
        prep_errors: Dict[str, str] = {}

        # Prepare FASTA inputs
        for rec in batch:
            try:
                pdb_path = Path(rec.structure_path)
                if not pdb_path.exists():
                    raise FileNotFoundError(f"Structure file not found: {pdb_path}")

                chain_to_seq = _extract_chain_sequences(pdb_path)
                if not chain_to_seq:
                    raise ValueError("No protein sequence could be extracted from PDB (no peptides found).")

                selected = _select_chains(
                    chain_to_seq,
                    chain_policy=args.chain_policy,
                    explicit_chain=args.chain,
                )
                if not selected:
                    raise ValueError("No non-empty chain sequences after applying chain selection.")

                qids: List[str] = []
                for chain_id, seq in selected:
                    seq_used, truncated, orig_len = _apply_length_policy(
                        seq,
                        max_len=args.max_len,
                        long_seq_policy=args.long_seq_policy,
                    )

                    # Query ID must match what we’ll parse from the CSV (no commas!)
                    if args.chain_policy == "all" or args.chain.strip():
                        qid = f"{rec.structure_id}|{chain_id}"
                    else:
                        qid = rec.structure_id

                    # In rare cases, chain_policy=all could create duplicate qids if structure_id repeats.
                    # Add a suffix to ensure uniqueness while still being traceable.
                    if qid in qmeta:
                        qid = f"{qid}|dup{len(qmeta)}"

                    qids.append(qid)
                    qmeta[qid] = {
                        "structure_id": rec.structure_id,
                        "accession": rec.accession,
                        "sample_num": rec.sample_num,
                        "chain_id": chain_id,
                        "sequence_length": len(seq_used),
                        "sequence_truncated": bool(truncated),
                        "original_length": int(orig_len),
                    }
                    fasta_records.append((qid, seq_used))

                struct_to_qids[rec.structure_id] = qids

            except Exception as e:
                prep_errors[rec.structure_id] = str(e)

        # Write failed JSONs for any prep errors (and skip them in CLEAN call)
        for rec in batch:
            if rec.structure_id not in prep_errors:
                continue

            out_json = output_json_path(args.output_root, run_name, eval_name, rec.accession, rec.sample_num)
            t0 = time.perf_counter()

            out_obj: Dict[str, Any] = {
                "structure_id": rec.structure_id,
                "structure_path": rec.structure_path,
                "accession": rec.accession,
                "sample_num": rec.sample_num,
                "eval_type": "clean",
                "status": "failed",
                "error": prep_errors[rec.structure_id],
                "predictions": None,
                "raw_output_path": None,
                "timestamp": utc_now_iso(),
                "runtime_seconds": float(time.perf_counter() - t0),
            }
            out_obj["tool"] = str(clean_dir / "CLEAN_infer_fasta.py")

            try:
                validate_output_json(out_obj)
            except Exception as ve:
                out_obj["error"] = f"Schema validation failed: {ve}"

            write_json_atomic(out_json, out_obj)
            n_fail += 1

        # If nothing to run in this batch, continue
        if not fasta_records:
            logger.info("Batch %d: no runnable structures after prep (all failed)", bi)
            continue

        # Write batch FASTA
        write_fasta(batch_fasta_path, fasta_records)

        # Run CLEAN once for the batch
        t_batch0 = time.perf_counter()
        status_batch = "success"
        err_batch: Optional[str] = None
        maxsep_csv_path: Optional[Path] = None
        parsed: Dict[str, List[Dict[str, Any]]] = {}

        try:
            rc, so, se = run_clean_infer(
                clean_dir=clean_dir,
                fasta_data=fasta_data,
                python_bin=sys.executable,
                env=env_sub,
            )
            if rc != 0:
                raise RuntimeError(f"CLEAN inference failed (rc={rc}). stderr: {se.strip() or '<empty>'}")

            maxsep_csv_path = find_maxsep_csv(clean_dir, fasta_data)
            parsed = parse_maxsep_csv(maxsep_csv_path)

        except Exception as e:
            status_batch = "failed"
            err_batch = str(e)

        batch_runtime = time.perf_counter() - t_batch0

        # Optionally clean up batch files inside CLEAN repo
        if not args.keep_batch_files:
            try:
                if batch_fasta_path.exists():
                    batch_fasta_path.unlink()
            except Exception:
                pass
            try:
                if maxsep_csv_path is not None and maxsep_csv_path.exists():
                    maxsep_csv_path.unlink()
            except Exception:
                pass

        # Per-item runtime approximation (only among runnable query records)
        per_item_runtime = batch_runtime / max(1, len(fasta_records))

        # If batch failed, write failed JSON for remaining structures in this batch that were runnable
        if status_batch != "success":
            for rec in batch:
                if rec.structure_id in prep_errors:
                    continue  # already wrote
                out_json = output_json_path(args.output_root, run_name, eval_name, rec.accession, rec.sample_num)
                t0 = time.perf_counter()

                out_obj: Dict[str, Any] = {
                    "structure_id": rec.structure_id,
                    "structure_path": rec.structure_path,
                    "accession": rec.accession,
                    "sample_num": rec.sample_num,
                    "eval_type": "clean",
                    "status": "failed",
                    "error": f"Batch {bi} failed: {err_batch}",
                    "predictions": None,
                    "raw_output_path": None,
                    "timestamp": utc_now_iso(),
                    "runtime_seconds": float(time.perf_counter() - t0),
                }
                out_obj["tool"] = str(clean_dir / "CLEAN_infer_fasta.py")
                out_obj["batch_fasta_data"] = fasta_data

                try:
                    validate_output_json(out_obj)
                except Exception as ve:
                    out_obj["error"] = f"Schema validation failed: {ve}"

                write_json_atomic(out_json, out_obj)
                n_fail += 1

            logger.warning("Batch %d failed: %s", bi, err_batch)
            continue

        # Batch succeeded -> write per-structure results
        for rec in batch:
            if rec.structure_id in prep_errors:
                continue

            out_json = output_json_path(args.output_root, run_name, eval_name, rec.accession, rec.sample_num)

            # raw artifacts directory next to JSON:
            # results/<run>/clean/<acc>/sample_<N>.clean/...
            raw_dir_base = out_json.with_suffix("")  # drop .json
            results_raw_dir = Path(str(raw_dir_base) + ".clean")
            ensure_dir(results_raw_dir)

            # Prepare predictions per chain
            qids = struct_to_qids.get(rec.structure_id, [])
            chain_preds: List[Dict[str, Any]] = []
            lines_out: List[str] = []
            any_pred = False

            for qid in qids:
                preds = parsed.get(qid, [])
                meta = qmeta.get(qid, {})
                topk = preds[: max(0, int(args.max_ec))] if preds else []
                if topk:
                    any_pred = True

                chain_preds.append(
                    {
                        "query_id": qid,
                        "chain_id": meta.get("chain_id"),
                        "sequence_length": meta.get("sequence_length"),
                        "sequence_truncated": meta.get("sequence_truncated"),
                        "original_length": meta.get("original_length"),
                        "ec_topk": topk,
                        "top_ec": (topk[0]["ec"] if topk else None),
                        "top_maxsep": (topk[0]["maxsep"] if topk else None),
                    }
                )

                # Save the raw CSV line for this qid if possible (reconstruct minimal)
                if preds:
                    toks = [f"EC:{p['ec']}/{p['maxsep']}" for p in preds]
                    lines_out.append(",".join([qid] + toks))

            # Write raw artifacts: input fasta for this structure + output lines
            # input.fasta includes only this structure's query ids
            struct_fasta_records: List[Tuple[str, str]] = []
            for qid in qids:
                # recover sequence by searching fasta_records in this batch
                # (small O(n) is fine for batch sizes)
                for rid, seq in fasta_records:
                    if rid == qid:
                        struct_fasta_records.append((rid, seq))
                        break
            write_fasta(results_raw_dir / "input.fasta", struct_fasta_records)

            (results_raw_dir / "output_lines.txt").write_text(
                ("\n".join(lines_out) + ("\n" if lines_out else "")),
                encoding="utf-8",
            )

            predictions: Dict[str, Any] = {
                "chain_policy": args.chain_policy,
                "explicit_chain": args.chain.strip() or None,
                "num_query_chains": len(qids),
                "chains": chain_preds,
                "had_any_prediction": bool(any_pred),
            }

            out_obj: Dict[str, Any] = {
                "structure_id": rec.structure_id,
                "structure_path": rec.structure_path,
                "accession": rec.accession,
                "sample_num": rec.sample_num,
                "eval_type": "clean",
                "status": "success",
                "error": None,
                "predictions": predictions,
                "raw_output_path": str(results_raw_dir),
                "timestamp": utc_now_iso(),
                "runtime_seconds": float(per_item_runtime),
            }
            out_obj["tool"] = str(clean_dir / "CLEAN_infer_fasta.py")
            out_obj["batch_fasta_data"] = fasta_data

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

        logger.info(
            "Batch %d done: runnable_queries=%d runtime=%.3fs per_item=%.3fs",
            bi,
            len(fasta_records),
            batch_runtime,
            per_item_runtime,
        )

    logger.info("Done. success=%d failed=%d skipped=%d", n_ok, n_fail, n_skip)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

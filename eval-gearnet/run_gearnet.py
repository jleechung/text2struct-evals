#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTEINA_ROOT = (REPO_ROOT / "eval-gearnet" / "vendor" / "proteina").resolve()
if PROTEINA_ROOT.exists() and str(PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTEINA_ROOT))

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
    logger = logging.getLogger("eval-gearnet")
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


def default_data_path() -> Path:
    env = os.environ.get("DATA_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "eval-gearnet" / "cache" / "proteina_additional_files").resolve()


def _load_cath_names(path: Path) -> Optional[Dict[str, Dict[str, Dict[str, str]]]]:
    """
    Expects cath_node_names.json written by build_cath_names_cache.py:
      {"meta": {...}, "levels": {"C": {"1": {"name":..}}, "A": {...}, "T": {...}}}
    Returns the "levels" dict or None if missing/unreadable.
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        levels = obj.get("levels")
        if isinstance(levels, dict):
            return levels
    except Exception:
        pass
    return None


def load_label_mapping(label_mapping_path: Path) -> Dict[str, Dict[int, str]]:
    """
    cath_label_mapping.pt: {"C": {label->idx}, "A": {label->idx}, "T": {label->idx}}
    Return: {"C": {idx->label}, ...}
    """
    import torch

    obj = torch.load(str(label_mapping_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in cath_label_mapping.pt, got {type(obj).__name__}")

    inv: Dict[str, Dict[int, str]] = {}
    for level in ("C", "A", "T"):
        m = obj.get(level)
        if m is None:
            continue
        if not isinstance(m, dict):
            raise ValueError(f"Expected dict for mapping[{level}], got {type(m).__name__}")
        inv[level] = {int(v): str(k) for k, v in m.items()}
    return inv


def topk_from_logits_row(logits_row, k: int, idx_to_label: Optional[Dict[int, str]]):
    import torch

    probs = torch.softmax(logits_row, dim=-1)
    kk = int(min(max(k, 1), probs.shape[-1]))
    top_probs, top_idx = torch.topk(probs, kk, dim=-1)

    top_probs_l = top_probs.detach().cpu().tolist()
    top_idx_l = top_idx.detach().cpu().tolist()
    top_logits_l = logits_row[top_idx].detach().cpu().tolist()

    out = []
    for rank, (ix, p, lg) in enumerate(zip(top_idx_l, top_probs_l, top_logits_l), start=1):
        rec = {"rank": rank, "idx": int(ix), "prob": float(p), "logit": float(lg)}
        if idx_to_label is not None and int(ix) in idx_to_label:
            rec["label"] = idx_to_label[int(ix)]
        out.append(rec)
    return out, probs


class GearNetDataset:
    """
    Similar to Proteina's DatasetWrapper, but returns a dict so we can:
      - keep per-item metadata
      - catch per-item graph build errors without killing the run
    """

    def __init__(self, recs: List[Rec]) -> None:
        self.recs = recs

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.recs[idx]
        pdb_path = Path(rec.structure_path)

        try:
            if not pdb_path.exists():
                raise FileNotFoundError(f"Structure file not found: {pdb_path}")

            # Import inside worker for torch DataLoader multiprocessing compatibility
            import torch
            from graphein_utils.graphein_utils import protein_to_pyg
            from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

            graph = protein_to_pyg(str(pdb_path), deprotonate=False)

            coord_mask = graph.coords != 1e-5
            graph.coord_mask = coord_mask[..., 0]

            graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :].float()
            graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

            graph.node_id = torch.arange(graph.coords.shape[0]).unsqueeze(-1)

            # Attach rec index as a tensor so it survives batching if needed later
            graph.rec_idx = torch.tensor([idx], dtype=torch.long)

            return {"ok": True, "rec_idx": idx, "graph": graph, "error": None}

        except Exception as e:
            return {"ok": False, "rec_idx": idx, "graph": None, "error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Proteina GearNet fold classifier over a manifest of PDB paths.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--output_root", default="results")
    ap.add_argument("--log_root", default="logs")

    # Here batch_size is true model batch size (as in Proteina)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_missing_files", action="store_true")
    ap.add_argument("--gpu", default="", help='GPU id for CUDA_VISIBLE_DEVICES (e.g. "0").')

    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--data_path", default="", help="Override DATA_PATH (proteina_additional_files root).")
    ap.add_argument("--ckpt_path", default="", help="Override checkpoint (.pth) path.")
    ap.add_argument("--label_mapping_path", default="", help="Override cath_label_mapping.pt path.")
    ap.add_argument(
        "--cath_names_path",
        default="",
        help="Optional path to cath_node_names.json. If present, adds label_name to topk entries.",
    )

    ap.add_argument("--bb_model", action="store_true", help="Use backbone model (requires gearnet.pth). Default is CA-only.")

    args = ap.parse_args()

    # Set visibility BEFORE importing torch to avoid odd CUDA device ordering.
    if args.gpu.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu.strip()

    import torch
    from torch_geometric.data import Batch
    from torch.utils.data import DataLoader
    from proteinfoundation.metrics.gearnet_utils import NoTrainBBGearNet, NoTrainCAGearNet

    eval_name = "gearnet"
    run_name = args.run_name.strip()
    log_path = Path(args.log_root) / run_name / eval_name / "run.log"
    logger = setup_logger(log_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.data_path).expanduser().resolve() if args.data_path.strip() else default_data_path()
    if args.ckpt_path.strip():
        ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    else:
        model_name = "gearnet.pth" if args.bb_model else "gearnet_ca.pth"
        ckpt_path = data_path / "metric_factory" / "model_weights" / model_name

    if args.label_mapping_path.strip():
        label_mapping_path = Path(args.label_mapping_path).expanduser().resolve()
    else:
        label_mapping_path = data_path / "pdb_raw" / "cath_label_mapping.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not label_mapping_path.exists():
        raise FileNotFoundError(f"Label mapping not found: {label_mapping_path}")
    
    default_names = (REPO_ROOT / "eval-gearnet" / "cache" / "cath_node_names.json").resolve()
    cath_names_path = Path(args.cath_names_path).expanduser().resolve() if args.cath_names_path.strip() else default_names
    cath_names = _load_cath_names(cath_names_path) if cath_names_path.exists() else None

    if cath_names is not None:
        logger.info("Loaded CATH names: %s", str(cath_names_path))
    else:
        logger.info("CATH names not found (optional): %s", str(cath_names_path))

    idx_to_label = load_label_mapping(label_mapping_path)

    logger.info("Starting GearNet eval")
    logger.info("manifest=%s run_name=%s", args.manifest, run_name)
    logger.info("proteina_root=%s", str(PROTEINA_ROOT))
    logger.info("device=%s cuda_available=%s", device, torch.cuda.is_available())
    logger.info("DATA_PATH=%s", str(data_path))
    logger.info("ckpt_path=%s", str(ckpt_path))
    logger.info("label_mapping_path=%s", str(label_mapping_path))
    logger.info("batch_size=%d num_workers=%d", args.batch_size, args.num_workers)

    # Load model once (Proteina-style)
    model = NoTrainBBGearNet(str(ckpt_path)) if args.bb_model else NoTrainCAGearNet(str(ckpt_path))
    model = model.to(device).eval()

    df = load_manifest(args.manifest, resolve_paths=True, allow_missing_files=args.allow_missing_files)
    all_recs: List[Rec] = [
        Rec(
            structure_id=str(sid),
            structure_path=str(spath),
            accession=str(acc),
            sample_num=int(sn),
        )
        for (sid, spath, acc, sn) in iter_structures(df)
    ]

    # Skip existing outputs up front (so dataloader only processes pending)
    pending: List[Rec] = []
    n_skip = 0
    for rec in all_recs:
        out_json = output_json_path(args.output_root, run_name, eval_name, rec.accession, rec.sample_num)
        if out_json.exists() and not args.overwrite:
            n_skip += 1
            continue
        pending.append(rec)

    logger.info("Total=%d pending=%d skipped=%d", len(all_recs), len(pending), n_skip)

    # DataLoader batching like Proteina (but with safe collate)
    dataset = GearNetDataset(pending)

    # Collate returns list[dict] so we can filter failures before batching into Batch.from_data_list(...)
    def collate_keep_list(items):
        return items

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_keep_list,
        pin_memory=torch.cuda.is_available(),
    )

    n_ok = 0
    n_fail = 0

    for bi, items in enumerate(loader, start=1):
        t_batch0 = time.perf_counter()

        # 1) write outputs for any graph-build failures in this batch
        ok_items = [it for it in items if it["ok"] and it["graph"] is not None]
        bad_items = [it for it in items if not it["ok"]]

        for it in bad_items:
            rec = pending[int(it["rec_idx"])]
            out_json = output_json_path(args.output_root, run_name, eval_name, rec.accession, rec.sample_num)

            out_obj: Dict[str, Any] = {
                "structure_id": rec.structure_id,
                "structure_path": rec.structure_path,
                "accession": rec.accession,
                "sample_num": rec.sample_num,
                "eval_type": "gearnet",
                "status": "failed",
                "error": it["error"],
                "predictions": None,
                "raw_output_path": None,
                "timestamp": utc_now_iso(),
                "runtime_seconds": 0.0,
                "tool": "proteina",
                "tool_component": "NoTrainBBGearNet" if args.bb_model else "NoTrainCAGearNet",
                "ckpt_path": str(ckpt_path),
                "label_mapping_path": str(label_mapping_path),
                "data_path": str(data_path),
                "device": str(device),
            }

            try:
                validate_output_json(out_obj)
            except Exception as ve:
                out_obj["error"] = f"Schema validation failed: {ve}"

            write_json_atomic(out_json, out_obj)
            n_fail += 1

        # 2) If nothing OK, continue
        if not ok_items:
            logger.info("Batch %d: all %d failed in graph build", bi, len(items))
            continue

        graphs = [it["graph"] for it in ok_items]
        rec_indices = [int(it["rec_idx"]) for it in ok_items]

        # 3) Forward pass on the batch (Proteina-style)
        batch_obj = Batch.from_data_list(graphs).to(device)

        with torch.no_grad():
            out = model(batch_obj)

        # Expected keys per gearnet_utils.py forward()
        feat = out.get("protein_feature")
        pred_T = out.get("pred_T")
        pred_A = out.get("pred_A")
        pred_C = out.get("pred_C")

        if feat is None or pred_T is None or pred_A is None or pred_C is None:
            # If this happens, fail the whole batch's OK items (rare)
            err = "Model output missing required keys (protein_feature / pred_T / pred_A / pred_C)"
            for rec_idx in rec_indices:
                rec = pending[rec_idx]
                out_json = output_json_path(args.output_root, run_name, eval_name, rec.accession, rec.sample_num)
                out_obj = {
                    "structure_id": rec.structure_id,
                    "structure_path": rec.structure_path,
                    "accession": rec.accession,
                    "sample_num": rec.sample_num,
                    "eval_type": "gearnet",
                    "status": "failed",
                    "error": err,
                    "predictions": None,
                    "raw_output_path": None,
                    "timestamp": utc_now_iso(),
                    "runtime_seconds": float(time.perf_counter() - t_batch0),
                    "tool": "proteina",
                }
                try:
                    validate_output_json(out_obj)
                except Exception as ve:
                    out_obj["error"] = f"Schema validation failed: {ve}"
                write_json_atomic(out_json, out_obj)
                n_fail += 1
            continue

        # Per-item runtime approximation
        batch_runtime = time.perf_counter() - t_batch0
        per_item_runtime = batch_runtime / max(1, len(rec_indices))

        # 4) Write per-structure outputs
        for j, rec_idx in enumerate(rec_indices):
            rec = pending[rec_idx]

            out_json = output_json_path(args.output_root, run_name, eval_name, rec.accession, rec.sample_num)
            raw_pt = out_json.with_suffix(".gearnet.pt")
            ensure_dir(out_json.parent)

            try:
                # Slice each item out of the batch output
                feat_j = feat[j : j + 1]           # [1, 512]
                predT_j = pred_T[j]                # [1336]
                predA_j = pred_A[j]                # [43]
                predC_j = pred_C[j]                # [5]

                top_T, probs_T = topk_from_logits_row(predT_j, args.top_k, idx_to_label.get("T"))
                top_A, probs_A = topk_from_logits_row(predA_j, args.top_k, idx_to_label.get("A"))
                top_C, probs_C = topk_from_logits_row(predC_j, args.top_k, idx_to_label.get("C"))

                def enrich(level: str, topk_list: list[dict]):
                    if cath_names is None:
                        return
                    m = cath_names.get(level, {})
                    for r in topk_list:
                        lab = r.get("label")
                        if isinstance(lab, str) and lab in m:
                            info = m[lab]
                            if "name" in info and info["name"]:
                                r["label_name"] = info["name"]
                            if "example_domain" in info and info["example_domain"]:
                                r["label_example_domain"] = info["example_domain"]

                enrich("T", top_T)
                enrich("A", top_A)
                enrich("C", top_C)

                # Save raw tensors for this structure
                torch.save(
                    {
                        "protein_feature": feat_j.detach().cpu(),
                        "pred_T_logits": predT_j.detach().cpu(),
                        "pred_A_logits": predA_j.detach().cpu(),
                        "pred_C_logits": predC_j.detach().cpu(),
                        "pred_T_probs": probs_T.detach().cpu(),
                        "pred_A_probs": probs_A.detach().cpu(),
                        "pred_C_probs": probs_C.detach().cpu(),
                        "topk": {"T": top_T, "A": top_A, "C": top_C},
                        "meta": {
                            "bb_model": bool(args.bb_model),
                            "ckpt_path": str(ckpt_path),
                            "label_mapping_path": str(label_mapping_path),
                            "data_path": str(data_path),
                        },
                    },
                    str(raw_pt),
                )

                emb_norm = float(torch.norm(feat_j[0]).item())

                predictions = {
                    "bb_model": bool(args.bb_model),
                    "levels": {
                        "T": {"num_classes": int(pred_T.shape[-1]), "topk": top_T},
                        "A": {"num_classes": int(pred_A.shape[-1]), "topk": top_A},
                        "C": {"num_classes": int(pred_C.shape[-1]), "topk": top_C},
                    },
                    "embedding_dim": int(feat_j.shape[-1]),
                    "embedding_norm": emb_norm,
                }

                out_obj: Dict[str, Any] = {
                    "structure_id": rec.structure_id,
                    "structure_path": rec.structure_path,
                    "accession": rec.accession,
                    "sample_num": rec.sample_num,
                    "eval_type": "gearnet",
                    "status": "success",
                    "error": None,
                    "predictions": predictions,
                    "raw_output_path": str(raw_pt),
                    "timestamp": utc_now_iso(),
                    "runtime_seconds": float(per_item_runtime),
                    "tool": "proteina",
                    "tool_component": "NoTrainBBGearNet" if args.bb_model else "NoTrainCAGearNet",
                    "ckpt_path": str(ckpt_path),
                    "label_mapping_path": str(label_mapping_path),
                    "data_path": str(data_path),
                    "device": str(device),
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

            except Exception as e:
                out_obj = {
                    "structure_id": rec.structure_id,
                    "structure_path": rec.structure_path,
                    "accession": rec.accession,
                    "sample_num": rec.sample_num,
                    "eval_type": "gearnet",
                    "status": "failed",
                    "error": str(e),
                    "predictions": None,
                    "raw_output_path": None,
                    "timestamp": utc_now_iso(),
                    "runtime_seconds": float(per_item_runtime),
                    "tool": "proteina",
                }
                try:
                    validate_output_json(out_obj)
                except Exception as ve:
                    out_obj["error"] = f"Schema validation failed: {ve}"
                write_json_atomic(out_json, out_obj)
                n_fail += 1

        logger.info(
            "Batch %d done: size=%d ok=%d fail(graph)=%d runtime=%.3fs",
            bi,
            len(items),
            len(ok_items),
            len(bad_items),
            batch_runtime,
        )

    logger.info("Done. success=%d failed=%d skipped=%d", n_ok, n_fail, n_skip)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

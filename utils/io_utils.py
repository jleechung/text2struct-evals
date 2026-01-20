"""I/O utilities for manifest parsing and file operations.

Manifest columns (assumed header row):
- text_description, structure_path, accession, sample_num

Path convention (Option B):
- results/<run_name>/<eval_name>/<accession>/sample_<N>.json  (no zero padding)
- logs/<run_name>/<eval_name>/...

structure_path is the *only* locator for reading structures.
structure_id exists for stable identification across runs/evals.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

import pandas as pd


ManifestPath = Union[str, Path]
DEFAULT_MANIFEST_COLS = ("text_description", "structure_path", "accession", "sample_num")


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Union[str, Path]) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(
    path: Union[str, Path],
    data: dict,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """Write JSON atomically (temp file + replace) to avoid partial writes."""
    p = Path(path)
    ensure_dir(p.parent)

    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(p.parent), encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=indent, sort_keys=sort_keys)
        tmp.write("\n")
        tmp_path = Path(tmp.name)

    os.replace(tmp_path, p)


def validate_structure_path(path: Union[str, Path]) -> bool:
    try:
        return Path(path).expanduser().exists()
    except Exception:
        return False


def _coerce_int_series(series: pd.Series, col: str) -> pd.Series:
    try:
        s = pd.to_numeric(series, errors="raise").astype(int)
    except Exception as e:
        raise ValueError(f"Manifest column '{col}' must be integer-like. Error: {e}") from e
    return s


def _resolve_path_value(value: object, *, manifest_dir: Path) -> str:
    """Resolve a structure_path value to an absolute path string.

    - Expands env vars and ~
    - If relative, resolves relative to the manifest file directory (not CWD)
    """
    if value is None:
        return ""

    s = str(value).strip()
    if not s:
        return ""

    s = os.path.expandvars(os.path.expanduser(s))
    p = Path(s)
    if not p.is_absolute():
        p = (manifest_dir / p).resolve()
    return str(p)


def load_manifest(
    manifest_path: ManifestPath,
    *,
    required_cols: Sequence[str] = DEFAULT_MANIFEST_COLS,
    resolve_paths: bool = True,
    allow_missing_files: bool = True,
) -> pd.DataFrame:
    mp = Path(manifest_path)
    if not mp.exists():
        raise FileNotFoundError(f"Manifest not found: {mp}")

    df = pd.read_csv(mp)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Manifest missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["accession"] = df["accession"].astype(str).str.strip()
    df["sample_num"] = _coerce_int_series(df["sample_num"], "sample_num")

    if resolve_paths:
        manifest_dir = mp.parent.resolve()
        df["structure_path"] = df["structure_path"].apply(
            lambda v: _resolve_path_value(v, manifest_dir=manifest_dir)
        )

    if not allow_missing_files:
        missing_paths = df.loc[~df["structure_path"].apply(validate_structure_path), "structure_path"]
        if len(missing_paths) > 0:
            sample = missing_paths.head(20).tolist()
            raise FileNotFoundError(
                f"{len(missing_paths)} structure_path entries do not exist. "
                f"First {len(sample)}: {sample}"
            )

    return df


def make_structure_id(accession: str, sample_num: int, *, width: int = 2) -> str:
    """Stable identifier (useful for logging/merging across runs)."""
    acc = str(accession).strip()
    sn = int(sample_num)
    return f"{acc}_structure_{sn:0{width}d}"


def make_sample_filename(sample_num: int, *, prefix: str = "sample_") -> str:
    """Filename for per-sample outputs: sample_<N>.json (no zero padding)."""
    sn = int(sample_num)
    return f"{prefix}{sn}.json"


def output_json_path(
    output_root: Union[str, Path],
    run_name: str,
    eval_name: str,
    accession: str,
    sample_num: int,
) -> Path:
    """Option B output path:
    results/<run_name>/<eval_name>/<accession>/sample_<N>.json
    """
    root = Path(output_root)
    run = str(run_name).strip()
    ev = str(eval_name).strip()
    acc = str(accession).strip()

    if not run:
        raise ValueError("run_name must be non-empty")
    if not ev:
        raise ValueError("eval_name must be non-empty")
    if not acc:
        raise ValueError("accession must be non-empty")

    fname = make_sample_filename(sample_num)
    return root / run / ev / acc / fname


def iter_structures(df: pd.DataFrame) -> Iterable[Tuple[str, str, str, int]]:
    """Yield (structure_id, structure_path, accession, sample_num)."""
    for _, row in df.iterrows():
        accession = str(row["accession"]).strip()
        sample_num = int(row["sample_num"])
        structure_path = str(row["structure_path"])
        structure_id = make_structure_id(accession, sample_num)
        yield (structure_id, structure_path, accession, sample_num)

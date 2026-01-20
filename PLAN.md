
# Protein Structure Evaluation Pipeline - Implementation Plan

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Standard Interfaces](#standard-interfaces)
5. [Output Schema](#output-schema)
6. [Implementation Order](#implementation-order)
7. [Detailed Implementation Steps](#detailed-implementation-steps)
8. [Testing & Validation](#testing--validation)
9. [Appendix](#appendix)

---

## Project Overview

### Goal
Build a scalable evaluation pipeline for text-conditioned protein structure generation.

**Step 1 (now)**: Run 5 evaluation tools on generated protein structures
- Input: protein structures (PDB files) + text descriptions
- Output: per-structure JSON outputs + raw tool artifacts

**Step 2 (future)**: Compare predictions to original text prompts

### Scale & Resources
- Initial scale: ~10,000 structures (10 samples × 1,000 proteins)
- Hardware: H200 GPUs available on cluster
- Execution: SLURM (cluster), plus lightweight shell scripts for interactive testing

### Evaluation Tools

| Tool | Type | GPU | Speed | Primary Metric |
|------|------|-----|-------|----------------|
| DSSP (mkdssp) | Structural | No | <1s | Secondary structure assignment |
| Proteina | Structural | Yes | ~5s | CATH fold classification |
| P2Rank | Functional | No | <1s | Binding site prediction |
| CLEAN | Functional | Yes | <1s | EC number prediction |
| ThermoMPNN | Functional | Yes | ~10s | Stability / ΔΔG prediction |

**Implementation priority**: DSSP → P2Rank → Proteina → CLEAN → ThermoMPNN

---

## Architecture

### Design Principles

1. **Modularity**
   - Each evaluation lives in its own module directory: `eval-<name>/`
   - Each module has its own conda environment (separate, reproducible)
   - Modules share a small `utils/` library and a common output schema

2. **Consistency**
   - Standard CLI: `--manifest`, `--run_name`, `--output_root`, `--log_root`, `--batch_size`
   - Standard JSON schema for all per-structure outputs
   - Fail-gracefully: one bad structure writes a failed JSON and does not halt the run

3. **Scalability**
   - Batch processing inside each eval runner
   - Run evals independently across nodes/GPUs (“eval-first”)

4. **Robustness**
   - Always emit a JSON result (success or failed)
   - Persist raw tool outputs next to JSON for debugging

### Processing Strategy

**Eval-first approach** (recommended):

```

For each evaluation tool (possibly on different nodes):
Load manifest
Batch structures
For each structure:
Run tool
Write JSON + raw artifacts

```

---

## Directory Structure

`export BASE_PATH=/gscratch/h2lab/jxlee`

```

$BASE_PATH/text2struct-evals/
├── scripts/
│   ├── eval-dssp.sh                 # simple local/interactive test runner
│   └── eval-*.slurm                 # SLURM scripts (one per eval)
├── envs/
│   ├── core.yml                     # shared deps (pandas/jsonschema)
│   ├── dssp.yml                     # per-eval env specs (example)
│   └── ...
├── results/
│   └── <run_name>/
│       └── <eval_name>/
│           └── <accession>/
│               ├── sample_<N>.json
│               └── sample_<N>.<raw_ext>
├── logs/
│   └── <run_name>/
│       └── <eval_name>/
│           └── run.log
├── test_data/
│   └── manifest.csv                 # test manifest; structures can live anywhere
├── utils/
│   ├── **init**.py
│   ├── io_utils.py                  # manifest parsing, JSON I/O, output path helpers
│   ├── batch_utils.py               # batching
│   └── schema.py                    # JSON schema validation
├── eval-dssp/
│   ├── environment.yml              # optional: copy of envs/dssp.yml for convenience
│   ├── run_dssp.py
│   └── temp/
├── eval-p2rank/
├── eval-proteina/
├── eval-clean/
└── eval-thermompnn/

````

### Key Design Decisions

**Results and logs are organized by run**:
- Results: `results/<run_name>/<eval_name>/<accession>/sample_<N>.json`
- Logs: `logs/<run_name>/<eval_name>/run.log`

This supports multiple runs over the same accession/sample indices, e.g. `baseline` vs `model_v1`.

**Test data**:
- `test_data/` may contain just a manifest pointing to any valid structure paths.

---

## Standard Interfaces

### Manifest Format

CSV with header row:

```csv
text_description,structure_path,accession,sample_num
"Design a mainly-alpha oxidoreductase enzyme","/path/to/protein_001/sample_0.pdb","protein_001","0"
"Design a mainly-alpha oxidoreductase enzyme","/path/to/protein_001/sample_1.pdb","protein_001","1"
...
````

Notes:

* `structure_path` can be absolute or relative to the **manifest directory** (wrappers resolve relative paths).
* `sample_num` is integer-like.

### Python Script Interface (all eval runners)

```bash
python eval-<eval>/run_<eval>.py \
  --manifest /path/to/manifest.csv \
  --run_name baseline \
  --output_root /path/to/text2struct-evals/results \
  --log_root /path/to/text2struct-evals/logs \
  --batch_size 128 \
  [--overwrite]
```

Tool-specific flags are allowed (e.g. `--mkdssp_bin`, `--gpu_id`, etc.).

**Import convention**: eval runners must be able to import `utils/` reliably on the cluster.

* Recommended: in each runner, add repo root to `sys.path` using `Path(__file__).resolve().parents[1]`.

### Shell Scripts (testing)

For interactive testing, prefer a minimal shell wrapper per eval, e.g. `scripts/eval-dssp.sh`, that:

* sets `BASE_PATH`
* activates conda env
* calls the Python runner with defaults

### SLURM Script Template

Some cluster restrictions:

* Jobs must request at least one GPU
* Max runtime is 24h

Template (paths updated for run-based outputs):

```bash
#!/usr/bin/env bash
#SBATCH --job-name=eval-{eval}
#SBATCH --qos=normal
#SBATCH --account=jxlee
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=$BASE_PATH/text2struct-evals/logs/slurm_%j.out
#SBATCH --error=$BASE_PATH/text2struct-evals/logs/slurm_%j.err

BASE_PATH=/gscratch/h2lab/jxlee
BASE_DIR=$BASE_PATH/text2struct-evals

# conda
source $BASE_PATH/miniconda3/etc/profile.d/conda.sh
conda activate text2struct-{eval}

RUN_NAME=baseline
MANIFEST=/path/to/manifest.csv

python $BASE_DIR/eval-{eval}/run_{eval}.py \
  --manifest $MANIFEST \
  --run_name $RUN_NAME \
  --output_root $BASE_DIR/results \
  --log_root $BASE_DIR/logs \
  --batch_size 128
```

---

## Output Schema

All evaluations write one JSON per structure:

* JSON path: `results/<run_name>/<eval_name>/<accession>/sample_<N>.json`
* Raw artifact path: next to JSON (extension depends on tool)

### Standard JSON Format

```json
{
  "structure_id": "protein_001_structure_00",
  "structure_path": "/abs/path/to/sample_0.pdb",
  "accession": "protein_001",
  "sample_num": 0,
  "eval_type": "dssp",
  "status": "success",
  "error": null,
  "predictions": { },
  "raw_output_path": "/abs/path/to/sample_0.dssp",
  "timestamp": "2026-01-16T00:00:00+00:00",
  "runtime_seconds": 0.12
}
```

Notes:

* `structure_id` is a **stable identifier** (useful for joins/aggregation); it is not used for locating input files.
* The input structure is always identified by `structure_path`.

### Error Handling

If evaluation fails, write:

* `status: "failed"`
* `error: <message>`
* `predictions: null`

Raw artifact may still be present if the tool produced partial output.

### Evaluation-Specific Predictions

Keep eval-specific fields under `predictions`.

Example (DSSP):

* `secondary_structure`: full per-residue DSSP string
* `counts`: dict of SS symbol counts
* `residues`: per-residue records (chain, aa, ss, accessibility, etc.)

(Other eval prediction schemas TBD per module; they can evolve without breaking top-level schema.)

---

## Implementation Order

### Phase 1: Infrastructure

1. Create base repo structure (scripts/envs/utils/eval-* dirs)
2. Implement shared utilities:

   * `utils/io_utils.py` (manifest parsing + output path helper)
   * `utils/batch_utils.py`
   * `utils/schema.py` (jsonschema validation)
3. Create a **core env** spec (`envs/core.yml`) containing shared deps (at minimum: pandas, jsonschema).

### Phase 2: DSSP (template module)

1. Create per-eval env spec:

   * `envs/dssp.yml` (can be created by cloning core + adding `dssp`)
2. Implement `eval-dssp/run_dssp.py`:

   * Calls `mkdssp --output-format dssp input.pdb output.dssp`
   * Parses classic DSSP output
   * Writes JSON + `.dssp` raw artifact to run-based results layout
   * Writes logs to `logs/<run>/dssp/run.log`
3. Add a simple test script: `scripts/eval-dssp.sh`

### Remaining Phases

Repeat the DSSP pattern for P2Rank → Proteina → CLEAN → ThermoMPNN.

---

## Detailed Implementation Steps

### Shared utilities (already implemented)

* `load_manifest()` resolves relative structure paths relative to manifest directory.
* `output_json_path()` enforces: `results/<run>/<eval>/<accession>/sample_<N>.json`.

### New evaluation module checklist

1. Create module directory: `eval-<name>/`
2. Create env spec in `envs/<name>.yml` (and optionally copy to `eval-<name>/environment.yml`)
3. Implement `eval-<name>/run_<name>.py` using the DSSP runner pattern:

   * CLI: `--manifest --run_name --output_root --log_root --batch_size --overwrite`
   * Tool invocation (subprocess / python API)
   * Parse tool outputs
   * Emit JSON + raw artifacts

---

## Testing & Validation

### Local / interactive test

```bash
bash scripts/eval-dssp.sh
# Writes to results/testrun/... and logs/testrun/...
```

### Output checks

* Confirm expected file counts under `results/<run>/<eval>/...`
* Spot-check `status`, `predictions`, and that `raw_output_path` exists for successes
* Validate schema via `utils/schema.py` (done inside each runner)

---

## Appendix

### Environments

* `envs/core.yml`: shared dependencies used by all eval runners.
* `envs/<eval>.yml`: per-eval environment (may be created by cloning core and adding tool deps).

Example (DSSP):

```yaml
name: text2struct-dssp
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pandas
  - jsonschema
  - dssp
```

### Notes

* Cluster policy requires requesting a GPU even for CPU-only tools; keep runner code CPU-safe.

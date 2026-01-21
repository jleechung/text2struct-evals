# text2struct-evals

This repo runs a set of evaluators over generated protein structures listed in an input manifest CSV.

Evaluators:
- **DSSP** (secondary structure via `mkdssp`)
- **P2Rank** (binding pocket prediction)
- **ThermoMPNN** (stability / ddG-style scoring)
- **GearNet / Proteina** (CATH fold classification)
- **CLEAN** (enzyme EC prediction from sequence)

## Prereqs

- `conda` installed and available in your shell
- You are running from a machine/node that can access the internet for setup steps (downloads/clones)
- For GPU-heavy evals (GearNet, ThermoMPNN, CLEAN), use a GPU node

---

## 1. Create all conda envs

Each evaluator has a separate conda env. Set these up from the repo root:

```bash
bash scripts/setup-envs.sh
````

This creates/updates all environments defined under `envs/`.

## 2. Setup each evaluator

Run these once (they clone/download external dependencies into `eval-*/vendor` and caches).

```bash
bash scripts/setup-evals.sh
````

## 3. Prepare an input manifest

Each evaluator expects as input a CSV manifest with columns:
* `text_description`
* `structure_path`
* `accession`
* `sample_num`

A small dummy example is provided:

```bash
cat test_data/manifest.csv
```

## 4. Run each evaluator

Each eval writes results to:

* JSON: `results/<RUN_NAME>/<EVAL_NAME>/<ACCESSION>/sample_<N>.json`
* Logs: `logs/<RUN_NAME>/<EVAL_NAME>/run.log`
* Raw artifacts: adjacent `sample_<N>.<eval>/` dirs (varies by eval)

Set `RUN_NAME`, `MANIFEST`, `BATCH_SIZE`, and `OVERWRITE` in the script header if desired.

One json per structure sample (usually we have multiple structures generated per protein).

### DSSP

```bash
bash scripts/eval-dssp.sh
```

See [here](https://github.com/jleechung/text2struct-evals/blob/main/results/testrun/dssp/5fo5_A/sample_0.json) for example output for one structure.

### P2Rank

```bash
bash scripts/eval-p2rank.sh
```

See [here](https://github.com/jleechung/text2struct-evals/blob/main/results/testrun/p2rank/5fo5_A/sample_0.json) for example output for one structure.

### GearNet / Proteina

```bash
bash scripts/eval-gearnet.sh
```

See [here](https://github.com/jleechung/text2struct-evals/blob/main/results/testrun/gearnet/5fo5_A/sample_0.json) for example output for one structure.

### ThermoMPNN

```bash
bash scripts/eval-thermompnn.sh
```

See [here](https://github.com/jleechung/text2struct-evals/blob/main/results/testrun/thermompnn/5fo5_A/sample_0.json) for example output for one structure.

### CLEAN

```bash
bash scripts/eval-clean.sh
```

See [here](https://github.com/jleechung/text2struct-evals/blob/main/results/testrun/clean/5fo5_A/sample_0.json) for example output for one structure.

## To-dos

For now, these scripts run evaluators and return raw outputs. Evaluating against text descriptions is not yet implemented.
# Text-to-Structure Protein Diffusion Model: Evaluation Framework

## Overview

### Context

This framework evaluates **text-conditioned protein structure generation models** (e.g., diffusion models that generate 3D protein structures from natural language descriptions). The evaluation pipeline follows:

```
Text Prompt → Generative Model → 3D Protein Structure → Evaluation Tools → Metrics
```

**Example workflow:**
- Input: "Design a mainly-alpha oxidoreductase enzyme"
- Output: PDB file with 3D coordinates
- Evaluation: Does the structure match the prompt specifications?
    1. First, run prediction models on structures
    2. Second, compare predictions to text prompts

### Purpose

The choice of evaluation metrics directly informs:
1. **Dataset curation**: What structural/functional annotations do we need in training data?
2. **Prompt design**: What properties can we reliably condition on?
3. **Model capabilities**: Which aspects of protein design can the model control?

For instance, if we evaluate CATH fold classification, we need training data with CATH annotations. If we evaluate enzyme function, we need EC numbers in our dataset.

### Evaluation Categories

Evaluations are organized into two categories:

**Structural Evaluations**: Properties verifiable directly from 3D coordinates
- Secondary structure (helices, sheets)
- Fold classification (CATH hierarchy)

**Functional Evaluations**: Biological properties requiring additional inference
- Ligand binding sites
- Enzyme function (EC numbers)
- Stability predictions (relative) (E. coli expression)

---

## Classification: Structural vs Functional

### Structural Evaluations
Properties that can be verified directly from 3D coordinates without additional biological knowledge:
- **Secondary structure**: Helices, sheets, coils (geometric patterns)
- **Fold classification**: CATH hierarchy (structural similarity)

### Functional Evaluations
Biological properties that require inference models trained on functional data:
- Ligand binding sites (where molecules bind)
- Enzyme function (catalytic activity, EC numbers)
- Stability (thermodynamic predictions)
- Solubility (E. coli expression compatibility)

---

## STRUCTURAL EVALUATIONS

### 1. Secondary Structure Verification

**Tool: DSSP (Define Secondary Structure of Proteins)**

**Why DSSP?**
- Gold standard in the field since 1983
- Most widely cited secondary structure assignment method (>25,000 citations)
- Used as the reference for training all modern prediction methods
- Fast, deterministic, well-maintained

**Installation:**
```bash
# Via conda
conda install -c salilab dssp

# Or download from: https://github.com/PDB-REDO/dssp
```

**Input:** PDB file

**Example Usage:**
```bash
mkdssp input.pdb output.dssp
```

**Output Format:**
```
# Example DSSP output (simplified)
  #  RESIDUE AA STRUCTURE
    1    1 A M     E        <- beta sheet
    2    2 A V     E
    3    3 A K     C        <- coil
    4    4 A V     H        <- alpha helix
    5    5 A K     H
```

**Metrics:**
- Helix count, sheet count
- Helix/sheet spans (residue ranges)
- Secondary structure content (% helical, % sheet, % coil)

**Example Prompts:**
1. **Specific**: "Design an alpha-helical protein with 8 helices spanning 60% of the structure"
2. **Comparative**: "Design a mainly-beta protein" vs "Design a mainly-alpha protein"

**Expected Output:**
- Text file with per-residue assignments (H=helix, E=sheet, C=coil, etc.)
- Can be parsed to count secondary structure elements
- Example: "Protein has 12 helices, 8 sheets, 45% helical content"

---

### 2. CATH Fold Classification

**Primary Tool: Proteina CATH Classifier (NVIDIA)**

**Why Proteina Classifier?**
- Trained on millions of structures from AlphaFold Database with CATH labels
- State-of-the-art architecture (transformer-based)
- Used for FPSD (Fréchet Protein Structure Distance) metric in benchmarking
- Part of published NeurIPS/ICLR paper with rigorous evaluation
- Hierarchical classification across CATH levels (Class/Architecture/Topology)

**Source:**
- GitHub: https://github.com/NVIDIA-Digital-Bio/proteina
- Paper: "Proteina: Scaling Flow-based Protein Structure Generative Models" (ICLR 2025)

**Training Data:**
- Trained on AlphaFold Database structures (588K-20M structures)
- CATH labels from TED (The Encyclopedia of Domains)
- 99.9% labeled coverage for 588K set, 69.7% for 20M set

**Installation:**
```bash
git clone https://github.com/NVIDIA-Digital-Bio/proteina
cd proteina
conda env create -f environment.yaml
conda activate proteina_env
pip install -e .
```

**Usage (for evaluation):**
```bash
# Using Proteina's CATH classifier for FPSD calculation
python script_utils/inference_fid.py \
    --data_dir <path_to_your_proteins> \
    --ca_only \
    --batch_size 12 \
    --num_workers 32
```

**Input:** PDB file (Cα coordinates)

**Output:**
- CATH class predictions at multiple hierarchy levels
- Class: Mainly Alpha / Mainly Beta / Alpha-Beta / Few Secondary Structures
- Architecture: e.g., "Alpha-Beta Barrel", "Immunoglobulin-like"
- Topology: Specific fold classification
- Feature embeddings for FPSD calculation

**Metrics:**
- CATH class accuracy (Class/Architecture/Topology levels)
- Fold diversity (Fold Score - fS)
- Distribution similarity (FPSD, fJSD)
- Hierarchical classification accuracy

**Example Prompts:**
1. **High-level**: "Design a mainly-alpha protein" → Check predicted CATH class
2. **Specific fold**: "Design a TIM barrel fold protein" → Check topology classification
3. **Architecture-level**: "Design an immunoglobulin-like beta sandwich" → Check architecture

**Expected Evaluation:**
```python
# Example evaluation workflow
from proteina import CATHClassifier

classifier = CATHClassifier(weights='path/to/weights.pth')
predictions = classifier.predict(structure_path='generated.pdb')

# Output format:
# {
#   'class': 'Alpha Beta',
#   'architecture': 'Alpha-Beta Barrel', 
#   'topology': 'TIM Barrel',
#   'confidence': 0.92
# }
```

**Alternative: DeepFRI (for functional context)**
- Can also classify CATH folds as part of its annotations
- Better for function prediction than pure fold classification
- See "Functional Evaluations" section for details

**Note on CATH Classification:**
- CATH hierarchy: **C**lass → **A**rchitecture → **T**opology → **H**omologous superfamily
- Topology level is most relevant for "fold" evaluation
- Official CATH database provides manual classifications for PDB structures

---

## FUNCTIONAL EVALUATIONS

### 1. Ligand Binding Site Prediction

**Tool: P2Rank**

**Why P2Rank?**
- State-of-the-art performance (outperforms Fpocket, SiteHound, MetaPocket)
- Fast: <1 second per structure
- Well-maintained: 396 stars on GitHub, actively developed
- Published in *Journal of Cheminformatics* (2018, 900+ citations)
- Template-free (no database required)
- Comprehensive benchmarking in 2024 study confirms top performance

**GitHub:** https://github.com/rdk/p2rank (396 stars)

**Installation:**
```bash
git clone https://github.com/rdk/p2rank.git
cd p2rank
# No dependencies needed - pre-built Java executable
```

**Input:** PDB file (accepts both experimental and predicted structures including AlphaFold)

**Example Usage:**
```bash
prank predict input.pdb
```

**Output Format:**
```
# Example P2Rank output (simplified)
Pocket 1:  rank=1  score=32.5  probability=0.89
  Residues: A:VAL45, A:LEU67, A:ASP89, A:PHE112, A:TYR135
  Center: [12.3, 45.6, 78.9]
  
Pocket 2:  rank=2  score=18.2  probability=0.65
  Residues: B:TRP23, B:HIS56, B:ARG78
  Center: [34.2, 12.1, 56.7]
```

**Metrics:**
- **Pocket detection accuracy**: Does it find known binding sites?
- **Residue-level precision/recall**: Are predicted residues correct?
- **Top-N success rate**: Is the true pocket in top N predictions?

**Example Prompts:**
1. **Specific**: "Design a protein that binds ATP at residues 45, 67, 89"
2. **General**: "Design a protein with a deep hydrophobic pocket suitable for small molecule binding"

**Expected Evaluation:**
- Run P2Rank on generated structure
- Check if predicted pockets match prompted specifications
- Compare pocket properties (depth, hydrophobicity, size) to prompt claims

---

### 2. Enzyme Function (EC Number) Prediction

**Tool: CLEAN (Contrastive Learning-Enabled Enzyme ANnotation)**

**Why CLEAN?**
- Published in *Science* (2023, Yu et al.)
- Best performance on challenging benchmarks (New-392, Price-149)
- Precision: 0.597, Recall: 0.481 (vs ProteInfer: 0.36/0.43, DeepEC: 0.29/0.40)
- Outperforms 6 state-of-the-art tools (ProteInfer, DeepEC, BLASTp, DEEPre, CatFam, ECPred)
- Handles multifunctional enzymes well
- Robust to low-homology cases

**GitHub:** https://github.com/tttianhao/CLEAN (cited by CLEAN-Contact, ProtDETR, EC-Bench)

**Installation:**
```bash
git clone https://github.com/tttianhao/CLEAN.git
cd CLEAN
pip install -r requirements.txt
```

**Input:** Protein sequence (FASTA format) OR PDB file

**Example Usage:**
```bash
python predict.py --input protein.fasta --output predictions.json
```

**Output Format:**
```json
{
  "protein_id": "example_protein",
  "predictions": [
    {
      "EC_number": "1.1.1.1",
      "confidence": 0.89,
      "description": "Alcohol dehydrogenase"
    },
    {
      "EC_number": "1.1.1.37", 
      "confidence": 0.76,
      "description": "Malate dehydrogenase"
    }
  ]
}
```

**Metrics:**
- **EC classification accuracy**: Hierarchical (Class → Subclass → Sub-subclass → Substrate)
- **Precision/Recall/F1**: At each EC level
- **Multifunctional detection**: Can it predict multiple functions correctly?

**Example Prompts:**
1. **Specific**: "Design an oxidoreductase enzyme (EC 1.x.x.x) that catalyzes NAD-dependent reactions"
2. **General**: "Design a kinase enzyme"

**Expected Evaluation:**
- Extract sequence from generated PDB
- Run CLEAN prediction
- Check if predicted EC matches prompted enzyme class
- Example: "oxidoreductase" → should predict EC 1.x.x.x

**EC Number Hierarchy:**
```
EC 1.x.x.x  = Oxidoreductases
EC 2.x.x.x  = Transferases (e.g., kinases)
EC 3.x.x.x  = Hydrolases
EC 4.x.x.x  = Lyases
EC 5.x.x.x  = Isomerases
EC 6.x.x.x  = Ligases
EC 7.x.x.x  = Translocases
```

---

### 3. Stability Prediction (Relative)

**Tool: ThermoMPNN**

**Why ThermoMPNN?**
- Published in *PNAS* (2024, Diaz et al.)
- Trained on 272K mutations from Megascale dataset
- Transfer learning from ProteinMPNN (19K PDB structures)
- State-of-the-art ΔΔG prediction performance
- Fast structure-based predictions

**Note:** ThermoMPNN predicts **relative** stability (ΔΔG for mutations), not absolute ΔG. Useful for comparative evaluations.

**GitHub:** https://github.com/Kuhlman-Lab/ThermoMPNN

**Installation:**
```bash
git clone https://github.com/Kuhlman-Lab/ThermoMPNN.git
cd ThermoMPNN
pip install -r requirements.txt
```

**Input:** PDB file

**Example Usage:**
```bash
python predict_ddg.py --pdb input.pdb --output predictions.csv
```

**Output Format:**
```csv
position,wildtype,mutation,predicted_ddg
45,V,A,-0.5
45,V,I,0.2
45,V,D,2.1
67,L,A,-0.3
...
```

**Metrics:**
- **Mean predicted ΔΔG**: Across all positions
- **Mutation tolerance**: Fraction of neutral/stabilizing mutations
- **Sequence optimality**: Is chosen amino acid in top-K by stability?

**Example Prompts:**
1. **Comparative**: "Design a thermostable protein" vs "Design an unstable protein"
2. **Relative**: Compare stability distributions to natural proteins

**Expected Evaluation:**
- Scan all positions for all 20 amino acids
- Calculate mean ΔΔG (more negative = harder to destabilize = more stable)
- Compare "stable" vs "unstable" designed proteins
- Example output: "Thermostable protein: mean ΔΔG = -1.2 kcal/mol vs Unstable protein: mean ΔΔG = +0.8 kcal/mol"

**Interpretation:**
- ΔΔG < 0: Current amino acid is more stabilizing than mutation
- ΔΔG > 0: Mutation would be more stabilizing
- Mean ΔΔG across all mutations indicates how "optimized" the sequence is

---

## EVALUATION PROTOCOL EXAMPLES

### Example 1: Binding Site Prompt

**Prompt:** "Design a protein that binds ATP with a binding pocket at residues 50-60"

**Evaluation Steps:**
1. Generate structure with diffusion model
2. Save as PDB file
3. Run P2Rank:
   ```bash
   prank predict generated_protein.pdb
   ```
4. Check output:
   - Is there a high-scoring pocket (probability >0.7)?
   - Are residues 50-60 included in the predicted pocket?
   - **Success metric**: Overlap between predicted pocket residues and specified range

**Expected P2Rank Output:**
```
Pocket 1: score=28.3, probability=0.82
Residues: A:ILE50, A:VAL52, A:LEU55, A:ASP58, A:PHE60
```
✅ **Success**: Pocket detected at correct location with high confidence

---

### Example 2: Enzyme Function Prompt

**Prompt:** "Design an oxidoreductase enzyme"

**Evaluation Steps:**
1. Generate structure
2. Extract sequence from PDB
3. Run CLEAN:
   ```bash
   python CLEAN/predict.py --input sequence.fasta
   ```
4. Check output:
   - Is predicted EC number in class 1.x.x.x (oxidoreductases)?
   - What is the confidence score?
   - **Success metric**: Correct EC class with confidence >0.5

**Expected CLEAN Output:**
```json
{
  "predictions": [
    {"EC_number": "1.1.1.37", "confidence": 0.84, "description": "Malate dehydrogenase"}
  ]
}
```
✅ **Success**: Predicted EC 1.x.x.x (oxidoreductase) with high confidence

---

### Example 3: Comparative Stability Prompt

**Prompt A:** "Design a thermostable helical protein"
**Prompt B:** "Design an unstable helical protein"

**Evaluation Steps:**
1. Generate both structures
2. Run ThermoMPNN on both:
   ```bash
   python ThermoMPNN/predict_ddg.py --pdb protein_A.pdb
   python ThermoMPNN/predict_ddg.py --pdb protein_B.pdb
   ```
3. Calculate mean ΔΔG for all mutations
4. Compare distributions

**Expected ThermoMPNN Output:**
```
Protein A (thermostable): mean ΔΔG = -1.5 kcal/mol
  - 78% of mutations are destabilizing (ΔΔG > 0)
  - High mutation intolerance = stable sequence

Protein B (unstable): mean ΔΔG = +0.6 kcal/mol
  - 42% of mutations are destabilizing
  - Low mutation intolerance = suboptimal sequence
```
✅ **Success**: Thermostable protein shows more negative ΔΔG, higher mutation intolerance

---

## SCALABILITY ANALYSIS

### Most Scalable (10,000+ structures/day)
1. **DSSP** - Secondary structure (~0.1s per structure)
2. **P2Rank** - Binding sites (<1s per structure)
3. **CLEAN** - EC numbers (sequence-based, <1s)

### Moderately Scalable (1,000-5,000 structures/day)
4. **Proteina CATH Classifier** - Fold classification (~5s per structure, batch-friendly)
5. **ThermoMPNN** - Stability (~10s for all mutations)

### Notes on Scalability:
- All tools accept standard PDB format
- Most can be batched/parallelized for large-scale evaluation
- Sequence-based tools (CLEAN) only need PDB→FASTA conversion
- Structure-based tools (P2Rank, ThermoMPNN, DSSP, CATH classifier) use coordinates
- GPU acceleration available for: Proteina classifier, CLEAN (optional)

---

## SUMMARY TABLE

| Evaluation Type | Tool | Input | Speed | Primary Metric | Category |
|----------------|------|-------|-------|----------------|----------|
| **Secondary Structure** | **DSSP** | **PDB** | **<1s** | **Helix/sheet count** | **Structural** |
| **CATH Fold Classification** | **Proteina Classifier** | **PDB** | **~5s** | **Fold class accuracy** | **Structural** |
| Binding Sites | P2Rank | PDB | <1s | Pocket detection accuracy | Functional |
| Enzyme Function | CLEAN | FASTA/PDB | <1s | EC classification accuracy | Functional |
| Stability (relative) | ThermoMPNN | PDB | ~10s | Mean ΔΔG, mutation tolerance | Functional |

---

## VALIDATION STRATEGY

### Cross-Validation Approach
For each evaluation:
1. **Positive control**: Prompt for specific property
2. **Negative control**: Prompt for opposite property
3. **Ablation**: Test specificity (generic → specific prompts)
4. **Consistency**: Multiple generations from same prompt should cluster

### Example Validation Matrix

| Prompt | DSSP | P2Rank | CLEAN | ThermoMPNN |
|--------|------|--------|-------|------------|
| "Design alpha-helical oxidoreductase with ATP binding pocket" | ✓ Check helix count | ✓ Check pocket | ✓ Check EC 1.x | - |
| "Design thermostable protein" | - | - | - | ✓ Check ΔΔG |

---

## RECOMMENDED WORKFLOW

```bash
# 1. Generate structure from text prompt
python generate_structure.py \
    --prompt "design a mainly-alpha oxidoreductase enzyme" \
    --output structure.pdb

# 2. Convert to sequence (for sequence-based tools)
python pdb_to_fasta.py structure.pdb > sequence.fasta

# 3. Run STRUCTURAL evaluations
# Secondary structure
mkdssp structure.pdb structure.dssp

# CATH fold classification
python proteina/script_utils/inference_fid.py \
    --data_dir ./structures \
    --ca_only \
    --batch_size 12

# 4. Run FUNCTIONAL evaluations
# Binding site prediction
prank predict structure.pdb

# Enzyme function prediction
python CLEAN/predict.py --input sequence.fasta

# Stability prediction (relative)
python ThermoMPNN/predict_ddg.py --pdb structure.pdb

# 5. Parse and aggregate all results
python parse_evaluation_results.py \
    --dssp structure.dssp \
    --cath cath_predictions.json \
    --prank prank_output/ \
    --clean clean_predictions.json \
    --thermompnn thermompnn_ddg.csv \
    --output aggregated_metrics.json

# 6. Compare against prompt
python compare_to_prompt.py \
    --prompt "design a mainly-alpha oxidoreductase enzyme" \
    --metrics aggregated_metrics.json
```

### Expected Outputs:
- **DSSP**: Secondary structure assignments, helix/sheet counts
- **Proteina**: CATH class/architecture/topology predictions with confidence
- **P2Rank**: Predicted binding pockets with scores
- **CLEAN**: EC number predictions (e.g., EC 1.x.x.x for oxidoreductase)
- **ThermoMPNN**: ΔΔG values for all possible mutations

---

## FUTURE DIRECTIONS

### Additional Evaluations to Consider:
- **GO term prediction**: DeepGO, DeepFRI (for functional annotations beyond EC numbers)
- **Disulfide bond prediction**: Validation of predicted disulfide bridges
- **Protein-protein interfaces**: For designed complexes and binding evaluations

### Advanced Metrics:
- **Designability**: sc-RMSD from structure prediction (AF2/ESMFold) - fold protein to verify structure
- **ProTrek score**: Text-structure alignment in joint embedding space
- **Rosetta energy**: Overall structural quality and folding stability
- **pLDDT from AlphaFold**: Confidence scores for generated structures

### Benchmark Datasets:
- Curate "ground truth" proteins with experimentally verified properties
- Create standardized prompt → structure → evaluation pipeline
- Establish baseline scores from natural proteins for comparison

---

## REFERENCES

1. **DSSP**: Kabsch W, Sander C. Dictionary of protein secondary structure. *Biopolymers* 1983. [GitHub](https://github.com/PDB-REDO/dssp)

2. **Proteina CATH Classifier**: Geffner T, et al. Proteina: Scaling Flow-based Protein Structure Generative Models. *ICLR* 2025. [GitHub](https://github.com/NVIDIA-Digital-Bio/proteina) | [Paper](https://openreview.net/forum?id=TVQLu34bdw)

3. **P2Rank**: Krivák R, Hoksza D. P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure. *J Cheminform* 2018. [GitHub](https://github.com/rdk/p2rank)

4. **CLEAN**: Yu T, et al. Enzyme function prediction using contrastive learning. *Science* 2023. [GitHub](https://github.com/tttianhao/CLEAN)

5. **ThermoMPNN**: Diaz DJ, et al. Transfer learning to leverage larger datasets for improved prediction of protein stability changes. *PNAS* 2024. [GitHub](https://github.com/Kuhlman-Lab/ThermoMPNN)

---

## NOTES

- All tools accept standard PDB format output from diffusion models
- Most evaluations can be automated in a pipeline
- Sequence-based evaluations (CLEAN) require PDB→FASTA conversion
- Consider creating benchmark dataset of "ground truth" proteins with known properties
- For statistical significance, need N>100 samples per prompt type
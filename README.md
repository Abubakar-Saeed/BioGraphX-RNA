# BioGraphX-RNA
[![DOI](https://img.shields.io/badge/DOI-10.64898%252F2026.02.23.889451-blue)](https://doi.org/10.64898/2026.02.23.707573)
![Python Version](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-Academic-blue.svg)

**BioGraphX-RNA** is a physicochemical graph encoding framework that transforms RNA primary sequences into high-fidelity 149-dimensional biophysical feature vectors. It models RNAs as nucleotide interaction networks based on fundamental RNA biochemical principles, enabling structure-independent sequence analysis for subcellular localization prediction and functional characterization.
---

## 🎯 Key Features

### 🧬 Architecture
* **RNA Physics Engine**: Models 5 RNA-specific interaction types (canonical Watson-Crick, wobble pairs, non-canonical pairs, base stacking, phosphate backbone) with linear sequence distance constraints and minimum loop size enforcement (distance ≥ 4 for base pairing).
* **Adaptive Processor**: Smart truncation (30% 5' / 40% Core / 30% 3') preserving localization signals, regulatory elements, and structural motifs. Sliding window strategy for ultra-long transcripts (>5000 nt).
* **Graph Construction Engine**: Nucleotide-level interaction networks with hybrid interaction tracking (WC+stacking, wobble+stacking, stacking+backbone) and cooperative binding scores.
* **Frustration Analysis**: Per-nucleotide conformational conflict detection from constraint graph topology, identifying competing interaction patterns and structural stress points.
* **Localization Profiler**: Scans for compartment-specific RNA motifs (e.g., AREs, Shine-Dalgarno, nuclear retention signals, mitochondrial targeting sequences).

### 📊 Comprehensive Feature Extraction
| Feature Category | Count | Description |
| :--- | :---: | :--- |
| **Graph Topology** | 65 | Degree distributions, centrality measures (betweenness/closeness/eigenvector with cutoff=10), community structure, path efficiency, regional densities (5'/3'/GC/AU clusters) |
| **Hybrid Features** | 15 | Cooperative interaction scores (WC+stacking, wobble+stacking, stacking+backbone), regional hybrid enrichment, hybrid network properties, energy-based metrics |
| **Knowledge Profiles** | 27 | Compartment-specific motif scores for 9 localizations (Nucleus, Exosome, Cytosol, Cytoplasm, Ribosome, Membrane, ER, Microvesicles, Mitochondrion) with hybrid and GC compatibility |
| **Frustration Analysis** | 17 | Per-nucleotide frustration, structured vs flexible region contrast, hotspot detection, motif-associated stress, long-range interaction frustration, frustration entropy |
| **Global Physics** | 25 | GC/AU content, nucleotide frequencies, skew measures, Shannon entropy, MFE per nucleotide, dinucleotide frequencies, pairing potential statistics, GC autocorrelation |

---

## 🏗️ Architecture
```text
BioGraphX-RNA Pipeline
├── RNAPhysicsStrategy (5 Interaction Rules + Hybrid Detection)
├── RNASequencePreprocessor (Motif-Preserving Truncation & Sliding Windows)
├── RNAMotifProfiler (9 Compartment-specific Localization Scoring)
├── RNAGraphEngine (Nucleotide Interaction Network Construction)
└── RNAFrustrationAnalyzer (Conformational Conflict Detection)

```
## Quick Start
### Installation
# Clone repository
```text
from biographx_rna.pipeline import RNAsubLocalizationPipeline

# Initialize pipeline
pipeline = RNAsubLocalizationPipeline()

# Process a single RNA sequence
sequence = "AUGGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAG"
features = pipeline.process_rna_sequence(sequence)

print(f"Extracted {len(features)} biophysical features")
print(f"GC Content: {features[0]:.3f}")  # First feature is GC_Content
print(f"Nuclear motif score: {features[65+15+0]:.3f}")  # Profile_Nucleus_Motif
print(f"5' frustration: {features[65+15+27+25+0]:.3f}")  # Frustration_5Prime_Mean

```
### Process a protein sequences
```text
# Process a batch of RNA sequences
sequences = [
    "AUGGCUAGCUAGCUAGCUA",  # Short RNA
    "AUG" + "CUA"*100 + "UAA",  # Medium RNA
    "AUG" + "CUA"*1000 + "UAA"  # Long RNA (will use sliding windows)
]

results = pipeline.process_rna_batch(sequences)
for i, features in enumerate(results):
    print(f"Sequence {i+1}: {len(features)} features extracted")
```
### Batch Processing from CSV
```text
from biographx_rna.pipeline import run_rna_pipeline

# Process large-scale transcriptomics datasets
run_rna_pipeline(
    input_file="rna_sequences.csv",      # Requires 'Sequence' column
    output_file="rna_features.csv",
    chunk_size=500,                       # Sequences per chunk
    n_jobs=10                             # Parallel workers
)
```
## Scientific Basis
BioGraphX-RNA represents RNAs as mathematical graphs where nodes are nucleotides and edges represent potential RNA-specific interactions. All distance constraints are measured in linear sequence positions (nucleotide indices), NOT 3D Ångström spatial coordinates, enabling structure-independent analysis of primary sequences.
### Localization Patterns
The pipeline identifies specific interaction patterns associated with:

* **Nucleus:** G/A-rich clusters, RRHY/YRRR motifs
* **Exsosome:** AU-rich elements (AREs), UUAUUUAUU degradation signals
* **Ribosome:** Shine-Dalgarno sequences, Kozak context, AUG start codons
* **Mitochondrion:** 5' A-rich regions, GC-rich stretches
* **Membrane:** U-rich clusters, hydrophobicity signals
* **ER:** SRP binding signals, U-flanked purine-rich motifs
* **Microvesicles:** Extended AU regions, export motifs

## Citation
If you use BioGraphX-RNA in your research, please cite:
Saeed, A., & Abbas, W. (2026). BioGraphX-RNA: A universal physicochemical graph encoding for interpretable RNA subcellular localization prediction. https://doi.org/10.64898/2026.02.23.707573


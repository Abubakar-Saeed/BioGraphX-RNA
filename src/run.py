"""
Author: Abubakar Saeed
Date: February 23, 2026
Description: Execution script for RNA subcellular localization feature extraction pipeline.
This script processes RNA sequences from an input CSV file and generates a comprehensive
149-dimensional feature matrix for machine learning applications.

The pipeline integrates:
- RNA-specific graph construction and analysis
- Motif-based localization signal detection
- Physicochemical property extraction
- Structural frustration analysis
- Adaptive processing for sequences of varying lengths

Note: All distance-based calculations use linear sequence positions (nucleotides),
not 3D spatial distances in Angstroms.
"""

import sys
import os
from biographx_rna.pipeline import run_rna_pipeline

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Add source directory to Python path to ensure module imports work correctly
# This allows importing from the src directory regardless of execution location
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# =============================================================================
# INPUT/OUTPUT CONFIGURATION
# =============================================================================

# Input file: CSV containing RNA sequences with required 'Sequence' column
# Expected format: CSV file with at least a 'Sequence' column containing RNA strings
# Additional metadata columns will be preserved in output
input_file = r"D:\BioGraphX-RNA\BioGraphX-RNA\src\biographx_rna\data\miRNA.csv"

# Output file: CSV file that will contain:
#   - All original columns from input (except 'Sequence')
#   - 149 RNA feature columns (RNA_COMPLETE_FEATURE_NAMES)
# The 'Sequence' column is replaced by the feature vectors
output_file = r"D:\BioGraphX-RNA\BioGraphX-RNA\src\processed_data\miRNA_encoded.csv"

# Create output directory structure if it doesn't exist
# This prevents errors when writing to a new directory
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

# Execute the main RNA processing pipeline
# The pipeline will:
#   1. Read input CSV in chunks to manage memory
#   2. Process each RNA sequence through the adaptive feature extraction pipeline
#   3. Handle sequences of different lengths appropriately:
#      - ≤1000 nt: Full processing
#      - 1000-5000 nt: Smart truncation preserving critical regions
#      - >5000 nt: Sliding window with weighted aggregation
#   4. Write results incrementally to output file
#   5. Display progress updates during processing

run_rna_pipeline(
    input_file=input_file,      # Path to input CSV with RNA sequences
    output_file=output_file,    # Path for output feature matrix
    chunk_size=500,             # Number of sequences to process per chunk
    n_jobs=10                   # Recommended: set to number of CPU cores

)

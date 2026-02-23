"""
Author: Abubakar Saeed
Date: February 23, 2026
Description: Main pipeline for RNA subcellular localization feature extraction.
Integrates all RNA-specific components including physics-based rules, graph construction,
motif profiling, and frustration analysis to generate comprehensive feature vectors
for machine learning applications.

The pipeline handles sequences of varying lengths through adaptive strategies:
- ≤1000 nt: Full processing
- 1000-5000 nt: Smart truncation preserving critical regions
- >5000 nt: Sliding window with weighted aggregation
"""

from typing import Dict,List

from src.biographx_rna.biophysics import RNAPhysicsStrategy
from src.biographx_rna.preprocessor import RNASequencePreprocessor
from src.biographx_rna.profiler import RNAMotifProfiler
from src.biographx_rna.graph_engine import RNAGraphEngine
from src.biographx_rna.frustration_analyzer import RNAFrustrationAnalyzer
from src.biographx_rna.utils.feature_names import RNA_COMPLETE_FEATURE_NAMES

import numpy as np
import gc
import csv
import pandas as pd
from joblib import Parallel, delayed

class RNAsubLocalizationPipeline:
    """
    Main pipeline for RNA subcellular localization feature extraction.

    This class orchestrates all RNA-specific analysis components to generate
    a comprehensive feature set for predicting RNA subcellular localization.
    The pipeline integrates:

    1. RNA Physics Strategy: Base pairing rules, interaction energies, GC content
    2. Sequence Preprocessor: Adaptive truncation and motif-preserving windowing
    3. Motif Profiler: Detection of localization signals and functional motifs
    4. Graph Engine: Construction and analysis of RNA interaction networks
    5. Frustration Analyzer: Quantification of conflicting structural constraints

    Attributes:
        rna_physics (RNAPhysicsStrategy): RNA biophysical rules
        rna_preprocessor (RNASequencePreprocessor): Sequence preprocessing
        rna_motif_profiler (RNAMotifProfiler): Motif detection
        rna_graph_engine (RNAGraphEngine): Graph construction and analysis
        rna_frustration_analyzer (RNAFrustrationAnalyzer): Frustration computation
        total_features (int): Total feature dimension (149)
    """

    def __init__(self):
        """
        Initialize all RNA-specific components for the pipeline.

        Sets up the complete analysis pipeline with all required modules.
        """
        # Initialize all RNA-specific components
        self.rna_physics = RNAPhysicsStrategy()
        self.rna_preprocessor = RNASequencePreprocessor()
        self.rna_motif_profiler = RNAMotifProfiler(self.rna_physics)
        self.rna_graph_engine = RNAGraphEngine(self.rna_physics, self.rna_motif_profiler)
        self.rna_frustration_analyzer = RNAFrustrationAnalyzer(self.rna_physics)

        # Total feature dimension (sum of all component features)
        self.total_features = 149

    def extract_full_rna_features(self, sequence: str) -> np.ndarray:
        """
        Extract complete feature set for RNA sequence.

        Performs full analysis on a single RNA sequence, including:
        - Graph construction and basic graph features
        - Hybrid interaction features
        - Motif-based knowledge profiles
        - Global physicochemical features
        - Frustration-based features

        Args:
            sequence (str): RNA sequence (A, U, G, C)

        Returns:
            np.ndarray: 149-dimensional feature vector with NaN values cleaned
        """
        # Validate sequence contains only RNA nucleotides
        valid_nts = set('AUCGaucg')
        seq_clean = ''.join([nt for nt in sequence.upper() if nt in valid_nts])

        if not seq_clean or len(seq_clean) == 0:
            return np.zeros(self.total_features, dtype=np.float32)

        # Build RNA-specific graph from cleaned sequence
        graph, hybrid_scores = self.rna_graph_engine.build_rna_graph(seq_clean)

        # Extract features from all components
        basic_features = self.rna_graph_engine.extract_basic_rna_graph_features(graph)
        hybrid_features = self.rna_graph_engine.extract_rna_hybrid_features(graph, hybrid_scores, seq_clean)
        profile_features = self.rna_motif_profiler.extract_rna_knowledge_profiles(seq_clean, hybrid_scores)
        physics_features = self.rna_physics.extract_global_rna_physics(seq_clean)

        # NEW: Extract frustration features (quantifies structural conflicts)
        frustration_features = self.rna_frustration_analyzer.compute_from_rna_constraint_graph(graph, seq_clean)
        frustration_vector = self._extract_rna_frustration_vector(frustration_features)

        # Combine all features
        all_features = np.concatenate([basic_features, hybrid_features,
                                       profile_features, physics_features, frustration_vector])

        # Ensure correct dimension
        if len(all_features) < self.total_features:
            all_features = np.pad(all_features, (0, self.total_features - len(all_features)), 'constant')
        elif len(all_features) > self.total_features:
            all_features = all_features[:self.total_features]

        # Clean NaN values (replace with 0.0)
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

        return all_features

    def _extract_rna_frustration_vector(self, frustration_features: Dict) -> np.ndarray:
        """
        Convert RNA frustration features to vector.

        Extracts core frustration metrics from the dictionary returned by
        RNAFrustrationAnalyzer.

        Args:
            frustration_features (Dict): Dictionary containing frustration metrics

        Returns:
            np.ndarray: 17-dimensional frustration feature vector
        """
        # RNA frustration features (17 features)
        core_features = [
            frustration_features['Frustration_5Prime_Mean'],
            frustration_features['Frustration_3Prime_Mean'],
            frustration_features['Frustration_Middle_Mean'],
            frustration_features['Frustration_GC_Mean'],
            frustration_features['Frustration_GC_Variance'],
            frustration_features['Frustration_AU_Mean'],
            frustration_features['Frustration_AU_Variance'],
            frustration_features['Frustration_5vs3_Prime'],
            frustration_features['Frustration_StructuredVsFlexible'],
            frustration_features['Frustration_Paired_Mean'],
            frustration_features['Frustration_HotspotCount'],
            frustration_features['Frustration_HotspotDensity'],
            frustration_features['Frustration_Motif_Mean'],
            frustration_features['Frustration_Motif_Max'],
            frustration_features['Frustration_NonMotif_Mean'],
            frustration_features['Frustration_LongRange_Mean'],
            frustration_features['Frustration_Entropy']
        ]

        return np.array(core_features, dtype=np.float32)

    def adaptive_extract_rna_features(self, sequence: str) -> np.ndarray:
        """
        Adaptive feature extraction for RNA based on sequence length.

        Selects optimal processing strategy based on sequence length:
        - ≤1000 nt: Full processing (complete analysis)
        - 1000-5000 nt: Smart truncation to 1000nt preserving critical regions
        - >5000 nt: Sliding window with weighted aggregation

        This function is optimized for parallel batch processing with n_jobs.

        Args:
            sequence (str): RNA sequence

        Returns:
            np.ndarray: 149-dimensional feature vector
        """
        # 1. Clean sequence (remove invalid characters)
        valid_nts = set('AUCGaucg')
        seq_clean = ''.join([nt for nt in sequence.upper() if nt in valid_nts])

        length = len(seq_clean)
        if length == 0:
            return np.zeros(self.total_features, dtype=np.float32)

        # 2. Threshold Logic (Aligned with Preprocessor: 1k/5k)
        if length <= 1000:
            # Full processing for short/medium RNAs
            # These can be analyzed completely without truncation
            return self.extract_full_rna_features(seq_clean)

        elif length <= 5000:
            # Smart truncation to 1000nt (BioGraphX Standard)
            # Preserves 30% 5' / 40% Mid / 30% 3' to maintain functional regions
            truncated = self.rna_preprocessor.smart_truncate_rna(seq_clean, 1000)
            return self.extract_full_rna_features(truncated)

        else:
            # 3. Sequential Sliding Window for Very Long RNAs
            # Internal parallelization removed to allow n_jobs=10 to work efficiently
            # Each window is processed sequentially within the worker process
            windows_info = self.rna_preprocessor.create_rna_sliding_windows(seq_clean, 500, 250)
            window_features = []
            window_weights = []

            for window, info in windows_info:
                # Process each window sequentially within the worker process
                features = self.extract_full_rna_features(window)
                window_features.append(features)
                # Use Motif Score as weight to highlight localized "Zipcodes"
                # Regions with high motif density contribute more to final features
                window_weights.append(info['motif_score'])

            # 4. Weighted Aggregation
            # Combine window features using motif-based weights
            window_features = np.array(window_features)
            window_weights = np.array(window_weights)

            if window_weights.sum() > 0:
                window_weights = window_weights / window_weights.sum()
                aggregated = np.average(window_features, axis=0, weights=window_weights)
            else:
                aggregated = np.mean(window_features, axis=0)

            return aggregated

    def process_rna_sequence(self, seq: str) -> np.ndarray:
        """
        Process single RNA sequence.

        Convenience wrapper for processing individual sequences.

        Args:
            seq (str): RNA sequence

        Returns:
            np.ndarray: Feature vector
        """
        return self.adaptive_extract_rna_features(seq)

    def process_rna_batch(self, sequences: List[str]) -> List[np.ndarray]:
        """
        Process batch of RNA sequences.

        Processes multiple sequences sequentially, designed to be called
        by parallel workers for batch processing.

        Args:
            sequences (List[str]): List of RNA sequences

        Returns:
            List[np.ndarray]: List of feature vectors
        """
        results = []

        for idx, seq in enumerate(sequences):
            features = self.process_rna_sequence(seq)
            results.append(features)

        return results


def run_rna_pipeline(input_file: str, output_file: str,
                                     chunk_size: int = 500, n_jobs: int = 4) -> None:
    """
    Run RNA sublocalization pipeline while preserving original columns.

    Processes RNA sequences from an input CSV file, extracts features, and
    saves results to an output CSV file. Preserves all original columns
    except the 'Sequence' column (which is replaced by feature vectors).

    Args:
        input_file (str): Path to input CSV file (must contain 'Sequence' column)
        output_file (str): Path to output CSV file for features
        chunk_size (int): Number of sequences to process per chunk
        n_jobs (int): Number of parallel workers for batch processing

    Raises:
        ValueError: If input file lacks required 'Sequence' column

    Note:
        The pipeline uses adaptive processing based on sequence length:
        - Short sequences (<1k) are fully processed
        - Medium sequences (1k-5k) are smart-truncated
        - Long sequences (>5k) use sliding windows with weighted aggregation
    """
    # Count total sequences for progress tracking
    with open(input_file, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header

    print("=" * 70)
    print("🧬 RNA SUBLOCALIZATION GRAPH PIPELINE")
    print("=" * 70)
    print(f"📊 Features: {len(RNA_COMPLETE_FEATURE_NAMES)} dimensions")
    print(f"📈 Processing {total_rows:,} RNA sequences")
    print(f"⚡ Chunk size: {chunk_size}")
    print(f"🔧 Workers: {n_jobs}")
    print("=" * 70)

    pipeline = RNAsubLocalizationPipeline()

    # Read header only to get original column names
    original_df = pd.read_csv(input_file, nrows=0)
    original_columns = list(original_df.columns)

    if 'Sequence' not in original_columns:
        raise ValueError("Input file must contain 'Sequence' column")

    # 🔧  Remove 'Sequence' from output columns
    output_original_columns = [c for c in original_columns if c != 'Sequence']

    # Write header to output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(output_original_columns + RNA_COMPLETE_FEATURE_NAMES)

    total_processed = 0
    chunk_counter = 0

    # Process data in chunks for memory efficiency
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        sequences = chunk['Sequence'].astype(str).tolist()

        # Create batches for parallel processing
        batch_size = max(50, len(sequences) // (n_jobs * 2))
        batches = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]

        # Process batches in parallel
        encoded_batches = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            verbose=0
        )(delayed(pipeline.process_rna_batch)(batch) for batch in batches)

        # Flatten results
        encoded_vectors = [vec for batch in encoded_batches for vec in batch]

        # Drop Sequence column before writing output
        chunk_no_seq = chunk.drop(columns=['Sequence'])

        # Write chunk results to output file
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, vector in enumerate(encoded_vectors):

                # Validate feature vector length
                if len(vector) != len(RNA_COMPLETE_FEATURE_NAMES):
                    print(f"⚠️ Warning: Vector {i} has {len(vector)} features")
                    if len(vector) < len(RNA_COMPLETE_FEATURE_NAMES):
                        vector = np.pad(
                            vector,
                            (0, len(RNA_COMPLETE_FEATURE_NAMES) - len(vector)),
                            'constant'
                        )
                    else:
                        vector = vector[:len(RNA_COMPLETE_FEATURE_NAMES)]

                # Write original columns (without Sequence) + features
                writer.writerow(
                    chunk_no_seq.iloc[i].tolist() + vector.tolist()
                )

        total_processed += len(sequences)
        chunk_counter += 1
        completion = (total_processed / total_rows) * 100

        # Progress update
        print(
            f"\r📊 Progress: {completion:.1f}% "
            f"| {total_processed:,}/{total_rows:,} sequences "
            f"| Chunk {chunk_counter}",
            end=""
        )

        # Clean up to free memory
        del encoded_batches, encoded_vectors, batches, chunk, chunk_no_seq
        gc.collect()

    print("\n\n✅ RNA pipeline complete!")
    print(f"📁 Output saved: {output_file}")

    # Quick verification of output
    final_df = pd.read_csv(output_file, nrows=5)
    print(f"\n🔍 Total output columns: {len(final_df.columns)}")
    print(f"🧬 Feature columns: {pipeline.total_features}")
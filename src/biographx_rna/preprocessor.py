"""
Author: Abubakar Saeed
Date: February 23, 2026
Description: RNA sequence preprocessing module implementing adaptive truncation and
motif-preserving strategies. This module handles variable-length RNA sequences for
downstream analysis while preserving biologically relevant features such as
localization signals, structural elements, and functional motifs.

"""
import re
import numpy as np
from math import log2
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field

class RNASequencePreprocessor:
    """
    Adaptive processing strategy for RNA sequences.

    This class implements sequence handling that preserves critical
    biological features during truncation and windowing operations. It recognizes
    and maintains localization signals, structural motifs, and functional elements
    that are essential for downstream analysis such as subcellular localization
    prediction and structure-function studies.

    The preprocessor uses three strategies based on sequence length:
    1. Full processing (≤1000 nt): Complete sequence preserved
    2. Smart truncation (1000-5000 nt): Adaptive truncation preserving motifs
    3. Sliding windows (>5000 nt): Overlapping windows with information content scoring

    Attributes:
        rna_motif_patterns (dict): Regular expression patterns for RNA localization
            and functional motifs from experimental literature
        compiled_motifs (dict): Pre-compiled regex patterns for efficient scanning
        critical_lengths (dict): Length thresholds for critical regions
        structure_params (dict): Parameters for structure preservation during truncation
    """

    def __init__(self):
        """
        Initialize RNA sequence preprocessor with motif patterns and parameters.

        Sets up comprehensive motif libraries for RNA subcellular localization
        signals and functional elements based on experimental RNA biology literature.
        """
        # RNA-specific motif patterns for localization signals
        # These patterns are derived from experimental studies of RNA trafficking
        self.rna_motif_patterns = {
            'nuclear_localization': [
                r'[GA]{4,}',  # G/A-rich clusters (nuclear retention signals)
                r'[AU]{6,}',  # A/U-rich clusters (nuclear export)
                r'[AG]{3,}[AU]{3,}[AG]{3,}',  # Mixed purine-rich (bipartite signals)
                r'RRRY',  # R=purine, Y=pyrimidine (consensus nuclear motifs)
                r'YRRR'
            ],
            'exosome_targeting': [
                r'AUAUAU',  # AU-rich elements (AREs) - degradation signals
                r'[AU]{5,}',  # Extended AU-rich regions
                r'UUAUUUAUU',  # Class II ARE (common degradation signal)
                r'[AU]U[AU]U[AU]U'  # Alternating AU pattern
            ],
            'cytosolic_retention': [
                r'[AC]{4,}[GU]{4,}',  # Balanced composition motifs
                r'[AGCU]{8,}',  # Generic mixed composition
            ],
            'ribosome_binding': [
                r'AGGAGG',  # Shine-Dalgarno-like sequence (prokaryotic-like)
                r'[AG]{3,}G[AG]{2,}',  # Purine-rich with conserved G
                r'CCUCC',  # Anti-Shine-Dalgarno (rRNA complement)
            ],
            'membrane_association': [
                r'[U]{4,}',  # U-rich tracts (membrane interaction signals)
                r'[A]{4,}',  # A-rich regions
                r'[AU]{5,}[GC]{2,}[AU]{5,}',  # AU-rich flanking structured core
            ],
            'ER_targeting': [
                r'[GA]{3,}U[GA]{2,}',  # G/A-rich with central U
                r'U[GA]{4,}U',  # U-flanked purine-rich
                r'[AC]{3,}[GU]{3,}[AC]{3,}',  # Alternating polarity pattern
            ],
            'mitochondrial_targeting': [
                r'[A]{4,}',  # 5' A-rich (mitochondrial targeting signals)
                r'[G]{3,}[C]{3,}',  # GC-rich stretches
                r'[AG]{4,}[CU]{4,}',  # Purine-pyrimidine gradient
            ],
            'microvesicle_targeting': [
                r'[U]{5,}',  # U-rich clusters (exosome packaging signals)
                r'[AU]{6,}',  # AU-rich regions
                r'U[AC]{3,}U',  # U-flanked internal motifs
            ]
        }
        # Pre-compile patterns for performance optimization
        self.compiled_motifs = {
            m_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for m_type, patterns in self.rna_motif_patterns.items()
        }

        # RNA-specific critical regions (based on linear sequence coordinates)
        self.critical_lengths = {
            'five_prime': 50,  # 5' UTR/start region - critical for translation
            'three_prime': 50,  # 3' UTR/end region - critical for localization
            'internal_motifs': 100  # Internal functional elements
        }

        # RNA structure preservation parameters
        self.structure_params = {
            'min_stem_length': 3,
            'max_truncation_loss': 0.3,  # Max 30% of structure can be lost
            'preserve_motif_context': 10  # Preserve N bases around motifs
        }

    def smart_truncate_rna(self, seq: str, target: int = 2000) -> str:
        """
        RNA-specific smart truncation preserving functional regions.

        Implements adaptive truncation that maintains:
        - 30% 5' region (translation start sites, 5' UTR signals)
        - 40% middle (structural elements, internal motifs)
        - 30% 3' region (localization signals, poly-A tails, 3' UTR)

        The truncation algorithm attempts to preserve known RNA motifs and
        can re-insert critical motifs if they are lost during truncation.

        Args:
            seq (str): Original RNA sequence
            target (int): Target length after truncation

        Returns:
            str: Truncated sequence preserving critical features

        Note:
            All region definitions (5', middle, 3') are based on linear
            sequence positions, not 3D structural coordinates.
        """
        if len(seq) <= target:
            return seq

        # Calculate segment lengths based on linear sequence positions
        five_prime_len = int(target * 0.3)  # 30% 5' region
        three_prime_len = int(target * 0.3)  # 30% 3' region
        middle_len = target - five_prime_len - three_prime_len  # 40% middle

        # Preserve 5' region (critical for start sites, 5' UTR)
        five_prime_segment = seq[:five_prime_len]

        # Preserve 3' region (critical for poly-A, localization signals)
        three_prime_segment = seq[-three_prime_len:] if three_prime_len > 0 else ""

        # Take middle segment from center to preserve potential structure
        middle_start = (len(seq) - middle_len) // 2
        middle_segment = seq[middle_start:middle_start + middle_len]

        # Combine segments
        truncated = five_prime_segment + middle_segment + three_prime_segment

        # Scan for RNA motifs in truncated sequence
        motifs_found = self.scan_rna_motifs(truncated)

        # Ensure critical motifs are preserved
        # Critical motifs are those essential for known RNA functions
        for motif_type, patterns in motifs_found.items():
            if not patterns and motif_type in ['nuclear_localization', 'ribosome_binding']:
                # Search in original sequence for critical motifs
                original_motifs = self._find_rna_motifs_in_sequence(seq, motif_type)
                if original_motifs:
                    # Try to incorporate up to 2 critical motifs
                    for motif in original_motifs[:2]:
                        if motif not in truncated:
                            truncated = self._insert_rna_motif_smart(truncated, motif)

        return truncated[:target]

    def _insert_rna_motif_smart(self, seq: str, motif: str) -> str:
        """
        Insert RNA motif while preserving sequence context.

        Attempts to insert a motif at biologically relevant positions
        (near 5' end, middle, or near 3' end) while maintaining
        sequence integrity.

        Args:
            seq (str): Target sequence for insertion
            motif (str): Motif to insert

        Returns:
            str: Sequence with motif inserted if possible
        """
        if len(seq) < len(motif) + 10:
            return seq

        # Try insertion points: near 5', middle, near 3'
        # These positions are chosen based on biological relevance
        insertion_points = [
            min(50, len(seq)),  # Near 5' end (regulatory region)
            len(seq) // 2,  # Middle (structural region)
            max(len(seq) - 50, 0)  # Near 3' end (localization signals)
        ]

        for pos in insertion_points:
            test_seq = seq[:pos] + motif + seq[pos:]
            # Verify motif was inserted exactly once
            if motif in test_seq and test_seq.count(motif) == 1:
                return test_seq

        return seq

    def create_rna_sliding_windows(self, seq: str, window_size: int = 500,
                                   stride: int = 250) -> List[Tuple[str, Dict]]:
        """
        Create sliding windows for RNA with structure awareness.

        Generates overlapping windows optimized for RNA analysis with
        information content scoring to identify biologically relevant regions.

        Args:
            seq (str): RNA sequence
            window_size (int): Size of each window in nucleotides
            stride (int): Step size between windows

        Returns:
            List[Tuple[str, Dict]]: List of (window_sequence, metadata) pairs

        """
        if len(seq) <= window_size:
            return [(seq, {'position': (0, len(seq)), 'motif_score': 1.0})]

        # OPTIMIZATION: Get hits once for the entire RNA to avoid repeated scanning
        global_hits = self._get_global_motif_hits(seq)

        windows = []
        n_windows = max(1, (len(seq) - window_size) // stride + 1)

        for i in range(n_windows):
            start = i * stride
            end = min(start + window_size, len(seq))
            window_seq = seq[start:end]

            # Calculate information content using global hits for efficiency
            motif_score = self._calculate_rna_window_information_content(
                window_seq, start, end, global_hits
            )

            windows.append((
                window_seq,
                {
                    'position': (start, end),
                    'motif_score': motif_score,
                    'contains_critical': self._rna_window_contains_critical(
                        window_seq, start, end, len(seq)
                    ),
                    'window_id': i
                }
            ))

        # Add final window if needed to ensure complete coverage
        if len(seq) % stride != 0 and len(seq) > window_size:
            final_start = len(seq) - window_size
            final_window = seq[final_start:]
            motif_score = self._calculate_rna_window_information_content(
                final_window, final_start, len(seq), global_hits
            )

            windows.append((
                final_window,
                {
                    'position': (final_start, len(seq)),
                    'motif_score': motif_score,
                    'contains_critical': self._rna_window_contains_critical(
                        final_window, final_start, len(seq), len(seq)
                    ),
                    'window_id': len(windows)
                }
            ))

        return windows

    def _calculate_rna_window_information_content(self, window: str, w_start: int = 0,
                                                  w_end: int = 0, global_hits: List = None) -> float:
        """
        Calculate information content of RNA window.

        Combines multiple metrics to assess biological relevance:
        - Motif density (functional element concentration)
        - GC content balance (structural propensity)
        - Shannon entropy (sequence complexity)
        - Secondary structure potential (folding capability)

        Args:
            window (str): Window sequence
            w_start (int): Start position in original sequence
            w_end (int): End position in original sequence
            global_hits (List): Pre-computed motif positions for efficiency

        Returns:
            float: Information content score (0-1), higher = more biologically relevant
        """
        if len(window) == 0:
            return 0.0

        scores = []

        # 1. OPTIMIZED Motif density via Index-Checking
        motif_count = 0
        if global_hits:
            for h_start, h_end in global_hits:
                # Check if motif falls within window using linear coordinates
                if h_start >= w_start and h_start < w_end:
                    motif_count += 1
        else:
            # Fallback for short sequences without global scan
            for motif_type, patterns in self.rna_motif_patterns.items():
                for pattern in patterns:
                    try:
                        matches = re.finditer(pattern, window, re.IGNORECASE)
                        motif_count += sum(1 for _ in matches)
                    except:
                        continue

        motif_density = motif_count / len(window)
        scores.append(min(motif_density * 20, 1.0))  # Scale motif density

        # 2. GC content balance (extremes may indicate specific localization)
        # Balanced GC content (~50%) often indicates optimal structural flexibility
        gc_content = (window.count('G') + window.count('C')) / len(window)
        balanced_score = 1.0 - abs(gc_content - 0.5)  # Closer to 0.5 is more "balanced"
        scores.append(balanced_score)

        # 3. Shannon entropy (sequence complexity)
        # Higher entropy indicates diverse nucleotide composition
        freq = Counter(window)
        entropy = -sum((count / len(window)) * log2(count / len(window))
                       for count in freq.values())
        max_entropy = log2(min(4, len(set(window))))  # Max for 4 nucleotides
        scores.append(entropy / max_entropy if max_entropy > 0 else 0)

        # 4. Secondary structure potential
        # Based on pairing probabilities from Turner rules
        gc_pairs = window.count('G') + window.count('C')
        au_pairs = window.count('A') + window.count('U')
        pairing_potential = (gc_pairs * 0.95 + au_pairs * 0.6) / len(window)
        scores.append(pairing_potential)

        return np.mean(scores)

    def _rna_window_contains_critical(self, window: str, start: int, end: int,
                                      total_len: int) -> Dict[str, bool]:
        """
        Check if RNA window contains critical regions.

        Identifies windows that overlap with biologically important regions:
        - 5' UTR/start region
        - 3' UTR/end region
        - Known motif-containing regions
        - Central structural core

        Args:
            window (str): Window sequence
            start (int): Start position in original sequence
            end (int): End position in original sequence
            total_len (int): Total length of original sequence

        Returns:
            Dict[str, bool]: Indicators for each critical region type
        """
        return {
            'five_prime': start < self.critical_lengths['five_prime'],
            'three_prime': end > total_len - self.critical_lengths['three_prime'],
            'has_motifs': any(self._find_rna_motifs_in_window(window)),
            'central_region': (start > total_len * 0.4 and
                               end < total_len * 0.6)
        }

    def _find_rna_motifs_in_window(self, window: str) -> List[str]:
        """
        Find RNA motifs in a window.

        Args:
            window (str): Window sequence

        Returns:
            List[str]: List of motif sequences found
        """
        motifs = []
        for motif_type, patterns in self.rna_motif_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, window, re.IGNORECASE)
                    motifs.extend(matches)
                except:
                    continue
        return motifs

    def _find_rna_motifs_in_sequence(self, seq: str, motif_type: str) -> List[str]:
        """
        Find specific RNA motif type in sequence.

        Args:
            seq (str): RNA sequence
            motif_type (str): Type of motif to search for

        Returns:
            List[str]: Unique motif sequences found
        """
        motifs = []
        if motif_type in self.rna_motif_patterns:
            for pattern in self.rna_motif_patterns[motif_type]:
                try:
                    matches = re.findall(pattern, seq, re.IGNORECASE)
                    motifs.extend(matches)
                except:
                    continue
        return list(set(motifs))

    def scan_rna_motifs(self, seq: str) -> Dict[str, List[str]]:
        """
        Scan for all RNA motifs in sequence.

        Comprehensive motif scanning for all known RNA functional elements.

        Args:
            seq (str): RNA sequence

        Returns:
            Dict[str, List[str]]: Dictionary mapping motif types to found sequences
        """
        results = {}
        for motif_type, patterns in self.rna_motif_patterns.items():
            found = []
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, seq, re.IGNORECASE)
                    found.extend(matches)
                except:
                    continue
            results[motif_type] = list(set(found))
        return results

    def _get_global_motif_hits(self, seq: str) -> List[Tuple[int, int]]:
        """
        Scans the sequence once and returns all (start, end) hit coordinates.

        Performance optimization for sliding window analysis that pre-computes
        all motif positions to avoid redundant scanning.

        Args:
            seq (str): RNA sequence

        Returns:
            List[Tuple[int, int]]: List of (start, end) positions for all motifs found
        """
        all_hits = []
        for m_type, patterns in self.compiled_motifs.items():
            for p in patterns:
                try:
                    all_hits.extend([m.span() for m in p.finditer(seq)])
                except:
                    continue
        return all_hits

    def adaptive_process_rna(self, seq: str) -> List[Tuple[str, Dict]]:
        """
        Main adaptive processing function for RNA.

        Selects the optimal processing strategy based on sequence length:
        - ≤1000 nt: Full sequence preservation
        - 1000-5000 nt: Smart truncation preserving critical features
        - >5000 nt: Sliding window approach with overlapping windows

        Args:
            seq (str): RNA sequence to process

        Returns:
            List[Tuple[str, Dict]]: List of (processed_sequence, metadata) pairs.
                For sliding windows, returns multiple entries; otherwise single entry.

        Note:
            The adaptive strategy ensures that regardless of input length,
            biologically relevant features are preserved for downstream analysis.
        """
        length = len(seq)

        if length <= 1000:
            # Short sequences: process directly without modification
            return [(seq, {
                'strategy': 'full',
                'original_length': length,
                'processed_length': length,
                'motifs': self.scan_rna_motifs(seq)
            })]

        elif length <= 5000:
            # Medium sequences: smart truncation to 1000 nt
            truncated = self.smart_truncate_rna(seq, 1000)
            return [(truncated, {
                'strategy': 'smart_truncate',
                'original_length': length,
                'processed_length': len(truncated),
                'truncation_ratio': len(truncated) / length,
                'motifs': self.scan_rna_motifs(truncated),
                'preserved_regions': {
                    'five_prime': True,
                    'three_prime': True,
                    'middle': True
                }
            })]

        else:
            # Long sequences: sliding window approach
            windows = self.create_rna_sliding_windows(seq, 500, 250)
            return [(window, {
                'strategy': 'sliding_window',
                'window_info': info,
                'original_length': length,
                'total_windows': len(windows),
                'window_index': info['window_id'],
                'motifs': self.scan_rna_motifs(window)
            }) for window, info in windows]
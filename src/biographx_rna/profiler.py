"""
Author: Abubakar Saeed
Date: February 23, 2026
Description: RNA motif profiling module that detects and scores subcellular localization
signals based on sequence motifs and physicochemical properties. This module implements
knowledge-based scoring functions for nine distinct RNA compartments derived from
experimental RNA biology literature.

"""
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
from src.biographx_rna.biophysics import RNAPhysicsStrategy

class RNAMotifProfiler:
    """
    Handles RNA motif scanning and localization signal detection.

    This class implements scoring functions for nine RNA subcellular compartments:
    Nucleus, Exosome, Cytosol, Cytoplasm, Ribosome, Membrane, Endoplasmic reticulum,
    Microvesicles, and Mitochondrion. Each scoring function combines:

    1. Known sequence motifs from experimental literature
    2. Nucleotide composition biases
    3. GC content preferences
    4. Positional features (e.g., motifs near 5' end)

    The scores are designed to be interpretable (0-1 range) and can be combined
    with graph-based features for enhanced prediction accuracy.

    Attributes:
        rna_physics (RNAPhysicsStrategy): Reference to RNA physics rules and
            localization configurations
    """

    def __init__(self, rna_physics_strategy: RNAPhysicsStrategy):
        """
        Initialize RNA motif profiler with physics strategy.

        Args:
            rna_physics_strategy: RNAPhysicsStrategy instance containing
                localization configurations and nucleotide properties
        """
        self.rna_physics = rna_physics_strategy

    def score_nucleus(self, sequence: str) -> float:
        """
        Score for nuclear localization.

        Nuclear localization signals in RNA include:
        - G/A-rich clusters (often in snRNAs and snoRNAs)
        - Moderate GC content (35-60%)
        - Specific purine-pyrimidine patterns (RRRY, YRRR)
        - Longer sequences typical of nuclear transcripts

        Args:
            sequence (str): RNA sequence

        Returns:
            float: Nuclear localization score (0-1), higher = more likely nuclear
        """
        seq = sequence.upper()
        scores = []

        # 1. G/A-rich clusters (common in nuclear RNAs like snRNAs, snoRNAs)
        ga_matches = len(re.findall(r'[GA]{4,}', seq))
        if ga_matches > 0:
            scores.append(min(ga_matches / 5.0, 1.0))

        # 2. Balanced GC content (nuclear RNAs often moderate GC)
        # Too high GC or too low AU is atypical for most nuclear transcripts
        gc_content = self.rna_physics.calculate_gc_content(seq)
        if 0.35 <= gc_content <= 0.6:
            scores.append(0.7)

        # 3. Specific nuclear motifs from experimental studies
        nuclear_motifs = [
            r'RRRY', r'YRRR',  # Purine-rich with pyrimidine (R=purine, Y=pyrimidine)
            r'[AG]{3,}[AU]{3,}[AG]{3,}',  # Alternating pattern found in nuclear retention signals
        ]

        motif_score = 0.0
        for pattern in nuclear_motifs:
            if re.search(pattern, seq):
                motif_score = max(motif_score, 0.6)

        scores.append(motif_score)

        # 4. Length consideration (nuclear RNAs can be long, e.g., pre-mRNA)
        length_score = min(len(seq) / 2000.0, 1.0)
        scores.append(length_score * 0.3)  # Weighted contribution

        return np.mean(scores) if scores else 0.0

    def score_exosome(self, sequence: str) -> float:
        """
        Score for exosome targeting/degradation.

        Exosome-targeted RNAs typically contain:
        - AU-rich elements (AREs) - primary degradation signals
        - Low GC content (AU-rich overall)
        - Specific instability motifs

        Args:
            sequence (str): RNA sequence

        Returns:
            float: Exosome targeting score (0-1), higher = more likely degraded
        """
        seq = sequence.upper()
        scores = []

        # 1. AU-rich elements (AREs) - primary degradation signal
        # These are well-characterized in mRNA turnover
        are_patterns = [
            r'AUAUAU',  # Basic ARE pattern
            r'[AU]{5,}',  # Extended AU-rich region
            r'UUAUUUAUU',  # Class II ARE (strong degradation signal)
        ]

        are_score = 0.0
        for pattern in are_patterns:
            matches = len(re.findall(pattern, seq))
            are_score += matches * 0.2

        scores.append(min(are_score, 1.0))

        # 2. Low GC content (degradation-prone RNAs often AU-rich)
        gc_content = self.rna_physics.calculate_gc_content(seq)
        if gc_content < 0.4:
            scores.append(0.6)

        # 3. Instability motifs (contexts that enhance degradation)
        if re.search(r'[AU]{3,}[GC][AU]{3,}', seq):
            scores.append(0.5)

        return np.mean(scores) if scores else 0.0

    def score_cytosol(self, sequence: str) -> float:
        """
        Score for cytosolic localization.

        Cytosolic RNAs (non-compartmentalized) typically:
        - Have balanced nucleotide composition
        - Moderate GC content (40-60%)
        - Lack strong targeting signals to other compartments

        Args:
            sequence (str): RNA sequence

        Returns:
            float: Cytosolic localization score (0-1)
        """
        seq = sequence.upper()
        scores = []

        # 1. Balanced nucleotide composition
        # Extreme biases suggest specific targeting
        counts = Counter(seq)
        balanced = True
        for nt in ['A', 'U', 'G', 'C']:
            freq = counts.get(nt, 0) / len(seq)
            if freq > 0.5 or freq < 0.1:  # Too extreme
                balanced = False

        if balanced:
            scores.append(0.7)

        # 2. Moderate GC content
        gc_content = self.rna_physics.calculate_gc_content(seq)
        if 0.4 <= gc_content <= 0.6:
            scores.append(0.6)

        # 3. No strong targeting signals (negative scoring)
        # Check for motifs characteristic of other compartments
        has_strong_signals = (
                len(re.findall(r'[GA]{4,}', seq)) > 2 or  # Nuclear
                len(re.findall(r'AUAUAU', seq)) > 1 or  # Exosome
                len(re.findall(r'AGGAGG', seq)) > 0  # Ribosome
        )

        if not has_strong_signals:
            scores.append(0.5)

        return np.mean(scores) if scores else 0.0

    def score_cytoplasm(self, sequence: str) -> float:
        """
        Score for cytoplasmic localization.

        Cytoplasmic localization (broader than cytosol) features:
        - A/U-rich clusters (common in mRNAs)
        - Moderate sequence length
        - Often shares features with cytosolic RNAs

        Args:
            sequence (str): RNA sequence

        Returns:
            float: Cytoplasmic localization score (0-1)
        """
        seq = sequence.upper()

        # Similar to cytosol but may have different biases
        cytosol_score = self.score_cytosol(sequence)

        # Additional cytoplasm-specific features
        scores = [cytosol_score]

        # A/U-rich clusters (common in cytoplasmic mRNAs)
        au_clusters = len(re.findall(r'[AU]{4,}', seq))
        if au_clusters > 0:
            scores.append(min(au_clusters / 3.0, 0.7))

        # Moderate length (cytoplasmic mRNAs typically 500-3000 nt)
        length_norm = min(len(seq) / 1500.0, 1.0)
        scores.append(length_norm * 0.4)

        return np.mean(scores)

    def score_ribosome(self, sequence: str) -> float:
        """
        Score for ribosome association/translation.

        Ribosome-associated RNAs feature:
        - Shine-Dalgarno-like sequences near 5' end
        - Start codon (AUG) with favorable context (Kozak)
        - Higher GC content (structural stability)

        Args:
            sequence (str): RNA sequence

        Returns:
            float: Ribosome association score (0-1)
        """
        seq = sequence.upper()
        scores = []

        # 1. Shine-Dalgarno-like sequences
        # Note: Position near 5' end is important (linear sequence position)
        sd_patterns = [
            r'AGGAGG',  # Classic Shine-Dalgarno
            r'GGAGG',  # Minimal SD
            r'[AG]{3,}G[AG]{2,}',  # Purine-rich with G
        ]

        sd_score = 0.0
        for pattern in sd_patterns:
            # Look near 5' end (first 100 nucleotides in linear sequence)
            if re.search(pattern, seq[:100]):
                sd_score = 0.8

        scores.append(sd_score)

        # 2. Start codon context
        if 'AUG' in seq:
            # Check for good Kozak context (translation initiation)
            start_pos = seq.find('AUG')
            if start_pos > 0 and start_pos < len(seq) - 3:
                # Check -3 position (A or G preferred in Kozak consensus)
                if start_pos >= 3 and seq[start_pos - 3] in ['A', 'G']:
                    scores.append(0.6)
                # Check +4 position (G preferred)
                if start_pos + 6 < len(seq) and seq[start_pos + 3] == 'G':
                    scores.append(0.5)

        # 3. GC content (ribosomal RNAs and translated regions are GC-rich)
        gc_content = self.rna_physics.calculate_gc_content(seq)
        if gc_content > 0.5:
            scores.append(min((gc_content - 0.5) * 2, 0.7))

        return np.mean(scores) if scores else 0.0

    def score_membrane(self, sequence: str) -> float:
        """
        Score for membrane association.

        Membrane-associated RNAs often have:
        - U-rich clusters (interaction with membrane proteins)
        - Higher hydrophobicity index
        - Specific membrane interaction motifs

        Args:
            sequence (str): RNA sequence

        Returns:
            float: Membrane association score (0-1)
        """
        seq = sequence.upper()
        scores = []

        # 1. U-rich clusters (common in membrane-associated RNAs)
        u_rich = len(re.findall(r'U{4,}', seq))
        scores.append(min(u_rich / 2.0, 0.8))

        # 2. Hydrophobicity index (approximate)
        # Based on nucleotide hydrophobicity scales
        hydro_score = 0.0
        for nt in seq:
            hydro_score += self.rna_physics.rna_properties['hydrophobicity'].get(nt, 0)
        hydro_score /= len(seq)
        scores.append(hydro_score)

        # 3. Specific membrane motifs from experimental studies
        membrane_motifs = [
            r'[AU]{5,}[GC]{2,}[AU]{5,}',  # AU-rich flanking structured core
            r'U[ACGU]{3,}U',  # U-flanked internal region
        ]

        for pattern in membrane_motifs:
            if re.search(pattern, seq):
                scores.append(0.6)
                break

        return np.mean(scores) if scores else 0.0

    def score_er(self, sequence: str) -> float:
        """
        Score for endoplasmic reticulum targeting.

        ER-targeted RNAs contain:
        - Signal recognition particle (SRP) binding signals near 5' end
        - ER retention motifs
        - Moderate GC content

        Args:
            sequence (str): RNA sequence

        Returns:
            float: ER targeting score (0-1)
        """
        seq = sequence.upper()
        scores = []

        # 1. Signal recognition particle (SRP) binding signals
        # These are typically near the 5' end (linear sequence position)
        srp_patterns = [
            r'[GA]{3,}U[GA]{2,}',  # Purine-rich with central U
            r'U[GA]{4,}U',  # U-flanked purine-rich
        ]

        srp_score = 0.0
        for pattern in srp_patterns:
            # Signal usually near 5' end (first 150 nucleotides)
            if re.search(pattern, seq[:150]):
                srp_score = 0.8

        scores.append(srp_score)

        # 2. ER retention signals
        if re.search(r'[AC]{3,}[GU]{3,}[AC]{3,}', seq):
            scores.append(0.6)

        # 3. Moderate GC content
        gc_content = self.rna_physics.calculate_gc_content(seq)
        if 0.3 <= gc_content <= 0.6:
            scores.append(0.5)

        return np.mean(scores) if scores else 0.0

    def score_microvesicles(self, sequence: str) -> float:
        """
        Score for microvesicle/exosome packaging.

        RNAs packaged into microvesicles often have:
        - U-rich clusters (packaging signals)
        - AU-rich elements (AREs)
        - Specific export motifs

        Args:
            sequence (str): RNA sequence

        Returns:
            float: Microvesicle packaging score (0-1)
        """
        seq = sequence.upper()
        scores = []

        # 1. U-rich clusters (common in secreted RNAs)
        u_rich = len(re.findall(r'U{4,}', seq))
        scores.append(min(u_rich / 2.0, 0.8))

        # 2. AU-rich elements (also associated with secretion)
        are_score = len(re.findall(r'AUAUAU', seq)) * 0.3
        scores.append(min(are_score, 0.7))

        # 3. Specific export motifs from exosome studies
        export_patterns = [
            r'[AU]{6,}',  # Extended AU regions
            r'U[AC]{3,}U',  # U-flanked internal motif
        ]

        for pattern in export_patterns:
            if re.search(pattern, seq):
                scores.append(0.6)
                break

        return np.mean(scores) if scores else 0.0

    def score_mitochondrion(self, sequence: str) -> float:
        """
        Score for mitochondrial targeting.

        Mitochondrial RNAs feature:
        - A-rich 5' region (targeting signal)
        - High GC content (structural stability in mitochondrial environment)
        - Specific mitochondrial motifs
        - Shorter length (typical of mitochondrial transcripts)

        Args:
            sequence (str): RNA sequence

        Returns:
            float: Mitochondrial targeting score (0-1)
        """
        seq = sequence.upper()
        scores = []

        # 1. A-rich 5' region (characteristic of mitochondrial targeting)
        # Position near 5' end matters in linear sequence
        if len(seq) > 30:
            five_prime = seq[:30]
            a_content = five_prime.count('A') / 30
            if a_content > 0.4:
                scores.append(0.7)

        # 2. High GC content (mitochondrial RNAs often GC-rich for stability)
        gc_content = self.rna_physics.calculate_gc_content(seq)
        if gc_content > 0.5:
            scores.append(min((gc_content - 0.5) * 2, 0.8))

        # 3. Specific mitochondrial motifs
        mito_patterns = [
            r'[AG]{4,}[CU]{4,}',  # Purine-pyrimidine boundary
            r'[G]{3,}[C]{3,}',  # GC-rich stretch
        ]

        for pattern in mito_patterns:
            if re.search(pattern, seq):
                scores.append(0.6)
                break

        # 4. Length (mitochondrial RNAs are often shorter)
        if len(seq) < 1000:
            scores.append(0.5)

        return np.mean(scores) if scores else 0.0

    def extract_rna_knowledge_profiles(self, sequence: str, hybrid_scores: Dict[str, float]) -> np.ndarray:
        """
        Calculate enhanced profile scores for all 9 RNA compartments.

        Combines three types of evidence for each compartment:
        1. Motif-based scores (sequence pattern matching)
        2. Hybrid interaction scores (from graph-based analysis)
        3. GC content compatibility (with compartment-specific ranges)

        This creates a comprehensive profile vector suitable for machine learning
        input or ensemble prediction methods.

        Args:
            sequence (str): RNA sequence
            hybrid_scores (Dict[str, float]): Dictionary of hybrid interaction scores
                from graph-based analysis (keys: hybrid interaction types)

        Returns:
            np.ndarray: 27-dimensional feature vector (3 features × 9 compartments)
                Order matches self.rna_physics.localization_configs keys

        Note:
            The three feature types for each compartment are concatenated in order:
            [motif_score1, hybrid_score1, gc_score1, motif_score2, hybrid_score2, ...]
        """
        features = []

        # Map class names to scoring functions
        scorers = {
            "Nucleus": self.score_nucleus,
            "Exosome": self.score_exosome,
            "Cytosol": self.score_cytosol,
            "Cytoplasm": self.score_cytoplasm,
            "Ribosome": self.score_ribosome,
            "Membrane": self.score_membrane,
            "Endoplasmic.reticulum": self.score_er,
            "Microvesicles": self.score_microvesicles,
            "Mitochondrion": self.score_mitochondrion
        }

        # For each compartment, add three features
        for loc, config in self.rna_physics.localization_configs.items():
            # 1. Motif-based score (sequence pattern recognition)
            if loc in scorers:
                motif_score = scorers[loc](sequence)
            else:
                motif_score = 0.0
            features.append(motif_score)

            # 2. Hybrid interaction score (graph-based structural evidence)
            # Average of all expected hybrid interactions for this compartment
            hyb_list = [hybrid_scores[h] for h in config['expected_hybrids'] if h in hybrid_scores]
            hyb_score = np.mean(hyb_list) if hyb_list else 0.0
            features.append(hyb_score)

            # 3. GC content compatibility (with compartment-specific range)
            # Score is 1.0 at optimal GC, decreasing linearly to 0 at range boundaries
            gc_content = self.rna_physics.calculate_gc_content(sequence)
            gc_min, gc_max = config['gc_content_range']
            if gc_min <= gc_content <= gc_max:
                # Linear scaling within acceptable range
                gc_compat = 1.0 - abs(gc_content - (gc_min + gc_max) / 2) / ((gc_max - gc_min) / 2)
            else:
                gc_compat = 0.0
            features.append(gc_compat)

        return np.array(features, dtype=np.float32)
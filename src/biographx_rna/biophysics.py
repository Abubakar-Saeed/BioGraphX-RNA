"""
Author: Abubakar Saeed
Date: February 23, 2026
Description: RNA biophysics strategy module containing comprehensive rules and constants
for RNA-specific interactions, physicochemical properties, and structural parameters.
This module serves as the central repository for all RNA biology rules used in
secondary structure prediction and interaction analysis.

Note: All distance parameters in this module refer to linear sequence distances
(number of residues between positions), not 3D spatial distances measured in Angstroms.
This is consistent with RNA secondary structure analysis where contacts are defined
by sequence proximity rather than physical 3D space.
"""
from math import sin, cos, radians, sqrt, log2
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class RNAPhysicsStrategy:
    """
    Central repository for all RNA biophysical rules and constants.

    This class implements RNA-specific interaction rules based on established
    biochemical literature including canonical/non-canonical base pairing,
    base stacking energetics, and sequence-context dependent interactions.
    The rules are designed for RNA secondary structure analysis and subcellular
    localization prediction.

    Note on distance calculations: All distance thresholds (max_distance parameters)
    represent linear sequence separation in terms of nucleotide positions, not
    3D spatial distances in Angstroms. This is appropriate for analyzing
    secondary structure contacts where proximity in sequence determines
    potential base pairing interactions.

    Attributes:
        nucleotides (list): Standard RNA nucleotides ['A', 'U', 'G', 'C']
        interaction_rules (dict): Hierarchical rules for RNA-RNA interactions
        hybrid_interactions (dict): Combined interaction types with weighted contributions
        localization_configs (dict): RNA compartment-specific interaction preferences
        rna_properties (dict): Physicochemical properties of nucleotides
        structure_params (dict): Secondary structure folding parameters
    """

    def __init__(self):
        """
        Initialize RNA biophysics strategy with all interaction rules and constants.

        Sets up comprehensive rule sets for RNA base pairing, stacking, and
        hybrid interactions based on Turner rules and RNA biology literature.
        """
        # RNA nucleotides and their properties
        self.nucleotides = ['A', 'U', 'G', 'C']

        # 1. RNA-SPECIFIC INTERACTION RULES
        self.interaction_rules = {
            # Canonical Watson-Crick base pairing
            'canonical_wc': {
                'pairs': [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')],
                'strength': 1.0,
                'max_distance': 200,  # Long-range in secondary structure
                'bond_type': 'hydrogen_bond',
                'energy': -2.0  # kcal/mol approx
            },

            # Wobble base pairing (G-U)
            'wobble_pair': {
                'pairs': [('G', 'U'), ('U', 'G')],
                'strength': 0.8,
                'max_distance': 200,  # Linear sequence distance, not 3D Angstroms
                'bond_type': 'hydrogen_bond',
                'energy': -1.0
            },

            # Non-canonical base pairs from RNA biology literature
            'non_canonical': {
                'pairs': [
                    ('A', 'A'), ('A', 'G'), ('G', 'A'),  # Sheared G-A
                    ('G', 'G'), ('U', 'U'), ('C', 'C'),  # Homopurine/homopyrimidine
                    ('A', 'C'), ('C', 'A'),  # Trans Hoogsteen/Hoogsteen
                    ('G', 'U'), ('U', 'G'),  # Already in wobble, but included
                ],
                'strength': 0.6,
                'max_distance': 150,  # Linear sequence distance threshold
                'bond_type': 'hydrogen_bond',
                'energy': -0.5
            },

            # Base stacking interactions (adjacent or nearby in sequence/structure)
            'base_stacking': {
                'conditions': [
                    ('A', 'A'), ('U', 'U'), ('G', 'G'), ('C', 'C'),  # Self-stacking
                    ('A', 'G'), ('G', 'A'), ('C', 'U'), ('U', 'C'),  # Purine-purine, pyrimidine-pyrimidine
                ],
                'strength': 0.9,
                'max_distance': 3,  # Typically adjacent in sequence or structure
                'interaction_type': 'stacking',
                'energy': -1.5
            },

            # Phosphate backbone interactions
            'backbone': {
                'residues': ['A', 'U', 'G', 'C'],
                'strength': 0.75,
                'max_distance': 2,  # Linear sequence proximity for backbone contacts
                'interaction_type': 'electrostatic',
                'energy': -0.2
            },

        }

        # 2. HYBRID RNA INTERACTIONS
        self.hybrid_interactions = {
            'wc_stacking_hybrid': {
                'primary': 'canonical_wc',
                'secondary': 'base_stacking',
                'weight': 1.3,
                'description': 'WC pairing with adjacent stacking'
            },
            'wobble_stacking_hybrid': {
                'primary': 'wobble_pair',
                'secondary': 'base_stacking',
                'weight': 1.2,
                'description': 'Wobble pair with stacking stabilization'
            },
            'stacking_backbone_hybrid': {
                'primary': 'base_stacking',
                'secondary': 'backbone',
                'weight': 1.1,
                'description': 'Stacking with backbone stabilization'
            }
        }

        # 3. RNA LOCALIZATION CONFIGS
        # These configurations define expected interaction patterns and motifs
        # for different subcellular RNA compartments based on experimental data.
        self.localization_configs = {
            "Nucleus": {
                'expected_hybrids': ['wc_stacking_hybrid'],
                'key_motifs': ['AAAA', 'UUUU', 'RRRY', 'YRRR'],  # R=purine, Y=pyrimidine
                'gc_content_range': (0.3, 0.6)
            },
            "Exosome": {
                'expected_hybrids': ['wobble_stacking_hybrid', 'stacking_backbone_hybrid'],
                'key_motifs': ['AUAU', 'UAUA', 'UR-rich'],
                'gc_content_range': (0.2, 0.5)
            },
            "Cytosol": {
                'expected_hybrids': ['stacking_backbone_hybrid'],
                'key_motifs': ['generic', 'balanced'],
                'gc_content_range': (0.4, 0.6)
            },
            "Cytoplasm": {
                'expected_hybrids': ['wc_stacking_hybrid', 'stacking_backbone_hybrid'],
                'key_motifs': ['AA-rich', 'UU-rich'],
                'gc_content_range': (0.35, 0.55)
            },
            "Ribosome": {
                'expected_hybrids': ['wc_stacking_hybrid'],
                'key_motifs': ['Shine-Dalgarno', 'anti-Shine-Dalgarno', 'P-site', 'A-site'],
                'gc_content_range': (0.45, 0.7)
            },
            "Membrane": {
                'expected_hybrids': ['stacking_backbone_hybrid'],
                'key_motifs': ['U-rich', 'A-rich', 'hydrophobic_kmer'],
                'gc_content_range': (0.25, 0.5)
            },
            "Endoplasmic.reticulum": {
                'expected_hybrids': ['wc_stacking_hybrid', 'wobble_stacking_hybrid'],
                'key_motifs': ['ER_targeting', 'signal_sequence'],
                'gc_content_range': (0.3, 0.6)
            },
            "Microvesicles": {
                'expected_hybrids': ['stacking_backbone_hybrid'],
                'key_motifs': ['specific_export', 'U-rich_cluster'],
                'gc_content_range': (0.2, 0.5)
            },
            "Mitochondrion": {
                'expected_hybrids': ['wc_stacking_hybrid'],
                'key_motifs': ['mito_targeting', 'A-rich_5prime'],
                'gc_content_range': (0.4, 0.8)
            }
        }

        # 4. RNA PHYSICOCHEMICAL PROPERTIES
        self.rna_properties = {
            # Molecular weights (g/mol)
            'molecular_weight': {
                'A': 329.2, 'U': 306.2, 'G': 345.2, 'C': 305.2
            },

            # Hydrophobicity index (approximate)
            'hydrophobicity': {
                'A': 0.5, 'U': 0.8, 'G': 0.3, 'C': 0.7
            },

            # Stacking energies (kcal/mol, approximate)
            'stacking_energy': {
                ('A', 'A'): -1.5, ('U', 'U'): -1.0, ('G', 'G'): -2.0, ('C', 'C'): -1.8,
                ('A', 'U'): -1.2, ('U', 'A'): -1.2,
                ('G', 'C'): -2.5, ('C', 'G'): -2.5,
                ('G', 'U'): -0.8, ('U', 'G'): -0.8
            },

            # Base pairing probabilities (from Turner rules)
            'pairing_probability': {
                ('A', 'U'): 0.9, ('U', 'A'): 0.9,
                ('G', 'C'): 0.95, ('C', 'G'): 0.95,
                ('G', 'U'): 0.6, ('U', 'G'): 0.6
            },

            # Charge at pH 7 (phosphate backbone contributes -1 per nucleotide)
            'charge': {
                'A': -1, 'U': -1, 'G': -1, 'C': -1
            }
        }

        # 5. RNA SECONDARY STRUCTURE PARAMETERS
        self.structure_params = {
            'min_loop_length': 3,
            'max_loop_length': 30,
            'min_stem_length': 2,
            'max_bulge_size': 5,
            'temperature': 37.0,  # Celsius
            'salt_concentration': 1.0  # Molar
        }

    def check_interaction(self, nt1: str, nt2: str, interaction_type: str) -> bool:
        """
        Check RNA-specific interaction between two nucleotides.

        Args:
            nt1 (str): First nucleotide (A, U, G, or C)
            nt2 (str): Second nucleotide (A, U, G, or C)
            interaction_type (str): Type of interaction to check
                (e.g., 'canonical_wc', 'wobble_pair', 'base_stacking')

        Returns:
            bool: True if the interaction is possible between the nucleotides

        Note:
            This method checks interaction possibility based on nucleotide identity
            only.
        """
        rules = self.interaction_rules[interaction_type]

        if interaction_type in ['canonical_wc', 'wobble_pair', 'non_canonical']:
            # Check if (nt1, nt2) or (nt2, nt1) is in allowed pairs
            return (nt1, nt2) in rules['pairs'] or (nt2, nt1) in rules['pairs']

        elif interaction_type == 'base_stacking':
            # Check stacking conditions
            return (nt1, nt2) in rules['conditions'] or (nt2, nt1) in rules['conditions']

        elif interaction_type == 'backbone':
            # Always true for any nucleotide pair within distance
            return nt1 in rules['residues'] and nt2 in rules['residues']

        return False

    def calculate_gc_content(self, sequence: str) -> float:
        """
        Calculate GC content of RNA sequence.

        Args:
            sequence (str): RNA sequence string

        Returns:
            float: Fraction of G and C nucleotides in the sequence
        """
        if not sequence:
            return 0.0
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)

    def calculate_au_content(self, sequence: str) -> float:
        """
        Calculate AU content of RNA sequence.

        Args:
            sequence (str): RNA sequence string

        Returns:
            float: Fraction of A and U nucleotides in the sequence
        """
        if not sequence:
            return 0.0
        au_count = sequence.count('A') + sequence.count('U')
        return au_count / len(sequence)

    def calculate_mfe_per_nucleotide(self, sequence: str) -> float:
        """
        Approximate minimum free energy per nucleotide.

        Simplified version based on base composition and pairing using empirical
        approximations from Turner rules. This provides a rough estimate for
        comparative analysis rather than precise thermodynamic calculations.

        Args:
            sequence (str): RNA sequence string

        Returns:
            float: Estimated minimum free energy per nucleotide (kcal/mol)
        """
        if len(sequence) < 4:
            return 0.0

        # Approximate MFE based on GC content and length
        gc_content = self.calculate_gc_content(sequence)

        # Empirical approximation: GC pairs contribute ~-3 kcal/mol, AU ~-2 kcal/mol
        # Assume 50% of nucleotides are paired in optimal structure
        gc_pairs = gc_content * 0.5 * len(sequence)
        au_pairs = (1 - gc_content) * 0.5 * len(sequence)

        mfe_estimate = (gc_pairs * -3.0) + (au_pairs * -2.0)

        # Normalize by sequence length
        return mfe_estimate / len(sequence) if len(sequence) > 0 else 0.0

    def calculate_shannon_entropy(self, sequence: str) -> float:
        """
        Calculate Shannon entropy of nucleotide distribution.

        Measures the diversity and randomness of nucleotide composition.
        Higher entropy indicates more uniform distribution of nucleotides.

        Args:
            sequence (str): RNA sequence string

        Returns:
            float: Shannon entropy value (bits)
        """
        n = len(sequence)
        if n == 0:
            return 0.0
        counts = Counter(sequence)
        return -sum((c / n) * log2(c / n) for c in counts.values())

    def calculate_dinucleotide_frequency(self, sequence: str) -> Dict[str, float]:
        """
        Calculate dinucleotide frequencies.

        Computes normalized frequencies of all adjacent nucleotide pairs,
        which are important for capturing sequence context effects.

        Args:
            sequence (str): RNA sequence string

        Returns:
            Dict[str, float]: Dictionary mapping dinucleotides to their frequencies
        """
        freqs = {}
        total_pairs = max(1, len(sequence) - 1)

        for i in range(len(sequence) - 1):
            dinuc = sequence[i:i + 2]
            freqs[dinuc] = freqs.get(dinuc, 0) + 1

        # Normalize
        for dinuc in freqs:
            freqs[dinuc] /= total_pairs

        return freqs

    def calculate_rna_autocorrelation(self, seq: str, prop_map: Dict, lag_max: int = 10) -> List[float]:
        """
        Calculate autocorrelation of RNA properties.

        Measures how properties correlate along the sequence at different lags,
        capturing periodic patterns and sequence organization.

        Args:
            seq (str): RNA sequence string
            prop_map (Dict): Property mapping for each nucleotide
            lag_max (int): Maximum lag to calculate correlation for

        Returns:
            List[float]: Autocorrelation values for each lag
        """
        n = len(seq)
        if n < lag_max + 1:
            return [0.0] * lag_max

        try:
            vals = [prop_map.get(nt, 0) for nt in seq]
            mean_val = np.mean(vals)
            vals_centered = [x - mean_val for x in vals]
            variance = np.var(vals) if np.var(vals) > 0 else 1.0

            correlations = []
            for lag in range(1, lag_max + 1):
                s = sum(vals_centered[i] * vals_centered[i + lag] for i in range(n - lag))
                norm = (n - lag) * variance
                correlations.append(s / norm if norm > 0 else 0.0)

            return correlations
        except:
            return [0.0] * lag_max

    def calculate_base_pairing_potential(self, sequence: str, window_size: int = 50) -> List[float]:
        """
        Calculate local base pairing potential using sliding window.

        Estimates the propensity for base pairing in local sequence regions
        based on GC content and pairing probabilities.

        Args:
            sequence (str): RNA sequence string
            window_size (int): Size of sliding window for local calculation

        Returns:
            List[float]: Base pairing potential values for each window position
        """
        if len(sequence) < window_size:
            # For short sequences, calculate for the entire sequence
            window = sequence
            gc = window.count('G') + window.count('C')
            au = window.count('A') + window.count('U')
            potential = (gc * 0.95 + au * 0.6) / len(window) if len(window) > 0 else 0.0
            return [potential]

        potentials = []
        for i in range(0, len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            gc = window.count('G') + window.count('C')
            au = window.count('A') + window.count('U')

            # Pairing potential proportional to GC content and length
            # Note: 0.95 and 0.6 are pairing probabilities from Turner rules
            potential = (gc * 0.95 + au * 0.6) / window_size
            potentials.append(potential)

        return potentials if potentials else [0.0]

    def extract_global_rna_physics(self, sequence: str) -> np.ndarray:
        """
        Extract global physicochemical features for RNA.

        Comprehensive feature extraction for machine learning applications,
        combining composition, entropy, energy estimates, and correlation measures.
        Returns a fixed-length feature vector of 25 elements.

        Args:
            sequence (str): RNA sequence string

        Returns:
            np.ndarray: 25-dimensional feature vector of dtype float32

        Note:
            All features are derived from sequence information only, with no
            3D structural data required. Distance-based features use linear
            sequence positions.
        """
        features = []
        n = len(sequence)

        if n == 0:
            return np.zeros(25, dtype=np.float32)

        # 1. Basic composition features
        gc_content = self.calculate_gc_content(sequence)
        au_content = self.calculate_au_content(sequence)
        features.extend([gc_content, au_content])

        # 2. Nucleotide frequencies
        counts = Counter(sequence)
        for nt in ['A', 'U', 'G', 'C']:
            features.append(counts.get(nt, 0) / n)

        # 3. GC skew - asymmetry in G vs C distribution
        g_count = counts.get('G', 0)
        c_count = counts.get('C', 0)
        gc_skew = (g_count - c_count) / max(1, g_count + c_count)
        features.append(gc_skew)

        # 4. AU skew - asymmetry in A vs U distribution
        a_count = counts.get('A', 0)
        u_count = counts.get('U', 0)
        au_skew = (a_count - u_count) / max(1, a_count + u_count)
        features.append(au_skew)

        # 5. Shannon entropy - sequence complexity
        entropy = self.calculate_shannon_entropy(sequence)
        features.append(entropy)

        # 6. Approximate MFE per nucleotide
        mfe_per_nt = self.calculate_mfe_per_nucleotide(sequence)
        features.append(mfe_per_nt)

        # 7. Dinucleotide bias (selected important ones)
        dinuc_freqs = self.calculate_dinucleotide_frequency(sequence)
        important_dinucs = ['AA', 'UU', 'GG', 'CC', 'AU', 'UA', 'GC', 'CG', 'GU', 'UG']
        for dinuc in important_dinucs:
            features.append(dinuc_freqs.get(dinuc, 0.0))

        # 8. Base pairing potential statistics
        pairing_potentials = self.calculate_base_pairing_potential(sequence, 30)
        if pairing_potentials:
            features.extend([
                float(np.mean(pairing_potentials)),
                float(np.std(pairing_potentials)) if len(pairing_potentials) > 1 else 0.0,
                float(np.max(pairing_potentials)) if pairing_potentials else 0.0,
                float(np.min(pairing_potentials)) if pairing_potentials else 0.0
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # 9. Autocorrelation features - captures periodic patterns
        gc_map = {'A': 0, 'U': 0, 'G': 1, 'C': 1}
        gc_corr = self.calculate_rna_autocorrelation(sequence, gc_map, lag_max=5)

        # Take only the first lag for GC_Autocorrelation_Lag1
        if gc_corr and len(gc_corr) >= 1:
            features.append(gc_corr[0])
        else:
            features.append(0.0)

        # Ensure exactly 25 features
        if len(features) > 25:
            features = features[:25]
        elif len(features) < 25:
            # Pad with zeros to reach 25 features
            features.extend([0.0] * (25 - len(features)))

        return np.array(features, dtype=np.float32)
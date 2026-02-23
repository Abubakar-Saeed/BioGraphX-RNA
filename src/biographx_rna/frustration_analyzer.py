"""
Author: Abubakar Saeed
Date: February 23, 2026
Description: RNA frustration analysis module that computes per-nucleotide local frustration
scores from RNA constraint graphs. Frustration in RNA arises from conflicting structural
constraints, competing interaction patterns, and sequence-structure mismatches.

"""
import igraph as ig
import numpy as np
import math
import re
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter
from src.biographx_rna.biophysics import RNAPhysicsStrategy

class RNAFrustrationAnalyzer:
    """
    Computes per-nucleotide local frustration scores from RNA constraint graphs.

    This class implements RNA-specific frustration analysis based on the concept
    that frustration occurs when a nucleotide experiences conflicting constraints
    from different interaction types, sequence context, or structural requirements.

    RNA-specific frustration arises from:
    1. Conflicting base pairing constraints (e.g., competing pairing partners)
    2. Stacking vs pairing trade-offs (energy minimization conflicts)
    3. Secondary structure vs sequence motif conflicts (functional vs structural)
    4. Long-range vs local interaction competition (distance-based conflicts)


    Attributes:
        rna_physics (RNAPhysicsStrategy): Reference to RNA physics rules for
            interaction energies and properties
    """

    def __init__(self, rna_physics_strategy: RNAPhysicsStrategy):
        """
        Initialize RNA frustration analyzer with physics strategy.

        Args:
            rna_physics_strategy: RNAPhysicsStrategy instance containing
                interaction rules and energy parameters
        """
        self.rna_physics = rna_physics_strategy

    def compute_from_rna_constraint_graph(self, graph: ig.Graph, sequence: str) -> Dict[str, float]:
        """
        Compute frustration features from an RNA constraint graph.

        Transforms a constraint graph representation into quantitative frustration
        metrics by analyzing conflicting interaction patterns, energy variances,
        and sequence-structure mismatches.

        Args:
            graph: RNA constraint graph (from RNAGraphEngine.build_rna_graph)
                  containing nodes as nucleotides and edges as interactions
            sequence: RNA sequence string (A, U, G, C)

        Returns:
            Dictionary of RNA frustration features including per-region statistics,
            nucleotide-type specific frustration, and hotspot detection

        Note:
            For very short sequences (<10 nt) or graphs with no edges,
            returns default frustration values.
        """
        n = len(sequence)
        if n < 10 or graph.ecount() == 0:
            return self._get_default_rna_frustration_features()

        # 1. Extract constraint weights as energy proxies
        edge_weights = graph.es["weight"] if "weight" in graph.es.attributes() else [1.0] * graph.ecount()
        interaction_types = graph.es["interaction_type"] if "interaction_type" in graph.es.attributes() else [
                                                                                                                 ""] * graph.ecount()

        # 2. Compute per-nucleotide frustration
        per_nucleotide_frustration = self._compute_per_nucleotide_frustration(
            graph, edge_weights, interaction_types, sequence
        )

        # 3. Extract RNA-specific frustration features
        features = self._extract_rna_frustration_features(
            per_nucleotide_frustration, sequence, graph, edge_weights
        )

        return features

    def _compute_per_nucleotide_frustration(self, graph, edge_weights, interaction_types, sequence):
        """
        Compute local frustration score for each nucleotide.

        Calculates frustration by analyzing:
        - Energy variance from competing interactions
        - Interaction type conflicts
        - Local sequence context constraints
        - Structural frustration potential

        Args:
            graph: igraph Graph object with RNA constraints
            edge_weights: List of edge weights (higher = stronger interaction)
            interaction_types: List of interaction type strings
            sequence: RNA sequence string

        Returns:
            np.ndarray: Frustration scores for each nucleotide (0-1 scale)
                       where 1 = highly frustrated, 0 = minimally frustrated
        """
        n = len(sequence)

        # Initialize frustration array
        frustration = np.zeros(n)

        # If graph has no edges, return baseline frustration based on sequence
        if graph.ecount() == 0:
            # Even with no edges, there can be frustration potential
            for i in range(n):
                frustration[i] = self._calculate_baseline_frustration(i, sequence)
            return frustration

        # Pre-calculate node degrees and neighbors
        node_degrees = graph.degree()

        for pos in range(n):
            # Get incident edges for this position
            incident_edges = graph.incident(pos)

            # If no edges, use baseline frustration
            if len(incident_edges) == 0:
                frustration[pos] = self._calculate_baseline_frustration(pos, sequence)
                continue

            # Collect interaction energies and types
            energies = []
            interaction_variety = set()

            for edge_id in incident_edges:
                weight = edge_weights[edge_id]
                interaction = interaction_types[edge_id]

                # Convert weight to energy-like value (higher weight = lower energy = more favorable)
                # For RNA: Base this on interaction type using established energy scales
                if interaction == 'canonical_wc':
                    energy = -2.0 * weight  # Strong favorable interaction
                elif interaction == 'wobble_pair':
                    energy = -1.5 * weight  # Moderate favorable
                elif interaction == 'base_stacking':
                    energy = -1.8 * weight  # Strong stacking
                elif interaction == 'backbone':
                    energy = -0.5 * weight  # Weak backbone
                elif interaction == 'non_canonical':
                    energy = -1.0 * weight  # Weak pairing
                else:
                    energy = -1.0 * weight  # Default

                energies.append(energy)
                interaction_variety.add(interaction)

            # Calculate frustration components

            # 1. Energy variance (conflicting interaction strengths)
            # High variance indicates conflicting interaction strengths
            if len(energies) > 1:
                energy_variance = np.var(energies)
            else:
                energy_variance = 0.1  # Single interaction still has potential frustration

            # 2. Interaction type conflict
            # Multiple interaction types suggest conflicting structural roles
            type_conflict = 0.0
            if len(interaction_variety) > 1:
                type_conflict = (len(interaction_variety) - 1) / 3.0  # Normalize to 0-1

            # 3. Local sequence context frustration
            # Sequence context can constrain or enable interactions
            seq_context = self._calculate_sequence_context_frustration(pos, sequence)

            # 4. Structural frustration potential
            # Based on potential alternative interactions
            struct_potential = self._calculate_structural_frustration(pos, sequence, graph, incident_edges)

            # Combine frustration components (weights can be adjusted)
            frustration[pos] = (
                    0.4 * energy_variance +
                    0.3 * type_conflict +
                    0.2 * seq_context +
                    0.1 * struct_potential
            )

            # Ensure frustration is bounded [0, 1]
            frustration[pos] = max(0.0, min(1.0, frustration[pos]))

        # Apply smoothing to avoid extreme outliers
        frustration = self._smooth_frustration(frustration)

        return frustration

    def _calculate_baseline_frustration(self, pos, sequence):
        """
        Calculate baseline frustration for isolated nucleotides.

        Used when a nucleotide has no interactions in the constraint graph,
        representing its inherent frustration based on sequence context alone.

        Args:
            pos (int): Position in sequence
            sequence (str): Full RNA sequence

        Returns:
            float: Baseline frustration score (0-1)
        """
        n = len(sequence)
        nt = sequence[pos].upper()

        factors = []

        # 1. Nucleotide type frustration
        # GC nucleotides "want" to pair more than AU (higher frustration if unpaired)
        if nt in ['G', 'C']:
            factors.append(0.6)  # High frustration potential when unpaired
        elif nt in ['A', 'U']:
            factors.append(0.3)  # Moderate frustration potential when unpaired

        # 2. Position-based frustration
        # Ends of sequences can naturally be unpaired with lower frustration
        if pos < 5 or pos > n - 5:
            factors.append(0.2)  # Ends can be unpaired with low frustration
        else:
            factors.append(0.5)  # Middle should pair, higher frustration if not

        # 3. Local sequence context
        context_window = sequence[max(0, pos - 2):min(n, pos + 3)]
        gc_context = sum(1 for c in context_window if c.upper() in ['G', 'C']) / len(context_window)
        factors.append(gc_context * 0.5)  # More GC neighbors = more frustration

        return np.mean(factors) if factors else 0.3

    def _calculate_sequence_context_frustration(self, pos, sequence):
        """
        Calculate frustration from local sequence context.

        Evaluates how the surrounding sequence contributes to frustration
        through nucleotide diversity and repetitive patterns.

        Args:
            pos (int): Position in sequence
            sequence (str): Full RNA sequence

        Returns:
            float: Context-based frustration score
        """
        n = len(sequence)

        # Check if nucleotide is in a repetitive region
        window_size = 5
        start = max(0, pos - window_size)
        end = min(n, pos + window_size + 1)

        context = sequence[start:end]

        # Homopolymeric regions have low frustration (can form stable structures)
        if len(set(context)) == 1:
            return 0.1

        # Mixed regions have higher frustration (competing interactions)
        nucleotide_counts = Counter(context)
        entropy = -sum((count / len(context)) * math.log(count / len(context)) for count in nucleotide_counts.values())
        max_entropy = math.log(min(4, len(set(context))))

        return entropy / max_entropy if max_entropy > 0 else 0.5

    def _calculate_structural_frustration(self, pos, sequence, graph, incident_edges):
        """
        Calculate structural frustration potential.

        Evaluates frustration arising from potential alternative interactions
        and satisfaction of current interactions.

        Args:
            pos (int): Position in sequence
            sequence (str): Full RNA sequence
            graph: igraph Graph object with current interactions
            incident_edges: List of edge indices incident to this position

        Returns:
            float: Structural frustration score (0-1)
        """
        n = len(sequence)
        nt = sequence[pos].upper()

        factors = []

        # Can this nucleotide potentially pair with others?
        # More potential pairs = more frustration (competing possibilities)
        potential_pairs = 0
        for i in range(n):
            if i == pos:
                continue

            other_nt = sequence[i].upper()
            distance = abs(i - pos)

            # Check if they could pair (based on base complementarity)
            if (nt == 'G' and other_nt == 'C') or (nt == 'C' and other_nt == 'G'):
                potential_pairs += 1
            elif (nt == 'A' and other_nt == 'U') or (nt == 'U' and other_nt == 'A'):
                potential_pairs += 1
            elif (nt == 'G' and other_nt == 'U') or (nt == 'U' and other_nt == 'G'):
                potential_pairs += 1

        # More potential pairs = more frustration (competing possibilities)
        pair_density = potential_pairs / (n - 1) if n > 1 else 0
        factors.append(min(pair_density * 2, 1.0))

        # Is this nucleotide already satisfied with its current interactions?
        if incident_edges:
            # Get average interaction strength
            avg_strength = np.mean([graph.es[e]["weight"] for e in incident_edges])
            # High average strength = low frustration (well-satisfied)
            factors.append(1.0 - avg_strength)
        else:
            factors.append(0.7)  # No interactions = high frustration

        return np.mean(factors) if factors else 0.5

    def _smooth_frustration(self, frustration):
        """
        Apply smoothing to frustration vector to reduce noise.

        Args:
            frustration (np.ndarray): Raw frustration scores

        Returns:
            np.ndarray: Smoothed frustration scores normalized to [0.01, 1]
        """
        if len(frustration) < 3:
            return frustration

        # Simple moving average for smoothing
        smoothed = np.copy(frustration)
        for i in range(1, len(frustration) - 1):
            smoothed[i] = np.mean(frustration[i - 1:i + 2])

        # Ensure non-zero minimum and proper scaling
        if smoothed.max() > 0:
            # Scale so max is 1, min is at least 0.01
            smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
            smoothed = np.maximum(smoothed, 0.01)

        return smoothed

    def _extract_rna_frustration_features(self, per_nucleotide_frustration, sequence, graph, edge_weights):
        """
        Extract RNA-specific frustration features for downstream analysis.

        Computes a comprehensive set of frustration-based features including
        regional statistics, nucleotide-type specific metrics, and structural
        indicators.

        Args:
            per_nucleotide_frustration (np.ndarray): Frustration scores for each position
            sequence (str): RNA sequence string
            graph: igraph Graph object
            edge_weights: List of edge weights

        Returns:
            dict: Dictionary of frustration features including:
                - Regional frustration means (5', middle, 3')
                - GC/AU region frustration statistics
                - Paired/unpaired frustration
                - Hotspot counts and density
                - Motif-associated frustration
                - Long-range interaction frustration
                - Frustration entropy
                - Fixed-length per-nucleotide vector
        """
        n = len(sequence)
        features = {}

        # 1. Region-specific frustration
        # Define sequence regions: 5' end, middle, 3' end
        five_prime_len = min(50, n)
        three_prime_len = min(30, n)

        five_prime_region = per_nucleotide_frustration[:five_prime_len]
        three_prime_region = per_nucleotide_frustration[-three_prime_len:] if three_prime_len > 0 else np.array([])
        middle_region = per_nucleotide_frustration[
                        five_prime_len:-three_prime_len] if five_prime_len + three_prime_len < n else per_nucleotide_frustration[
                                                                                                      five_prime_len:]

        features['Frustration_5Prime_Mean'] = np.mean(five_prime_region) if len(five_prime_region) > 0 else 0
        features['Frustration_3Prime_Mean'] = np.mean(three_prime_region) if len(three_prime_region) > 0 else 0
        features['Frustration_Middle_Mean'] = np.mean(middle_region) if len(middle_region) > 0 else 0

        # 2. Functional region frustration
        # GC-rich regions (often structured/stem-forming)
        gc_positions = [i for i, nt in enumerate(sequence) if nt in ['G', 'C']]
        if gc_positions:
            gc_frustration = per_nucleotide_frustration[gc_positions]
            features['Frustration_GC_Mean'] = np.mean(gc_frustration)
            features['Frustration_GC_Variance'] = np.var(gc_frustration) if len(gc_frustration) > 1 else 0
        else:
            features['Frustration_GC_Mean'] = 0
            features['Frustration_GC_Variance'] = 0

        # AU-rich regions (often unstructured/flexible, e.g., in regulatory elements)
        au_positions = [i for i, nt in enumerate(sequence) if nt in ['A', 'U']]
        if au_positions:
            au_frustration = per_nucleotide_frustration[au_positions]
            features['Frustration_AU_Mean'] = np.mean(au_frustration)
            features['Frustration_AU_Variance'] = np.var(au_frustration) if len(au_frustration) > 1 else 0
        else:
            features['Frustration_AU_Mean'] = 0
            features['Frustration_AU_Variance'] = 0

        # 3. Structural frustration indicators
        # Differences between regions highlight structural transitions
        features['Frustration_5vs3_Prime'] = features['Frustration_5Prime_Mean'] - features['Frustration_3Prime_Mean']
        features['Frustration_StructuredVsFlexible'] = features['Frustration_GC_Mean'] - features['Frustration_AU_Mean']

        # 4. Secondary structure frustration
        if graph.ecount() > 0 and 'interaction_type' in graph.es.attributes():
            # Find ALL types of pairing interactions
            pairing_keywords = ['pair', 'wc', 'canonical', 'wobble']
            pairing_edges = [
                i for i, itype in enumerate(graph.es["interaction_type"])
                if any(keyword in itype.lower() for keyword in pairing_keywords)
            ]

            if pairing_edges:
                paired_nodes = set()
                for edge_idx in pairing_edges:
                    edge = graph.es[edge_idx]
                    paired_nodes.add(edge.source)
                    paired_nodes.add(edge.target)

                # Calculate paired frustration
                if paired_nodes:
                    paired_frustration = per_nucleotide_frustration[list(paired_nodes)]
                    features['Frustration_Paired_Mean'] = np.mean(paired_frustration)
            else:
                features['Frustration_Paired_Mean'] = 0.0

        else:
            # No pairing edges found - treat all as unpaired
            features['Frustration_Paired_Mean'] = 0.0

        if graph.ecount() > 0:
            # Get base pairing interactions
            pairing_edges = [i for i, itype in enumerate(graph.es["interaction_type"])
                             if 'pair' in itype.lower() or 'wc' in itype.lower()]

            if pairing_edges:
                pairing_subgraph = graph.subgraph_edges(pairing_edges, delete_vertices=False)
                paired_nodes = set()
                for edge in pairing_subgraph.es:
                    paired_nodes.add(edge.source)
                    paired_nodes.add(edge.target)

                paired_frustration = per_nucleotide_frustration[list(paired_nodes)]
                unpaired_frustration = per_nucleotide_frustration[~np.isin(range(n), list(paired_nodes))]

                features['Frustration_Paired_Mean'] = np.mean(paired_frustration) if len(paired_frustration) > 0 else 0
            else:
                features['Frustration_Paired_Mean'] = 0
        else:
            features['Frustration_Paired_Mean'] = 0

        # 5. Hotspot analysis - identify regions of high frustration
        if len(per_nucleotide_frustration) > 0:
            frustration_mean = np.mean(per_nucleotide_frustration)
            frustration_std = np.std(per_nucleotide_frustration)

            # Use more robust hotspot detection
            if frustration_std > 0:
                # Standard deviation-based threshold
                hotspot_threshold = frustration_mean + frustration_std
            else:
                # If no variance, use percentile-based threshold
                hotspot_threshold = np.percentile(per_nucleotide_frustration, 75)

            # Count hotspots (positions above threshold)
            hotspot_indices = np.where(per_nucleotide_frustration > hotspot_threshold)[0]
            features['Frustration_HotspotCount'] = len(hotspot_indices)
            features['Frustration_HotspotDensity'] = len(hotspot_indices) / len(sequence)
        else:
            features['Frustration_HotspotCount'] = 0
            features['Frustration_HotspotDensity'] = 0

        # 6. Motif frustration correlation
        # High frustration in motif regions might indicate regulatory conflicts
        motif_regions = self._identify_motif_regions(sequence)
        if motif_regions:
            motif_frustration = []
            for start, end in motif_regions:
                motif_frustration.extend(per_nucleotide_frustration[start:end])

            if motif_frustration:
                features['Frustration_Motif_Mean'] = np.mean(motif_frustration)
                features['Frustration_Motif_Max'] = np.max(motif_frustration)
                features['Frustration_NonMotif_Mean'] = np.mean(
                    [f for i, f in enumerate(per_nucleotide_frustration)
                     if not any(start <= i < end for start, end in motif_regions)]
                )
            else:
                features['Frustration_Motif_Mean'] = 0
                features['Frustration_Motif_Max'] = 0
                features['Frustration_NonMotif_Mean'] = features['Frustration_Middle_Mean']
        else:
            features['Frustration_Motif_Mean'] = 0
            features['Frustration_Motif_Max'] = 0
            features['Frustration_NonMotif_Mean'] = features['Frustration_Middle_Mean']

        # 7. Long-range frustration
        if graph.ecount() > 0:
            long_range_edges = []
            for edge in graph.es:
                # Calculate linear sequence distance (positions)
                distance = abs(edge.source - edge.target)
                if distance > 20:  # Long-range interactions in sequence space
                    long_range_edges.append(edge.index)

            if long_range_edges:
                long_range_nodes = set()
                for edge_idx in long_range_edges:
                    edge = graph.es[edge_idx]
                    long_range_nodes.add(edge.source)
                    long_range_nodes.add(edge.target)

                long_range_frustration = per_nucleotide_frustration[list(long_range_nodes)]
                features['Frustration_LongRange_Mean'] = np.mean(long_range_frustration) if len(
                    long_range_frustration) > 0 else 0
            else:
                features['Frustration_LongRange_Mean'] = 0
        else:
            features['Frustration_LongRange_Mean'] = 0

        # 8. Per-nucleotide vector (for neural network input)
        # Fixed-length representation for sequences of varying lengths
        fixed_length = 100
        if n >= fixed_length:
            # Take representative sample by stepping through sequence
            step = n // fixed_length
            features['Frustration_PerNucleotide_Vector'] = per_nucleotide_frustration[::step][:fixed_length]
        else:
            # Pad shorter sequences with zeros
            padded = np.pad(per_nucleotide_frustration, (0, fixed_length - n), 'constant')
            features['Frustration_PerNucleotide_Vector'] = padded

        # 9. Frustration entropy (measure of frustration localization)
        # High entropy = frustration spread evenly, low entropy = localized hotspots
        frustration_entropy = stats.entropy(per_nucleotide_frustration + 1e-10)
        features['Frustration_Entropy'] = frustration_entropy

        return features

    def _identify_motif_regions(self, sequence: str) -> List[Tuple[int, int]]:
        """
        Identify potential functional motif regions in RNA.

        Detects common RNA regulatory and structural motifs using pattern matching.

        Args:
            sequence (str): RNA sequence string

        Returns:
            List[Tuple[int, int]]: List of (start, end) positions for identified motifs
        """
        motifs = []
        seq = sequence.upper()

        # Common RNA motifs and their patterns
        motif_patterns = {
            'ARE': r'[AU]{5,}',  # AU-rich elements (destabilizing/regulatory)
            'G_rich': r'[G]{4,}',  # G-quadruplex potential
            'C_rich': r'[C]{4,}',  # C-rich regions
            'purine_rich': r'[AG]{4,}',  # Purine-rich (often in loops)
            'pyrimidine_rich': r'[CU]{4,}',  # Pyrimidine-rich (often in stems)
        }

        for motif_name, pattern in motif_patterns.items():
            for match in re.finditer(pattern, seq):
                motifs.append((match.start(), match.end()))

        # Remove overlapping motifs by merging
        motifs.sort(key=lambda x: x[0])
        merged = []
        for start, end in motifs:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        return merged

    def _get_default_rna_frustration_features(self):
        """
        Return default features for very short RNA sequences or empty graphs.

        Provides zero-initialized features with proper structure when
        frustration analysis cannot be meaningfully performed.

        Returns:
            dict: Default frustration features with zeros and zero vector
        """
        return {
            'Frustration_5Prime_Mean': 0,
            'Frustration_3Prime_Mean': 0,
            'Frustration_Middle_Mean': 0,
            'Frustration_GC_Mean': 0,
            'Frustration_GC_Variance': 0,
            'Frustration_AU_Mean': 0,
            'Frustration_AU_Variance': 0,
            'Frustration_5vs3_Prime': 0,
            'Frustration_StructuredVsFlexible': 0,
            'Frustration_Paired_Mean': 0,
            'Frustration_HotspotCount': 0,
            'Frustration_HotspotDensity': 0,
            'Frustration_Motif_Mean': 0,
            'Frustration_Motif_Max': 0,
            'Frustration_NonMotif_Mean': 0,
            'Frustration_LongRange_Mean': 0,
            'Frustration_Entropy': 0,
            'Frustration_PerNucleotide_Vector': np.zeros(100)
        }
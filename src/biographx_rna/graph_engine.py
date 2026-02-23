"""
Author: Abubakar Saeed
Date: February 23, 2026
Description: RNA graph construction and analysis engine that builds graph representations
of RNA sequences based on nucleotide interactions. Implements RNA-specific interaction
rules, hybrid interaction detection, and comprehensive feature extraction for machine
learning applications.

Note on distance calculations: All distance-based constraints (e.g., max_distance,
distance thresholds, loop size limits) refer to linear sequence distances measured
in nucleotide positions, not 3D spatial distances in Angstroms. This is consistent
with RNA secondary structure analysis where interactions depend on sequence proximity.
"""
import igraph as ig
import numpy as np
import warnings
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from src.biographx_rna.biophysics import RNAPhysicsStrategy
from src.biographx_rna.profiler import RNAMotifProfiler

class RNAGraphEngine:
    """
    Handles RNA-specific graph construction, feature extraction, and analysis.
    Builds graphs based on RNA nucleotide interactions including canonical pairing,
    wobble pairs, non-canonical interactions, base stacking, and backbone connections.

    The graph representation captures:
    - Nodes: Individual nucleotides with position and type attributes
    - Edges: Interactions between nucleotides with type, weight, and energy attributes
    - Hybrid interactions: Combined interaction types with enhanced stability

    Attributes:
        rna_physics (RNAPhysicsStrategy): Reference to RNA physics rules for interactions
        motif_profiler (RNAMotifProfiler): Reference to motif detection for localization
    """

    def __init__(self, rna_physics_strategy: RNAPhysicsStrategy, rna_motif_profiler: RNAMotifProfiler):
        """
        Initialize RNA graph engine with physics strategy and motif profiler.

        Args:
            rna_physics_strategy: RNAPhysicsStrategy instance containing interaction rules
            rna_motif_profiler: RNAMotifProfiler instance for motif-based analysis
        """
        self.rna_physics = rna_physics_strategy
        self.motif_profiler = rna_motif_profiler

    def build_rna_graph(self, sequence: str) -> Tuple[ig.Graph, Dict[str, float]]:
        """
        Build RNA graph with nucleotide interactions and hybrid tracking.

        Constructs a graph where nodes are nucleotides and edges represent
        various RNA-specific interactions. Implements geometric constraints
        on base pairing (minimum loop size) to ensure biologically realistic
        secondary structures.

        Args:
            sequence (str): RNA sequence string (A, U, G, C)

        Returns:
            Tuple[ig.Graph, Dict[str, float]]:
                - Graph object with node and edge attributes
                - Dictionary of hybrid interaction scores

        Note on distance constraints:
            - Pairing interactions (canonical, wobble, non-canonical) require
              linear sequence distance >= 4 nucleotides to allow minimum loop size
            - Stacking interactions can occur between adjacent nucleotides
            - Backbone connections always exist between consecutive nucleotides
            - Long-range interactions (>5 nt) have distance-based decay
        """
        n = len(sequence)
        seq = sequence.upper()

        graph = ig.Graph(n, directed=False)
        graph.vs["nucleotide"] = list(seq)
        graph.vs["position"] = list(range(n))

        edges = []
        edge_attributes = {
            "interaction_type": [],
            "weight": [],
            "interaction_set": [],
            "is_hybrid": [],
            "energy": []  # Interaction energy estimate (kcal/mol approx)
        }

        # Track interactions for hybrid analysis
        residue_interactions = defaultdict(set)
        hybrid_edge_counts = {hybrid_type: 0 for hybrid_type in self.rna_physics.hybrid_interactions.keys()}

        # ---------------------------------------------------------
        # 1. RNA-INTERACTIONS
        # ---------------------------------------------------------
        for i in range(n):
            for j in range(i + 1, n):
                distance = j - i  # Linear sequence distance in nucleotides
                nt1, nt2 = seq[i], seq[j]

                found_interactions = set()

                # Check each interaction type from physics strategy
                for interaction_type, rules in self.rna_physics.interaction_rules.items():
                    # Apply distance threshold (linear sequence positions)
                    if distance > rules['max_distance']:
                        continue

                    # 🔴 GEOMETRIC CONSTRAINT VALIDATION 🔴
                    # Enforce minimum loop size (hairpin limit) for base pairing.
                    # Adjacent nucleotides (distance < 4) cannot form base pairs due to
                    # steric constraints in RNA secondary structure. They can only
                    # interact via stacking or backbone connections.
                    # Without this constraint, neighbor pairings would dominate over
                    # biologically realistic interactions.
                    if interaction_type in ['canonical_wc', 'wobble_pair', 'non_canonical']:
                        if distance < 4:
                            continue

                    if self.rna_physics.check_interaction(nt1, nt2, interaction_type):
                        found_interactions.add(interaction_type)

                if found_interactions:
                    # Track for hybrid detection
                    residue_interactions[i].update(found_interactions)
                    residue_interactions[j].update(found_interactions)

                    # Select dominant interaction based on strength
                    dominant = max(found_interactions,
                                   key=lambda it: self.rna_physics.interaction_rules[it]['strength'])

                    weight = self.rna_physics.interaction_rules[dominant]['strength']

                    # Distance decay for long-range interactions
                    # Longer-range interactions have reduced effective strength
                    if distance > 5:
                        decay_factor = 1 - (
                                distance / self.rna_physics.interaction_rules[dominant]['max_distance']) * 0.5
                        weight *= max(0.1, decay_factor)

                    # Estimate interaction energy based on type (kcal/mol)
                    energy = 0.0
                    if dominant == 'canonical_wc':
                        energy = -2.0  # Watson-Crick: strongest stabilization
                    elif dominant == 'wobble_pair':
                        energy = -1.0  # Wobble: moderate stabilization
                    elif dominant == 'base_stacking':
                        energy = -1.5  # Stacking: strong stabilization

                    # Hybrid Detection for RNA
                    # A hybrid interaction occurs when multiple interaction types exist
                    # between the same nucleotides, indicating enhanced stability
                    is_hybrid = 0
                    if len(found_interactions) >= 2:
                        for hybrid_type, rules in self.rna_physics.hybrid_interactions.items():
                            if rules['primary'] in found_interactions and rules['secondary'] in found_interactions:
                                is_hybrid = 1
                                hybrid_edge_counts[hybrid_type] += 1
                                weight *= rules['weight']  # Enhanced weight for hybrid
                                energy *= 1.2  # Hybrid interactions are more stable
                                break

                    edges.append((i, j))
                    edge_attributes["interaction_type"].append(dominant)
                    edge_attributes["weight"].append(min(weight, 1.0))
                    edge_attributes["interaction_set"].append(frozenset(found_interactions))
                    edge_attributes["is_hybrid"].append(is_hybrid)
                    edge_attributes["energy"].append(energy)

        # Add edges to graph
        if edges:
            graph.add_edges(edges)
            for attr_name, attr_values in edge_attributes.items():
                graph.es[attr_name] = attr_values

        # ---------------------------------------------------------
        # 2. PHOSPHATE BACKBONE CONNECTIONS (Fallback/Reinforcement)
        # ---------------------------------------------------------
        # Ensure backbone connections exist between consecutive nucleotides
        # even if no other interactions were detected. This maintains the
        # linear chain structure of RNA.
        backbone_edges = []
        for i in range(n - 1):
            if not graph.are_adjacent(i, i + 1):
                backbone_edges.append((i, i + 1))

        if backbone_edges:
            start_idx = graph.ecount()
            graph.add_edges(backbone_edges)

            new_edge_slice = graph.es[start_idx:]
            new_edge_slice["weight"] = [0.3] * len(backbone_edges)  # Lower weight for backbone
            new_edge_slice["interaction_type"] = ["backbone"] * len(backbone_edges)
            new_edge_slice["is_hybrid"] = [0] * len(backbone_edges)
            new_edge_slice["interaction_set"] = [frozenset(['backbone'])] * len(backbone_edges)
            new_edge_slice["energy"] = [-0.2] * len(backbone_edges)  # Weak stabilization

        # ---------------------------------------------------------
        # 3. CALCULATE HYBRID SCORES
        # ---------------------------------------------------------
        hybrid_scores = self._calculate_rna_hybrid_scores(graph, residue_interactions, hybrid_edge_counts, seq)

        for hybrid_type, score in hybrid_scores.items():
            graph[hybrid_type] = score

        return graph, hybrid_scores

    def _calculate_rna_hybrid_scores(self, graph: ig.Graph, residue_interactions: Dict,
                                     hybrid_edge_counts: Dict[str, int], sequence: str) -> Dict[str, float]:
        """
        Calculate comprehensive hybrid interaction scores for RNA.

        Combines edge-based evidence (actual hybrid edges) with residue-based
        evidence (nucleotides capable of hybrid interactions) to produce
        robust hybrid interaction scores.

        Args:
            graph: Constructed RNA graph
            residue_interactions: Dictionary mapping positions to interaction types
            hybrid_edge_counts: Counts of edges with each hybrid type
            sequence: RNA sequence

        Returns:
            Dict[str, float]: Normalized scores for each hybrid interaction type
        """
        hybrid_scores = {hybrid_type: 0.0 for hybrid_type in self.rna_physics.hybrid_interactions.keys()}

        if graph.ecount() == 0:
            return hybrid_scores

        # 1. Edge-based hybrid scores (direct evidence)
        total_edges = graph.ecount()
        for hybrid_type in self.rna_physics.hybrid_interactions.keys():
            hybrid_scores[hybrid_type] = hybrid_edge_counts[hybrid_type] / total_edges if total_edges > 0 else 0

        # 2. Residue-based hybrid scores (potential evidence)
        residue_hybrid_scores = {hybrid_type: [] for hybrid_type in self.rna_physics.hybrid_interactions.keys()}

        for residue, interactions in residue_interactions.items():
            if len(interactions) >= 2:
                for hybrid_type, rules in self.rna_physics.hybrid_interactions.items():
                    if rules['primary'] in interactions and rules['secondary'] in interactions:
                        score = 1.0
                        residue_hybrid_scores[hybrid_type].append(score)

        # Combine edge and residue scores with weighting
        # Edge evidence (70%) is stronger than residue potential (30%)
        for hybrid_type in self.rna_physics.hybrid_interactions.keys():
            if residue_hybrid_scores[hybrid_type]:
                residue_mean = np.mean(residue_hybrid_scores[hybrid_type])
                hybrid_scores[hybrid_type] = 0.7 * hybrid_scores[hybrid_type] + 0.3 * residue_mean

        return hybrid_scores

    def extract_basic_rna_graph_features(self, graph: ig.Graph) -> np.ndarray:
        """
        Extract basic graph features for RNA analysis.

        Comprehensive feature extraction covering:
        - Basic topology (nodes, edges, density)
        - Degree statistics (mean, std, max, percentiles)
        - Weighted degree (using interaction strengths)
        - Interaction type distributions
        - Energy-based features (stabilization estimates)
        - Centrality measures (betweenness, closeness, eigenvector)
        - Community structure
        - RNA localization patterns
        - Path-based metrics

        Args:
            graph: Constructed RNA graph

        Returns:
            np.ndarray: 65-dimensional feature vector of dtype float32
        """
        features = []

        n = graph.vcount()
        e = graph.ecount()

        # 1. Basic topology
        features.extend([n, e, e / max(1, (n * (n - 1) / 2)) if n > 1 else 0])

        # 2. Degree statistics
        if e > 0:
            degrees = graph.degree()
            features.extend([
                np.mean(degrees), np.std(degrees), np.max(degrees),
                np.percentile(degrees, 25), np.median(degrees), np.percentile(degrees, 75)
            ])
        else:
            features.extend([0.0] * 6)

        # 3. Weighted degree (using interaction energies)
        if e > 0 and 'weight' in graph.es.attributes():
            weights = graph.es["weight"]
            weighted_degrees = graph.strength(weights=weights)
            features.extend([
                np.mean(weighted_degrees), np.std(weighted_degrees), np.max(weighted_degrees)
            ])
        else:
            features.extend([0.0] * 3)

        # 4. RNA interaction type distribution
        if e > 0 and 'interaction_type' in graph.es.attributes():
            interaction_counts = Counter(graph.es["interaction_type"])
            total_interactions = sum(interaction_counts.values())

            rna_interaction_types = [
                'canonical_wc', 'wobble_pair', 'non_canonical',
                'base_stacking', 'backbone'  # Five core interaction types
            ]

            for itype in rna_interaction_types:
                features.append(interaction_counts.get(itype, 0) / total_interactions if total_interactions > 0 else 0)
        else:
            features.extend([0.0] * 5)

        # 5. Energy-based features
        if e > 0 and 'energy' in graph.es.attributes():
            energies = graph.es["energy"]
            features.extend([
                np.mean(energies), np.std(energies), np.min(energies),  # Most negative = strongest
                np.sum(energies),  # Total estimated stabilization energy
                len([e for e in energies if e < -1.0]) / e  # Strong interactions ratio
            ])
        else:
            features.extend([0.0] * 5)

        # 6. Centrality features
        centrality_features = self._extract_rna_centrality(graph)
        features.extend(centrality_features)

        # 7. Community features
        community_features = self._extract_rna_community_features(graph)
        features.extend(community_features)

        # 8. RNA-specific pattern features
        sequence = ''.join(graph.vs["nucleotide"]) if "nucleotide" in graph.vs.attributes() else ""
        pattern_features = self._extract_rna_localization_patterns(graph, sequence)
        features.extend(pattern_features)

        # 9. Path features
        path_features = self._extract_rna_path_features(graph)
        features.extend(path_features)

        # 10. Additional RNA-specific metrics
        additional_features = self._extract_rna_additional_metrics(graph)
        features.extend(additional_features)

        # Ensure fixed feature dimension (65 features)
        target_len = 65
        if len(features) < target_len:
            features.extend([0.0] * (target_len - len(features)))

        return np.array(features[:target_len], dtype=np.float32)

    def _extract_rna_centrality(self, graph: ig.Graph) -> List[float]:
        """
        Extract centrality features for RNA graph with performance cutoffs.

        Uses restricted path lengths (cutoff=10) to maintain computational
        efficiency for long RNA sequences while capturing local connectivity
        patterns relevant to RNA structure.

        Args:
            graph: RNA graph

        Returns:
            List[float]: 9 centrality features
        """
        features = []
        n = graph.vcount()
        e = graph.ecount()

        if e == 0 or n < 3:
            return [0.0] * 9

        try:
            weights = graph.es["weight"] if "weight" in graph.es.attributes() else None

            # 1. Betweenness Centrality (Restricted to 10 hops)
            # Without 'cutoff=10', this is O(N^2) and will freeze on long RNAs
            # The cutoff limits computation to local neighborhoods, which is
            # biologically relevant for RNA structure.
            try:
                betweenness = graph.betweenness(weights=weights, cutoff=10)
                features.extend([
                    np.mean(betweenness), np.std(betweenness),
                    np.max(betweenness),
                ])
            except:
                features.extend([0.0, 0.0, 0.0])

            # 2. Closeness Centrality (Restricted to 10 hops)
            # Similarly limited to local neighborhoods for efficiency
            try:
                closeness = graph.closeness(weights=weights, cutoff=10)
                features.extend([
                    np.mean(closeness), np.std(closeness),
                    np.max(closeness), np.min(closeness)
                ])
            except:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # Eigenvector centrality (global measure, but computationally feasible)
            try:
                eigenvector = graph.eigenvector_centrality(weights=weights)
                features.extend([np.mean(eigenvector), np.std(eigenvector)])
            except:
                features.extend([0.0, 0.0])

        except:
            return [0.0] * 9

        return features

    def _extract_rna_community_features(self, graph: ig.Graph) -> List[float]:
        """
        Extract community/domain features for RNA.

        Identifies structural domains within the RNA using community detection,
        which correspond to independent folding units or functional modules.

        Args:
            graph: RNA graph

        Returns:
            List[float]: 6 community-related features
        """
        features = []
        n = graph.vcount()
        e = graph.ecount()

        if e < 10 or n < 20:
            return [0.0] * 6

        try:
            weights = graph.es["weight"] if 'weight' in graph.es.attributes() else None
            communities = graph.community_multilevel(weights=weights)

            community_sizes = [len(c) for c in communities]
            features.extend([
                len(communities),  # Number of structural domains
                graph.modularity(communities, weights=weights),  # Domain separation quality
                np.mean(community_sizes) if community_sizes else 0,
                np.std(community_sizes) if len(community_sizes) > 1 else 0,
                np.max(community_sizes) if community_sizes else 0,  # Largest domain size
            ])

            # Intra-community edge ratio (edges within vs between domains)
            if len(communities) > 1:
                community_map = {}
                for comm_id, community in enumerate(communities):
                    for node in community:
                        community_map[node] = comm_id

                intra_edges = 0
                for edge in graph.es:
                    if community_map[edge.source] == community_map[edge.target]:
                        intra_edges += 1

                features.append(intra_edges / e if e > 0 else 0)  # Domain cohesion
            else:
                features.append(1.0)  # Single domain: all edges are intra-domain

        except:
            features.extend([0.0] * 6)

        return features

    def _extract_rna_localization_patterns(self, graph: ig.Graph, sequence: str) -> np.ndarray:
        """
        Extract RNA-specific localization patterns from graph.

        Analyzes graph properties in functionally important regions:
        - 5' region (translation start, regulatory elements)
        - 3' region (localization signals, poly-A tail)
        - GC-rich clusters (structured domains)
        - AU-rich clusters (flexible/unstructured regions)

        Args:
            graph: RNA graph
            sequence: RNA sequence

        Returns:
            np.ndarray: 12-dimensional array of region-specific features
        """
        features = []

        n = len(sequence)
        if n < 10:
            return np.zeros(12, dtype=np.float32)

        # 1. 5' region features (first 30 nucleotides)
        five_prime_features = self._analyze_five_prime_graph(graph, sequence)
        features.extend(five_prime_features)

        # 2. 3' region features (last 30 nucleotides)
        three_prime_features = self._analyze_three_prime_graph(graph, sequence)
        features.extend(three_prime_features)

        # 3. GC-rich cluster analysis
        gc_features = self._analyze_gc_clusters(graph, sequence)
        features.extend(gc_features)

        # 4. AU-rich cluster analysis
        au_features = self._analyze_au_clusters(graph, sequence)
        features.extend(au_features)

        return np.array(features, dtype=np.float32)

    def _analyze_five_prime_graph(self, graph: ig.Graph, sequence: str) -> List[float]:
        """
        Analyze 5' region graph properties.

        The 5' region (first ~30 nucleotides) often contains translation
        initiation signals and regulatory elements.

        Args:
            graph: RNA graph
            sequence: RNA sequence

        Returns:
            List[float]: 3 features: density, GC content, strong interaction ratio
        """
        features = []

        five_prime_len = min(30, len(sequence))
        if five_prime_len < 5:
            return [0.0] * 3

        try:
            five_prime_nodes = list(range(five_prime_len))
            five_prime_subgraph = graph.induced_subgraph(five_prime_nodes)

            if five_prime_subgraph.ecount() > 0:
                # 1. Connectivity density in 5' region
                max_possible = (five_prime_len * (five_prime_len - 1)) / 2
                actual_edges = five_prime_subgraph.ecount()
                features.append(actual_edges / max_possible if max_possible > 0 else 0)

                # 2. Base pairing potential in 5' region (GC content)
                gc_count = sum(1 for i in five_prime_nodes if sequence[i] in ['G', 'C'])
                features.append(gc_count / five_prime_len)

                # 3. Strong interactions in 5' region
                if 'weight' in five_prime_subgraph.es.attributes():
                    strong_edges = len([w for w in five_prime_subgraph.es["weight"] if w > 0.7])
                    features.append(
                        strong_edges / five_prime_subgraph.ecount() if five_prime_subgraph.ecount() > 0 else 0)
                else:
                    features.append(0.0)
            else:
                features.extend([0.0] * 3)

        except:
            features.extend([0.0] * 3)

        return features

    def _analyze_three_prime_graph(self, graph: ig.Graph, sequence: str) -> List[float]:
        """
        Analyze 3' region graph properties.

        The 3' region (last ~30 nucleotides) often contains localization
        signals, polyadenylation signals, and stability elements.

        Args:
            graph: RNA graph
            sequence: RNA sequence

        Returns:
            List[float]: 3 features: density, AU content, terminal connectivity
        """
        features = []

        three_prime_len = min(30, len(sequence))
        if three_prime_len < 5:
            return [0.0] * 3

        try:
            three_prime_start = len(sequence) - three_prime_len
            three_prime_nodes = list(range(three_prime_start, len(sequence)))
            three_prime_subgraph = graph.induced_subgraph(three_prime_nodes)

            if three_prime_subgraph.ecount() > 0:
                # 1. Connectivity density in 3' region
                max_possible = (three_prime_len * (three_prime_len - 1)) / 2
                actual_edges = three_prime_subgraph.ecount()
                features.append(actual_edges / max_possible if max_possible > 0 else 0)

                # 2. AU content in 3' region (often indicates flexibility)
                au_count = sum(1 for i in three_prime_nodes if sequence[i] in ['A', 'U'])
                features.append(au_count / three_prime_len)

                # 3. Terminal nucleotide connectivity
                last_node = len(sequence) - 1
                if last_node < graph.vcount():
                    terminal_degree = graph.degree(last_node)
                    features.append(terminal_degree / 10.0)  # Normalized
                else:
                    features.append(0.0)
            else:
                features.extend([0.0] * 3)

        except:
            features.extend([0.0] * 3)

        return features

    def _analyze_gc_clusters(self, graph: ig.Graph, sequence: str) -> List[float]:
        """
        Analyze GC-rich clusters in RNA graph.

        GC-rich regions typically form stable structured domains like stems
        and helices. This analysis identifies clusters of GC nucleotides
        and their connectivity patterns.

        Args:
            graph: RNA graph
            sequence: RNA sequence

        Returns:
            List[float]: 3 features: connectivity, cluster count, largest cluster size
        """
        features = []

        # Find GC-rich positions
        gc_positions = [i for i, nt in enumerate(sequence) if nt in ['G', 'C']]

        if len(gc_positions) < 3:
            return [0.0] * 3

        # Create GC subgraph
        gc_subgraph = graph.induced_subgraph(gc_positions)

        if gc_subgraph.ecount() == 0:
            return [0.0] * 3

        # 1. GC cluster connectivity (density within GC-rich region)
        max_gc_edges = (len(gc_positions) * (len(gc_positions) - 1)) / 2
        actual_gc_edges = gc_subgraph.ecount()
        features.append(actual_gc_edges / max_gc_edges if max_gc_edges > 0 else 0)

        # 2. GC cluster components
        gc_components = gc_subgraph.connected_components()
        gc_component_sizes = [len(c) for c in gc_components]

        if gc_component_sizes:
            features.append(len(gc_components))  # Number of GC clusters
            features.append(np.max(gc_component_sizes) if gc_component_sizes else 0)  # Largest GC cluster
        else:
            features.extend([0.0, 0.0])

        return features

    def _analyze_au_clusters(self, graph: ig.Graph, sequence: str) -> List[float]:
        """
        Analyze AU-rich clusters in RNA graph.

        AU-rich regions are often flexible, unstructured, or involved in
        regulatory interactions. This analysis identifies clusters of AU
        nucleotides and their connectivity patterns.

        Args:
            graph: RNA graph
            sequence: RNA sequence

        Returns:
            List[float]: 3 features: connectivity, cluster count, largest cluster size
        """
        features = []

        # Find AU-rich positions
        au_positions = [i for i, nt in enumerate(sequence) if nt in ['A', 'U']]

        if len(au_positions) < 3:
            return [0.0] * 3

        # Create AU subgraph
        au_subgraph = graph.induced_subgraph(au_positions)

        if au_subgraph.ecount() == 0:
            return [0.0] * 3

        # 1. AU cluster connectivity (density within AU-rich region)
        max_au_edges = (len(au_positions) * (len(au_positions) - 1)) / 2
        actual_au_edges = au_subgraph.ecount()
        features.append(actual_au_edges / max_au_edges if max_au_edges > 0 else 0)

        # 2. AU cluster components
        au_components = au_subgraph.connected_components()
        au_component_sizes = [len(c) for c in au_components]

        if au_component_sizes:
            features.append(len(au_components))  # Number of AU clusters
            features.append(np.max(au_component_sizes) if au_component_sizes else 0)  # Largest AU cluster
        else:
            features.extend([0.0, 0.0])

        return features

    def _extract_rna_path_features(self, graph: ig.Graph) -> List[float]:
        """
        Extract path-based features for RNA graph.

        Calculates metrics related to communication paths within the graph,
        including average path length, diameter, and efficiency measures.

        Args:
            graph: RNA graph

        Returns:
            List[float]: 6 path-based features
        """
        features = []
        n = graph.vcount()
        e = graph.ecount()

        if n < 3 or e == 0:
            return [0.0] * 6

        try:
            weights = graph.es["weight"] if 'weight' in graph.es.attributes() else None

            # Use Largest Connected Component for path-based metrics
            if not graph.is_connected():
                clusters = graph.connected_components()
                subgraph = clusters.giant()
                sub_weights = subgraph.es["weight"] if 'weight' in subgraph.es.attributes() else None
            else:
                subgraph = graph
                sub_weights = weights

            # Calculate standard metrics
            avg_path = subgraph.average_path_length(weights=sub_weights, directed=False)
            diameter = subgraph.diameter(weights=sub_weights, directed=False)
            global_efficiency = 1.0 / avg_path if avg_path > 0 else 0.0

            features.extend([avg_path, diameter, global_efficiency])

            # Local efficiency sampling
            # Measures how well information is exchanged in neighborhoods
            sample_size = min(20, n)
            if sample_size > 2:
                sample_nodes = random.sample(range(n), sample_size)
                local_effs = []

                for v in sample_nodes:
                    neighbors = graph.neighbors(v)
                    if len(neighbors) < 2:
                        local_effs.append(0.0)
                        continue

                    neigh_graph = graph.induced_subgraph(neighbors)
                    if neigh_graph.ecount() == 0:
                        local_effs.append(0.0)
                        continue

                    try:
                        dists = neigh_graph.distances(weights=neigh_graph.es["weight"] if weights else None)
                        inv_dist_sum = 0.0
                        k = len(neighbors)

                        for r in range(k):
                            for c in range(r + 1, k):
                                d = dists[r][c]
                                if d > 0 and not np.isinf(d):
                                    inv_dist_sum += (1.0 / d)

                        if k > 1:
                            eff = (2.0 * inv_dist_sum) / (k * (k - 1))
                            local_effs.append(eff)
                        else:
                            local_effs.append(0.0)
                    except:
                        local_effs.append(0.0)

                avg_local_efficiency = np.mean(local_effs) if local_effs else 0.0
                features.append(avg_local_efficiency)
            else:
                features.append(0.0)

            # Path length statistics from sampling
            try:
                n_samples = min(30, n)
                sample_indices = random.sample(range(n), n_samples)
                sampled_paths = graph.distances(source=sample_indices, weights=weights)

                all_sampled_distances = []
                for row in sampled_paths:
                    for d in row:
                        if d > 0 and not np.isinf(d):
                            all_sampled_distances.append(d)

                if all_sampled_distances:
                    features.extend([
                        np.std(all_sampled_distances),
                        np.max(all_sampled_distances)
                    ])
                else:
                    features.extend([0.0, 0.0])
            except:
                features.extend([0.0, 0.0])

        except:
            features.extend([0.0] * 6)

        return features[:6]

    def _extract_rna_additional_metrics(self, graph: ig.Graph) -> List[float]:
        """
        Extract additional RNA-specific graph metrics.

        Includes clustering coefficient, assortativity, strong interaction
        ratios, and advanced energy statistics.

        Args:
            graph: RNA graph

        Returns:
            List[float]: 10 additional metrics
        """
        features = []
        n = graph.vcount()
        e = graph.ecount()

        if e == 0 or n < 3:
            return [0.0] * 10

        try:
            # Clustering coefficient (local connectivity)
            weights = graph.es["weight"] if 'weight' in graph.es.attributes() else None
            clustering = graph.transitivity_avglocal_undirected(weights=weights)
            features.append(clustering)

            # Assortativity (degree correlation)
            try:
                assortativity = graph.assortativity_degree()
                features.append(assortativity)
            except:
                features.append(0.0)

            # Strong interaction ratio
            if 'weight' in graph.es.attributes():
                strong_edges = len([w for w in graph.es["weight"] if w > 0.7])
                features.append(strong_edges / e)
            else:
                features.append(0.0)

            # Energy statistics
            if 'energy' in graph.es.attributes():
                energies = graph.es["energy"]
                features.extend([
                    np.mean(energies),
                    np.min(energies),  # Most negative = most stable
                    np.sum(energies) / n  # Average stabilization per nucleotide
                ])
            else:
                features.extend([0.0, 0.0, 0.0])

            # Hybrid edge ratio
            if 'is_hybrid' in graph.es.attributes():
                hybrid_edges = len([h for h in graph.es["is_hybrid"] if h == 1])
                features.append(hybrid_edges / e if e > 0 else 0)
            else:
                features.append(0.0)

            # Edge weight statistics
            if 'weight' in graph.es.attributes():
                weights = graph.es["weight"]
                features.extend([
                    np.mean(weights),
                    np.std(weights),
                    np.max(weights)
                ])
            else:
                features.extend([0.0, 0.0, 0.0])

        except:
            features.extend([0.0] * 10)

        return features[:10]

    def extract_rna_hybrid_features(self, graph: ig.Graph, hybrid_scores: Dict[str, float],
                                    sequence: str) -> np.ndarray:
        """
        Extract RNA-specific hybrid interaction features.

        Focuses on features related to hybrid interactions (multiple interaction types
        between same nucleotides), which indicate enhanced structural stability and
        potential functional significance.

        Args:
            graph: RNA graph
            hybrid_scores: Dictionary of hybrid interaction scores
            sequence: RNA sequence

        Returns:
            np.ndarray: 15-dimensional hybrid feature vector
        """
        features = []

        if graph.ecount() == 0:
            return np.zeros(15, dtype=np.float32)

        # 1. Basic hybrid scores (5 features)
        for hybrid_type in self.rna_physics.hybrid_interactions.keys():
            features.append(hybrid_scores.get(hybrid_type, 0.0))

        # 2. Regional hybrid density (4 features)
        seq = sequence.upper()
        regions = {
            'five_prime': list(range(0, min(30, len(seq)))),
            'three_prime': list(range(max(0, len(seq) - 30), len(seq))),
            'gc_regions': [i for i, nt in enumerate(seq) if nt in ['G', 'C']],
            'au_regions': [i for i, nt in enumerate(seq) if nt in ['A', 'U']]
        }

        for region_name, region_nodes in regions.items():
            if len(region_nodes) < 3:
                features.append(0.0)
                continue

            # Count hybrid edges in region
            hybrid_edges = 0
            total_region_edges = 0

            for i in range(len(region_nodes)):
                for j in range(i + 1, len(region_nodes)):
                    node_i = region_nodes[i]
                    node_j = region_nodes[j]

                    if graph.are_adjacent(node_i, node_j):
                        total_region_edges += 1
                        edge_id = graph.get_eid(node_i, node_j)
                        if graph.es[edge_id]["is_hybrid"] == 1:
                            hybrid_edges += 1

            if total_region_edges > 0:
                features.append(hybrid_edges / total_region_edges)
            else:
                features.append(0.0)

        # 3. Hybrid network properties (3 features)
        try:
            hybrid_edge_indices = [i for i, edge in enumerate(graph.es) if edge["is_hybrid"] == 1]

            if hybrid_edge_indices:
                hybrid_subgraph = graph.subgraph_edges(hybrid_edge_indices, delete_vertices=False)

                features.extend([
                    len(hybrid_edge_indices) / graph.ecount(),  # Hybrid edge ratio
                    hybrid_subgraph.density(),  # Connectivity among hybrid edges
                    len(hybrid_subgraph.connected_components())  # Hybrid cluster count
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        except:
            features.extend([0.0, 0.0, 0.0])

        # 4. Hybrid energy features (3 features)
        if 'energy' in graph.es.attributes() and 'is_hybrid' in graph.es.attributes():
            hybrid_energies = [graph.es[i]["energy"] for i in range(graph.ecount())
                               if graph.es[i]["is_hybrid"] == 1]

            if hybrid_energies:
                features.extend([
                    np.mean(hybrid_energies),  # Average hybrid stability
                    np.min(hybrid_energies),  # Most stable hybrid interaction
                    len([e for e in hybrid_energies if e < -1.0]) / len(hybrid_energies)  # Strong hybrid ratio
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])

        # 5. Advanced hybrid metrics (2 features)
        advanced_metrics = self._calculate_advanced_rna_hybrid_metrics(graph, hybrid_scores)
        features.extend(advanced_metrics)

        return np.array(features[:15], dtype=np.float32)

    def _calculate_advanced_rna_hybrid_metrics(self, graph: ig.Graph, hybrid_scores: Dict[str, float]) -> List[float]:
        """
        Calculate advanced RNA hybrid metrics.

        Provides higher-level hybrid interaction features including diversity
        and connectivity patterns.

        Args:
            graph: RNA graph
            hybrid_scores: Dictionary of hybrid interaction scores

        Returns:
            List[float]: 2 advanced metrics: hybrid diversity and connectivity
        """
        metrics = []

        # 1. Hybrid diversity (Shannon entropy of hybrid scores)
        # Measures how evenly hybrid interactions are distributed across types
        hybrid_values = np.array(list(hybrid_scores.values()))
        if np.sum(hybrid_values) > 0:
            normalized = hybrid_values / np.sum(hybrid_values)
            entropy = -np.sum(normalized * np.log(normalized + 1e-10))
            metrics.append(entropy / np.log(len(hybrid_values)))  # Normalized entropy
        else:
            metrics.append(0.0)

        # 2. Hybrid connectivity pattern
        # Average degree of nodes involved in hybrid interactions
        try:
            hybrid_edges = [i for i, edge in enumerate(graph.es) if edge["is_hybrid"] == 1]
            if hybrid_edges:
                hybrid_subgraph = graph.subgraph_edges(hybrid_edges, delete_vertices=False)
                if hybrid_subgraph.ecount() > 0:
                    degrees = hybrid_subgraph.degree()
                    metrics.append(np.mean(degrees))
                else:
                    metrics.append(0.0)
            else:
                metrics.append(0.0)
        except:
            metrics.append(0.0)

        return metrics[:2]  # Ensure exactly 2 features
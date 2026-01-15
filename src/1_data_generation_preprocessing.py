"""
Multi-Agent System - Subgraph Isomorphism Problem
Group 4 - Data Generation & Preprocessing Module
================================
This module generates synthetic graph data and preprocesses it for MAS
"""

import networkx as nx
import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
import random

class GraphDataGenerator:
    """Generate synthetic graph datasets for subgraph isomorphism testing"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.generated_data = []
        
    def generate_erdos_renyi_graph(self, n_nodes: int, prob: float, graph_id: str):
        """Generate Erdos-Renyi random graph"""
        G = nx.erdos_renyi_graph(n_nodes, prob, seed=42)
        
        # Add node labels (random attributes)
        for node in G.nodes():
            G.nodes[node]['label'] = random.choice(['A', 'B', 'C', 'D', 'E'])
            G.nodes[node]['weight'] = random.randint(1, 10)
        
        # Add edge weights
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = random.randint(1, 5)
            
        return G
    
    def extract_subgraph(self, G: nx.Graph, subgraph_size: int):
        """Extract induced subgraph as query pattern"""
        nodes = list(G.nodes())
        if len(nodes) < subgraph_size:
            subgraph_size = len(nodes) // 2
            
        selected_nodes = random.sample(nodes, subgraph_size)
        Q = G.subgraph(selected_nodes).copy()
        
        # Map nodes to contiguous integers
        mapping = {old: new for new, old in enumerate(Q.nodes())}
        Q = nx.relabel_nodes(Q, mapping)
        
        return Q, selected_nodes
    
    def graph_to_features(self, G: nx.Graph, graph_type: str) -> Dict:
        """Extract comprehensive graph features (≥10 features)"""
        features = {
            'graph_type': graph_type,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            'max_degree': max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0,
            'min_degree': min(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0,
            'avg_clustering': nx.average_clustering(G),
            'num_triangles': sum(nx.triangles(G).values()) // 3,
            'is_connected': nx.is_connected(G),
            'num_components': nx.number_connected_components(G),
            'diameter': nx.diameter(G) if nx.is_connected(G) else -1,
            'avg_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else -1,
            'degree_centrality_avg': np.mean(list(nx.degree_centrality(G).values())),
            'betweenness_centrality_avg': np.mean(list(nx.betweenness_centrality(G).values())),
            'timestamp': datetime.now().isoformat()
        }
        return features
    
    def generate_dataset(self, n_samples=100):
        """Generate comprehensive dataset with multiple graph instances"""
        dataset = []
        
        # Configuration for graph generation
        configs = [
            (50, 0.1), (50, 0.2), (50, 0.3),
            (100, 0.1), (100, 0.15), (100, 0.2),
            (200, 0.08), (200, 0.12), (200, 0.15),
            (300, 0.05), (300, 0.08), (300, 0.1),
            (500, 0.04), (500, 0.06), (500, 0.08)
        ]
        
        sample_id = 0
        for _ in range(n_samples):
            # Random configuration
            n_nodes, prob = random.choice(configs)
            
            # Generate target graph
            graph_id = f"G_{sample_id}"
            G = self.generate_erdos_renyi_graph(n_nodes, prob, graph_id)
            
            # Generate query graph (subgraph)
            query_size = random.randint(5, min(15, n_nodes // 3))
            Q, ground_truth_nodes = self.extract_subgraph(G, query_size)
            
            # Extract features
            target_features = self.graph_to_features(G, 'target')
            query_features = self.graph_to_features(Q, 'query')
            
            # Combine features
            combined_features = {
                'sample_id': sample_id,
                'target_nodes': target_features['num_nodes'],
                'target_edges': target_features['num_edges'],
                'target_density': target_features['density'],
                'target_avg_degree': target_features['avg_degree'],
                'target_max_degree': target_features['max_degree'],
                'target_clustering': target_features['avg_clustering'],
                'query_nodes': query_features['num_nodes'],
                'query_edges': query_features['num_edges'],
                'query_density': query_features['density'],
                'query_avg_degree': query_features['avg_degree'],
                'query_max_degree': query_features['max_degree'],
                'query_clustering': query_features['avg_clustering'],
                'complexity_ratio': target_features['num_nodes'] / query_features['num_nodes'],
                'edge_ratio': target_features['num_edges'] / max(query_features['num_edges'], 1),
                'has_solution': True,  # Guaranteed by construction
                'graph_target': G,
                'graph_query': Q,
                'ground_truth': ground_truth_nodes
            }
            
            dataset.append(combined_features)
            sample_id += 1
            
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_dir='./data'):
        """Save dataset in multiple formats"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save as CSV (features only)
        df_features = pd.DataFrame([
            {k: v for k, v in item.items() 
             if k not in ['graph_target', 'graph_query', 'ground_truth']}
            for item in dataset
        ])
        df_features.to_csv(f'{output_dir}/graph_features.csv', index=False)
        print(f"✓ Saved features CSV: {len(df_features)} rows, {len(df_features.columns)} columns")
        
        # 2. Save graphs as edge lists
        with open(f'{output_dir}/graph_edgelists.txt', 'w') as f:
            for item in dataset:
                f.write(f"# Sample {item['sample_id']} - Target Graph\n")
                for edge in item['graph_target'].edges():
                    f.write(f"{edge[0]} {edge[1]}\n")
                f.write(f"# Sample {item['sample_id']} - Query Graph\n")
                for edge in item['graph_query'].edges():
                    f.write(f"{edge[0]} {edge[1]}\n")
                f.write("\n")
        
        # 3. Save complete dataset as pickle
        with open(f'{output_dir}/complete_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        print(f"✓ Saved complete dataset (pickle)")
        
        # 4. Save metadata as JSON
        metadata = {
            'total_samples': len(dataset),
            'features': list(df_features.columns),
            'date_created': datetime.now().isoformat(),
            'description': 'Synthetic graph dataset for subgraph isomorphism MAS'
        }
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return df_features


class DataPreprocessor:
    """Preprocess graph data for MAS consumption"""
    
    @staticmethod
    def normalize_graph(G: nx.Graph) -> nx.Graph:
        """Normalize graph node IDs to contiguous integers"""
        mapping = {node: idx for idx, node in enumerate(sorted(G.nodes()))}
        return nx.relabel_nodes(G, mapping)
    
    @staticmethod
    def compute_degree_statistics(G: nx.Graph) -> Dict:
        """Compute degree-based statistics for reasoning"""
        degrees = dict(G.degree())
        return {
            'degree_sequence': sorted(degrees.values(), reverse=True),
            'degree_distribution': pd.Series(degrees.values()).value_counts().to_dict(),
            'max_degree': max(degrees.values()) if degrees else 0,
            'min_degree': min(degrees.values()) if degrees else 0,
            'avg_degree': np.mean(list(degrees.values())) if degrees else 0
        }
    
    @staticmethod
    def create_adjacency_matrix(G: nx.Graph) -> np.ndarray:
        """Create adjacency matrix representation"""
        return nx.to_numpy_array(G)
    
    @staticmethod
    def prepare_for_mas(dataset: List[Dict]) -> List[Dict]:
        """Prepare dataset for MAS processing"""
        processed = []
        
        for item in dataset:
            G_norm = DataPreprocessor.normalize_graph(item['graph_target'])
            Q_norm = DataPreprocessor.normalize_graph(item['graph_query'])
            
            processed_item = {
                'sample_id': item['sample_id'],
                'target_graph': G_norm,
                'query_graph': Q_norm,
                'target_adj_matrix': DataPreprocessor.create_adjacency_matrix(G_norm),
                'query_adj_matrix': DataPreprocessor.create_adjacency_matrix(Q_norm),
                'target_stats': DataPreprocessor.compute_degree_statistics(G_norm),
                'query_stats': DataPreprocessor.compute_degree_statistics(Q_norm),
                'ground_truth': item['ground_truth'],
                'complexity_ratio': item['complexity_ratio']
            }
            processed.append(processed_item)
        
        return processed


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent System - Data Generation & Preprocessing")
    print("Group 4 - Subgraph Isomorphism Problem")
    print("=" * 60)
    
    # Step 1: Generate dataset
    print("\n[STEP 1] Generating synthetic graph dataset...")
    generator = GraphDataGenerator(seed=42)
    raw_dataset = generator.generate_dataset(n_samples=150)
    print(f"✓ Generated {len(raw_dataset)} graph pairs")
    
    # Step 2: Save raw data
    print("\n[STEP 2] Saving raw dataset...")
    df = generator.save_dataset(raw_dataset)
    print(f"✓ Dataset shape: {df.shape}")
    print(f"✓ Features: {list(df.columns)}")
    
    # Step 3: Preprocessing
    print("\n[STEP 3] Preprocessing data for MAS...")
    preprocessor = DataPreprocessor()
    processed_dataset = preprocessor.prepare_for_mas(raw_dataset)
    
    # Save processed data
    with open('./data/processed_dataset.pkl', 'wb') as f:
        pickle.dump(processed_dataset, f)
    print(f"✓ Processed {len(processed_dataset)} samples for MAS")
    
    # Display statistics
    print("\n[DATASET STATISTICS]")
    print(f"Total samples: {len(processed_dataset)}")
    print(f"Target graph sizes: {df['target_nodes'].min()}-{df['target_nodes'].max()} nodes")
    print(f"Query graph sizes: {df['query_nodes'].min()}-{df['query_nodes'].max()} nodes")
    print(f"Average complexity ratio: {df['complexity_ratio'].mean():.2f}")
    
    print("\n✓ Data generation and preprocessing completed!")
    print("=" * 60)
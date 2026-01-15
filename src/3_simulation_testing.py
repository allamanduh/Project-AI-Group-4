"""
Multi-Agent System - Simulation & Testing Module
Group 4 - Comprehensive Testing Scenarios
================================
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
import time
from datetime import datetime

# Import MAS components (assumes previous file is imported)
# from multi_agent_system import MultiAgentSystem, create_test_graphs


class MASSimulator:
    """Orchestrate comprehensive MAS testing and evaluation"""
    
    def __init__(self, mas_instance):
        self.mas = mas_instance
        self.test_results = []
        self.log_file = None
        
    def setup_logging(self, log_dir='./results'):
        """Setup logging infrastructure"""
        import os
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f'{log_dir}/mas_test_log_{timestamp}.txt'
        
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Multi-Agent System - Testing Log\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message: str, print_console=True):
        """Write message to log file and optionally print"""
        if print_console:
            print(message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(message + "\n")
    
    def run_test_scenario_1_small_graphs(self):
        """Test Scenario 1: Small graphs - Functionality verification"""
        
        self.log("\n" + "="*80)
        self.log("TEST SCENARIO 1: Small Graphs (Functionality Verification)")
        self.log("="*80)
        
        import networkx as nx
        
        test_cases = [
            # Test 1: Triangle query in larger graph
            {
                'name': 'Triangle Pattern',
                'target': nx.Graph([(0,1), (1,2), (2,3), (3,4), (1,3), (0,2)]),
                'query': nx.Graph([(0,1), (1,2), (0,2)])
            },
            # Test 2: Star pattern
            {
                'name': 'Star Pattern',
                'target': nx.star_graph(6),
                'query': nx.star_graph(3)
            },
            # Test 3: Path pattern
            {
                'name': 'Path Pattern',
                'target': nx.path_graph(10),
                'query': nx.path_graph(4)
            }
        ]
        
        results = []
        for i, test in enumerate(test_cases, 1):
            self.log(f"\n[Test 1.{i}] {test['name']}")
            self.log(f"Target: {test['target'].number_of_nodes()}n, {test['target'].number_of_edges()}e")
            self.log(f"Query: {test['query'].number_of_nodes()}n, {test['query'].number_of_edges()}e")
            
            result = self.mas.run_competition(test['target'], test['query'])
            results.append({
                'scenario': f"1.{i}",
                'name': test['name'],
                'result': result
            })
            
            self.log(f"Winner: {result['winner']}")
            self.log(f"Speedup: {result['speedup']:.2f}x")
        
        return results
    
    def run_test_scenario_2_scalability(self, dataset):
        """Test Scenario 2: Scalability testing with increasing complexity"""
        
        self.log("\n" + "="*80)
        self.log("TEST SCENARIO 2: Scalability Testing")
        self.log("="*80)
        
        # Select samples of increasing size
        sizes = [50, 100, 200, 300]
        results = []
        
        for size in sizes:
            # Find samples with target size close to desired size
            matching_samples = [
                s for s in dataset 
                if abs(s['target_graph'].number_of_nodes() - size) < 20
            ]
            
            if not matching_samples:
                continue
            
            sample = matching_samples[0]
            
            self.log(f"\n[Test 2] Size class: ~{size} nodes")
            self.log(f"Target: {sample['target_graph'].number_of_nodes()}n")
            self.log(f"Query: {sample['query_graph'].number_of_nodes()}n")
            
            result = self.mas.run_competition(
                sample['target_graph'],
                sample['query_graph']
            )
            
            results.append({
                'scenario': '2',
                'size_class': size,
                'actual_size': sample['target_graph'].number_of_nodes(),
                'result': result
            })
            
            self.log(f"Winner: {result['winner']}")
            self.log(f"Baseline time: {result['results']['baseline']['execution_time']:.4f}s")
            self.log(f"Intelligent time: {result['results']['intelligent']['execution_time']:.4f}s")
            self.log(f"Speedup: {result['speedup']:.2f}x")
        
        return results
    
    def run_test_scenario_3_edge_cases(self):
        """Test Scenario 3: Edge cases and corner cases"""
        
        self.log("\n" + "="*80)
        self.log("TEST SCENARIO 3: Edge Cases Testing")
        self.log("="*80)
        
        import networkx as nx
        
        test_cases = [
            {
                'name': 'Impossible Match (Query > Target)',
                'target': nx.Graph([(0,1), (1,2)]),
                'query': nx.complete_graph(5)
            },
            {
                'name': 'Dense Query in Sparse Target',
                'target': nx.erdos_renyi_graph(20, 0.1, seed=42),
                'query': nx.complete_graph(5)
            },
            {
                'name': 'Identical Graphs',
                'target': nx.cycle_graph(6),
                'query': nx.cycle_graph(6)
            }
        ]
        
        results = []
        for i, test in enumerate(test_cases, 1):
            self.log(f"\n[Test 3.{i}] {test['name']}")
            
            result = self.mas.run_competition(test['target'], test['query'])
            results.append({
                'scenario': f"3.{i}",
                'name': test['name'],
                'result': result
            })
            
            self.log(f"Baseline found: {result['results']['baseline']['found_solution']}")
            self.log(f"Intelligent found: {result['results']['intelligent']['found_solution']}")
        
        return results
    
    def run_test_scenario_4_stress_test(self, dataset):
        """Test Scenario 4: Stress testing with complex graphs"""
        
        self.log("\n" + "="*80)
        self.log("TEST SCENARIO 4: Stress Testing (Large Complex Graphs)")
        self.log("="*80)
        
        # Select largest graphs
        large_samples = sorted(
            dataset, 
            key=lambda x: x['target_graph'].number_of_nodes(),
            reverse=True
        )[:5]
        
        results = []
        for i, sample in enumerate(large_samples, 1):
            self.log(f"\n[Test 4.{i}] Large Graph Test")
            self.log(f"Target: {sample['target_graph'].number_of_nodes()}n, "
                    f"{sample['target_graph'].number_of_edges()}e")
            self.log(f"Query: {sample['query_graph'].number_of_nodes()}n, "
                    f"{sample['query_graph'].number_of_edges()}e")
            
            result = self.mas.run_competition(
                sample['target_graph'],
                sample['query_graph']
            )
            
            results.append({
                'scenario': f"4.{i}",
                'result': result
            })
            
            self.log(f"Baseline: {result['results']['baseline']['execution_time']:.4f}s, "
                    f"{result['results']['baseline']['nodes_explored']} nodes")
            self.log(f"Intelligent: {result['results']['intelligent']['execution_time']:.4f}s, "
                    f"{result['results']['intelligent']['nodes_explored']} nodes")
        
        return results
    
    def run_all_scenarios(self, dataset):
        """Execute all test scenarios"""
        
        self.setup_logging()
        
        all_results = {
            'scenario_1': self.run_test_scenario_1_small_graphs(),
            'scenario_2': self.run_test_scenario_2_scalability(dataset),
            'scenario_3': self.run_test_scenario_3_edge_cases(),
            'scenario_4': self.run_test_scenario_4_stress_test(dataset)
        }
        
        # Overall statistics
        self.log("\n" + "="*80)
        self.log("OVERALL STATISTICS")
        self.log("="*80)
        
        stats = self.mas.get_statistics()
        self.log(f"Total competitions: {stats['total_competitions']}")
        self.log(f"Intelligent agent wins: {stats['intelligent_wins']} "
                f"({stats['intelligent_win_rate']*100:.1f}%)")
        self.log(f"Baseline agent wins: {stats['baseline_wins']}")
        self.log(f"Average speedup: {stats['average_speedup']:.2f}x")
        
        self.log(f"\n✓ Log saved to: {self.log_file}")
        
        return all_results, stats


class MASEvaluator:
    """Evaluate and visualize MAS performance"""
    
    def __init__(self, mas_instance):
        self.mas = mas_instance
        
    def create_performance_report(self, output_dir='./results'):
        """Generate comprehensive performance report with visualizations"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.mas.results_history:
            print("No results to evaluate. Run tests first.")
            return
        
        # Extract data for analysis
        data = []
        for result in self.mas.results_history:
            data.append({
                'target_size': result['target_size'],
                'query_size': result['query_size'],
                'baseline_time': result['results']['baseline']['execution_time'],
                'intelligent_time': result['results']['intelligent']['execution_time'],
                'baseline_nodes': result['results']['baseline']['nodes_explored'],
                'intelligent_nodes': result['results']['intelligent']['nodes_explored'],
                'speedup': result['speedup'],
                'winner': result['winner']
            })
        
        df = pd.DataFrame(data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Execution time comparison
        ax1 = axes[0, 0]
        width = 0.35
        x = np.arange(len(df))
        ax1.bar(x - width/2, df['baseline_time'], width, label='Baseline', alpha=0.8)
        ax1.bar(x + width/2, df['intelligent_time'], width, label='Intelligent', alpha=0.8)
        ax1.set_xlabel('Test Case')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup over graph size
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['target_size'], df['speedup'], 
                             c=df['query_size'], cmap='viridis', s=100, alpha=0.6)
        ax2.set_xlabel('Target Graph Size (nodes)')
        ax2.set_ylabel('Speedup (Baseline/Intelligent)')
        ax2.set_title('Speedup vs Graph Size')
        ax2.axhline(y=1, color='r', linestyle='--', label='No speedup')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Query Size')
        
        # Plot 3: Nodes explored comparison
        ax3 = axes[1, 0]
        ax3.scatter(df['baseline_nodes'], df['intelligent_nodes'], alpha=0.6, s=100)
        max_nodes = max(df['baseline_nodes'].max(), df['intelligent_nodes'].max())
        ax3.plot([0, max_nodes], [0, max_nodes], 'r--', label='Equal exploration')
        ax3.set_xlabel('Baseline Nodes Explored')
        ax3.set_ylabel('Intelligent Nodes Explored')
        ax3.set_title('Search Space Exploration')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Win rate
        ax4 = axes[1, 1]
        win_counts = df['winner'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax4.pie(win_counts.values, labels=win_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax4.set_title('Agent Win Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved performance visualization: {output_dir}/performance_analysis.png")
        
        # Save statistics to CSV
        df.to_csv(f'{output_dir}/detailed_results.csv', index=False)
        print(f"✓ Saved detailed results: {output_dir}/detailed_results.csv")
        
        # Generate summary statistics
        summary = {
            'total_tests': len(df),
            'intelligent_wins': (df['winner'] == 'intelligent').sum(),
            'baseline_wins': (df['winner'] == 'baseline').sum(),
            'ties': (df['winner'] == 'tie').sum(),
            'avg_speedup': df['speedup'].mean(),
            'median_speedup': df['speedup'].median(),
            'max_speedup': df['speedup'].max(),
            'avg_baseline_time': df['baseline_time'].mean(),
            'avg_intelligent_time': df['intelligent_time'].mean(),
            'time_reduction_pct': ((df['baseline_time'].mean() - df['intelligent_time'].mean()) / 
                                  df['baseline_time'].mean() * 100),
            'avg_nodes_reduction_pct': ((df['baseline_nodes'].mean() - df['intelligent_nodes'].mean()) / 
                                       df['baseline_nodes'].mean() * 100)
        }
        
        # Save summary as JSON
        def to_python_type(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        summary_clean = {k: to_python_type(v) for k, v in summary.items()}
        with open(f'{output_dir}/summary_statistics.json', 'w') as f:
            json.dump(summary_clean, f, indent=2)

        print(f"✓ Saved summary statistics: {output_dir}/summary_statistics.json")
        
        return summary


def main_simulation_pipeline():
    """Main simulation pipeline"""
    
    print("="*80)
    print("Multi-Agent System - Comprehensive Testing & Evaluation")
    print("Group 4 - Subgraph Isomorphism Solver")
    print("="*80)
    
    # Load preprocessed dataset
    print("\n[1] Loading dataset...")
    try:
        with open('./data/processed_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
        print(f"✓ Loaded {len(dataset)} samples")
    except FileNotFoundError:
        print("✗ Dataset not found. Run data preprocessing first.")
        return
    
    # Initialize MAS
    print("\n[2] Initializing Multi-Agent System...")
    # Import here to avoid circular dependency
    mas = MultiAgentSystem()
    print("✓ MAS initialized with 2 agents (Baseline, Intelligent)")
    
    # Run simulations
    print("\n[3] Running test scenarios...")
    simulator = MASSimulator(mas)
    results, stats = simulator.run_all_scenarios(dataset)
    
    # Evaluate performance
    print("\n[4] Generating performance report...")
    evaluator = MASEvaluator(mas)
    summary = evaluator.create_performance_report()
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total tests executed: {summary['total_tests']}")
    print(f"Intelligent agent win rate: {summary['intelligent_wins']/summary['total_tests']*100:.1f}%")
    print(f"Average speedup: {summary['avg_speedup']:.2f}x")
    print(f"Time reduction: {summary['time_reduction_pct']:.1f}%")
    print(f"Search space reduction: {summary['avg_nodes_reduction_pct']:.1f}%")
    print("\n✓ All testing and evaluation completed!")
    print("="*80)


if __name__ == "__main__":
    main_simulation_pipeline()
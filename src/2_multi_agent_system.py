"""
Multi-Agent System Implementation
Group 4 - Subgraph Isomorphism Solver
================================
Heterogeneous Competitive MAS with Intelligent Capabilities:
- Sensing & Reasoning
- Planning
- Searching
"""

import networkx as nx
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Set
from abc import ABC, abstractmethod
from collections import defaultdict
import copy


class Agent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.execution_time = 0
        self.steps_taken = 0
        self.nodes_explored = 0
        
    @abstractmethod
    def solve(self, target_graph: nx.Graph, query_graph: nx.Graph) -> Dict:
        """Main solving method - must be implemented by subclasses"""
        pass
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.execution_time = 0
        self.steps_taken = 0
        self.nodes_explored = 0


class SensingReasoningAgent:
    """
    Agent Capability 1: Sensing & Reasoning
    Analyzes graph topology and computes structural features
    """
    
    @staticmethod
    def sense_graph_features(G: nx.Graph) -> Dict:
        """Extract topological features from graph"""
        degree_dict = dict(G.degree())
        
        features = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'degrees': degree_dict,
            'degree_sequence': sorted(degree_dict.values(), reverse=True),
            'max_degree': max(degree_dict.values()) if degree_dict else 0,
            'min_degree': min(degree_dict.values()) if degree_dict else 0,
            'avg_degree': np.mean(list(degree_dict.values())) if degree_dict else 0
        }
        
        return features
    
    @staticmethod
    def compute_node_rarity(node: int, G_query: nx.Graph, G_target: nx.Graph) -> float:
        """
        Reasoning: Calculate how "rare" or "unique" a node is
        Lower rarity score = more unique = should be matched first
        """
        query_degree = G_query.degree(node)
        
        # Count how many nodes in target have same or higher degree
        target_degrees = [d for n, d in G_target.degree()]
        candidates_count = sum(1 for d in target_degrees if d >= query_degree)
        
        # Rarity: fewer candidates = more rare = lower score
        rarity_score = candidates_count / G_target.number_of_nodes() if G_target.number_of_nodes() > 0 else 1.0
        
        return rarity_score
    
    @staticmethod
    def reason_about_feasibility(G_query: nx.Graph, G_target: nx.Graph) -> Dict:
        """Use reasoning to check if subgraph isomorphism is feasible"""
        query_features = SensingReasoningAgent.sense_graph_features(G_query)
        target_features = SensingReasoningAgent.sense_graph_features(G_target)
        
        # Necessary conditions for subgraph isomorphism
        feasible = True
        reasons = []
        
        if query_features['num_nodes'] > target_features['num_nodes']:
            feasible = False
            reasons.append("Query has more nodes than target")
            
        if query_features['num_edges'] > target_features['num_edges']:
            feasible = False
            reasons.append("Query has more edges than target")
            
        if query_features['max_degree'] > target_features['max_degree']:
            feasible = False
            reasons.append("Query max degree exceeds target max degree")
        
        # Check degree sequence compatibility
        query_deg_seq = sorted(query_features['degree_sequence'], reverse=True)
        target_deg_seq = sorted(target_features['degree_sequence'], reverse=True)
        
        for qd in query_deg_seq:
            if not any(td >= qd for td in target_deg_seq):
                feasible = False
                reasons.append(f"No target node can match query degree {qd}")
                break
        
        return {
            'feasible': feasible,
            'reasons': reasons if not feasible else ["All necessary conditions satisfied"],
            'query_features': query_features,
            'target_features': target_features
        }


class PlanningAgent:
    """
    Agent Capability 2: Planning
    Creates optimal search strategy using Infrequent-First heuristic
    """
    
    @staticmethod
    def create_search_plan(G_query: nx.Graph, G_target: nx.Graph) -> List[int]:
        """
        Planning Strategy: "Infrequent First"
        Order query nodes by increasing rarity (most constrained first)
        This maximizes early pruning in backtracking
        """
        node_priorities = []
        
        for node in G_query.nodes():
            rarity = SensingReasoningAgent.compute_node_rarity(node, G_query, G_target)
            degree = G_query.degree(node)
            
            # Priority score: combine rarity and degree
            # Lower score = higher priority = should be matched first
            priority_score = rarity * 0.7 + (1.0 / (degree + 1)) * 0.3
            
            node_priorities.append((node, priority_score, degree, rarity))
        
        # Sort by priority (ascending - lowest score first)
        node_priorities.sort(key=lambda x: x[1])
        
        optimal_plan = [node for node, _, _, _ in node_priorities]
        
        plan_info = {
            'ordered_nodes': optimal_plan,
            'priorities': {node: (score, deg, rar) 
                          for node, score, deg, rar in node_priorities},
            'strategy': 'Infrequent-First (Most Constrained First)'
        }
        
        return plan_info
    
    @staticmethod
    def get_candidate_set(query_node: int, G_query: nx.Graph, G_target: nx.Graph) -> Set[int]:
        """
        Planning: Generate candidate nodes in target that could match query node
        Uses degree filtering for pruning
        """
        query_degree = G_query.degree(query_node)
        
        candidates = {
            n for n in G_target.nodes() 
            if G_target.degree(n) >= query_degree
        }
        
        return candidates


class SearchingAgent:
    """
    Agent Capability 3: Searching
    Executes backtracking search with refinement
    """
    
    def __init__(self):
        self.nodes_explored = 0
        self.backtrack_count = 0
        
    def search_with_backtracking(self, 
                                 G_query: nx.Graph, 
                                 G_target: nx.Graph,
                                 search_plan: List[int]) -> Optional[Dict[int, int]]:
        """
        Execute backtracking search following the optimal plan
        Returns mapping: query_node -> target_node
        """
        self.nodes_explored = 0
        self.backtrack_count = 0
        
        # Initialize candidate sets for each query node
        candidates = {}
        for q_node in G_query.nodes():
            candidates[q_node] = PlanningAgent.get_candidate_set(q_node, G_query, G_target)
        
        # Start recursive backtracking
        mapping = {}
        used_target_nodes = set()
        
        result = self._recursive_backtrack(
            0, search_plan, mapping, used_target_nodes,
            G_query, G_target, candidates
        )
        
        return result
    
    def _recursive_backtrack(self, 
                            plan_idx: int,
                            search_plan: List[int],
                            mapping: Dict[int, int],
                            used_nodes: Set[int],
                            G_query: nx.Graph,
                            G_target: nx.Graph,
                            candidates: Dict[int, Set[int]]) -> Optional[Dict[int, int]]:
        """Recursive backtracking with constraint checking"""
        
        # Base case: all query nodes mapped
        if plan_idx == len(search_plan):
            return mapping.copy()
        
        query_node = search_plan[plan_idx]
        self.nodes_explored += 1
        
        # Try each candidate for current query node
        for target_node in candidates[query_node]:
            if target_node in used_nodes:
                continue
            
            # Check if mapping is consistent with edges
            if self._is_consistent(query_node, target_node, mapping, G_query, G_target):
                # Make assignment
                mapping[query_node] = target_node
                used_nodes.add(target_node)
                
                # Recurse
                result = self._recursive_backtrack(
                    plan_idx + 1, search_plan, mapping, used_nodes,
                    G_query, G_target, candidates
                )
                
                if result is not None:
                    return result
                
                # Backtrack
                del mapping[query_node]
                used_nodes.remove(target_node)
                self.backtrack_count += 1
        
        return None
    
    def _is_consistent(self, 
                       q_node: int, 
                       t_node: int,
                       mapping: Dict[int, int],
                       G_query: nx.Graph,
                       G_target: nx.Graph) -> bool:
        """Check if current assignment is consistent with existing mapping"""
        
        # Check all already-mapped neighbors
        for q_neighbor in G_query.neighbors(q_node):
            if q_neighbor in mapping:
                t_neighbor = mapping[q_neighbor]
                
                # Edge must exist in target
                if not G_target.has_edge(t_node, t_neighbor):
                    return False
        
        return True


class BaselineAgent(Agent):
    """
    Baseline Agent (Ullmann-style)
    Simple backtracking without intelligent planning
    """
    
    def __init__(self):
        super().__init__("baseline_agent", "Baseline")
        self.searcher = SearchingAgent()
        
    def solve(self, target_graph: nx.Graph, query_graph: nx.Graph) -> Dict:
        """Solve using naive node ordering (0, 1, 2, ...)"""
        
        start_time = time.time()
        
        # Simple ordering: just use node IDs as-is
        naive_plan = list(query_graph.nodes())
        
        # Execute search
        mapping = self.searcher.search_with_backtracking(
            query_graph, target_graph, naive_plan
        )
        
        end_time = time.time()
        
        self.execution_time = end_time - start_time
        self.nodes_explored = self.searcher.nodes_explored
        self.steps_taken = self.searcher.backtrack_count
        
        return {
            'agent_type': self.agent_type,
            'mapping': mapping,
            'found_solution': mapping is not None,
            'execution_time': self.execution_time,
            'nodes_explored': self.nodes_explored,
            'backtracks': self.steps_taken,
            'search_plan': naive_plan
        }


class IntelligentAgent(Agent):
    """
    Intelligent Agent (RSM-Inspired)
    Combines Sensing, Reasoning, Planning, and Searching
    """
    
    def __init__(self):
        super().__init__("intelligent_agent", "Intelligent")
        self.sensing_agent = SensingReasoningAgent()
        self.planning_agent = PlanningAgent()
        self.searching_agent = SearchingAgent()
        
    def solve(self, target_graph: nx.Graph, query_graph: nx.Graph) -> Dict:
        """Solve using intelligent multi-capability approach"""
        
        start_time = time.time()
        
        # CAPABILITY 1: Sensing & Reasoning
        feasibility = self.sensing_agent.reason_about_feasibility(
            query_graph, target_graph
        )
        
        if not feasibility['feasible']:
            end_time = time.time()
            return {
                'agent_type': self.agent_type,
                'mapping': None,
                'found_solution': False,
                'execution_time': end_time - start_time,
                'nodes_explored': 0,
                'backtracks': 0,
                'early_termination': True,
                'reason': feasibility['reasons']
            }
        
        # CAPABILITY 2: Planning
        plan_info = self.planning_agent.create_search_plan(query_graph, target_graph)
        optimal_plan = plan_info['ordered_nodes']
        
        # CAPABILITY 3: Searching
        mapping = self.searching_agent.search_with_backtracking(
            query_graph, target_graph, optimal_plan
        )
        
        end_time = time.time()
        
        self.execution_time = end_time - start_time
        self.nodes_explored = self.searching_agent.nodes_explored
        self.steps_taken = self.searching_agent.backtrack_count
        
        return {
            'agent_type': self.agent_type,
            'mapping': mapping,
            'found_solution': mapping is not None,
            'execution_time': self.execution_time,
            'nodes_explored': self.nodes_explored,
            'backtracks': self.steps_taken,
            'search_plan': optimal_plan,
            'plan_strategy': plan_info['strategy'],
            'feasibility_check': feasibility,
            'early_termination': False
        }


class MultiAgentSystem:
    """
    Heterogeneous Competitive Multi-Agent System
    Manages multiple agents solving the same problem
    """
    
    def __init__(self):
        self.agents = {
            'baseline': BaselineAgent(),
            'intelligent': IntelligentAgent()
        }
        self.results_history = []
        
    def run_competition(self, target_graph: nx.Graph, query_graph: nx.Graph) -> Dict:
        """Run both agents competitively on the same problem"""
        
        print(f"\n{'='*60}")
        print(f"MAS COMPETITION: Target({target_graph.number_of_nodes()}n, "
              f"{target_graph.number_of_edges()}e) vs Query({query_graph.number_of_nodes()}n, "
              f"{query_graph.number_of_edges()}e)")
        print(f"{'='*60}")
        
        results = {}
        
        # Run each agent
        for agent_name, agent in self.agents.items():
            print(f"\n[{agent_name.upper()} AGENT] Starting...")
            agent.reset_metrics()
            
            result = agent.solve(target_graph, query_graph)
            results[agent_name] = result
            
            print(f"  ✓ Completed in {result['execution_time']:.4f}s")
            print(f"  ✓ Nodes explored: {result['nodes_explored']}")
            print(f"  ✓ Solution found: {result['found_solution']}")
        
        # Determine winner
        winner = self._determine_winner(results)
        
        competition_result = {
            'timestamp': time.time(),
            'target_size': target_graph.number_of_nodes(),
            'query_size': query_graph.number_of_nodes(),
            'results': results,
            'winner': winner,
            'speedup': self._calculate_speedup(results)
        }
        
        self.results_history.append(competition_result)
        
        return competition_result
    
    def _determine_winner(self, results: Dict) -> str:
        """Determine which agent performed better"""
        
        # Both failed
        if not results['baseline']['found_solution'] and not results['intelligent']['found_solution']:
            return 'tie'
        
        # Only one succeeded
        if results['baseline']['found_solution'] and not results['intelligent']['found_solution']:
            return 'baseline'
        if results['intelligent']['found_solution'] and not results['baseline']['found_solution']:
            return 'intelligent'
        
        # Both succeeded - compare efficiency
        if results['intelligent']['execution_time'] < results['baseline']['execution_time']:
            return 'intelligent'
        else:
            return 'baseline'
    
    def _calculate_speedup(self, results: Dict) -> float:
        """Calculate speedup of intelligent agent over baseline"""
        
        if results['baseline']['execution_time'] == 0:
            return 0.0
        
        return results['baseline']['execution_time'] / results['intelligent']['execution_time']
    
    def get_statistics(self) -> Dict:
        """Get aggregate statistics across all competitions"""
        
        if not self.results_history:
            return {}
        
        baseline_wins = sum(1 for r in self.results_history if r['winner'] == 'baseline')
        intelligent_wins = sum(1 for r in self.results_history if r['winner'] == 'intelligent')
        ties = sum(1 for r in self.results_history if r['winner'] == 'tie')
        
        avg_speedup = np.mean([r['speedup'] for r in self.results_history 
                               if r['speedup'] > 0])
        
        return {
            'total_competitions': len(self.results_history),
            'baseline_wins': baseline_wins,
            'intelligent_wins': intelligent_wins,
            'ties': ties,
            'intelligent_win_rate': intelligent_wins / len(self.results_history),
            'average_speedup': avg_speedup
        }


# Utility function for testing
def create_test_graphs():
    """Create simple test graphs for quick validation"""
    
    # Small target graph
    G_target = nx.Graph()
    G_target.add_edges_from([
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)
    ])
    
    # Query is a small triangle
    G_query = nx.Graph()
    G_query.add_edges_from([(0, 1), (1, 2), (0, 2)])
    
    return G_target, G_query


if __name__ == "__main__":
    print("Multi-Agent System - Subgraph Isomorphism")
    print("Testing MAS components...")
    
    # Quick test
    G_target, G_query = create_test_graphs()
    
    mas = MultiAgentSystem()
    result = mas.run_competition(G_target, G_query)
    
    print(f"\n{'='*60}")
    print(f"COMPETITION RESULTS")
    print(f"{'='*60}")
    print(f"Winner: {result['winner'].upper()}")
    print(f"Speedup: {result['speedup']:.2f}x")
    print(f"\n✓ MAS implementation complete!")
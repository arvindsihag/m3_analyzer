import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import heapq
from datetime import datetime
import os

class ActionSequenceAnalyzer:
    def __init__(self, action_base_file: str = "action_base.json"):
        self.operator_sequences = defaultdict(list)
        self.sequence_graphs = {}
        self.transition_counts = defaultdict(Counter)
        self.assembly_dependencies = {}
        self.assembly_workflow = {}
        self.action_base_file = action_base_file
        self.action_base = self._load_action_base()
        self.unique_actions = set()
        self.action_times = {}


    def _load_action_base(self) -> Dict:
        if os.path.exists(self.action_base_file):
            try:
                with open(self.action_base_file, 'r') as f:
                    base = json.load(f)
                   
                    if 'best_performance' not in base:
                        base['best_performance'] = {}
                    if 'live_actions' not in base:
                        base['live_actions'] = {}
                    return base
            except:
                return self._initialize_action_base()
        else:
            return self._initialize_action_base()


    def _initialize_action_base(self) -> Dict:
        return {
            'best_performance': {},  # Store best performance data
            'live_actions': {},      # Store all operator data
            'statistics': {
                'total_actions_recorded': 0,
                'unique_actions': 0,
                'best_actions_count': 0,
                'live_actions_count': 0,
                'last_updated': datetime.now().isoformat()
            },
            'workflow_patterns': {},
            'metadata': {
                'version': '1.0',
                'created_date': datetime.now().isoformat(),
                'description': 'Dynamic action performance database'
            }
        }


    def _save_action_base(self):
        self.action_base['statistics']['unique_actions'] = len(set(
            list(self.action_base['best_performance'].keys()) + 
            list(self.action_base['live_actions'].keys())
        ))
        self.action_base['statistics']['best_actions_count'] = len(self.action_base['best_performance'])
        self.action_base['statistics']['live_actions_count'] = len(self.action_base['live_actions'])
        self.action_base['statistics']['last_updated'] = datetime.now().isoformat()
        
        with open(self.action_base_file, 'w') as f:
            json.dump(self.action_base, f, indent=2)


    def extract_action_components(self, action_name: str) -> Dict:
        clean_name = re.sub(r'^\d+_', '', action_name)
        clean_name = re.sub(r'^clip_', '', clean_name)
        clean_name = re.sub(r'\.mp4$', '', clean_name)
        clean_name = re.sub(r'_rgb$', '', clean_name)
        

        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', clean_name.replace('_', ' '))
        word_count = len(words) if words else 1

        action_type = "unknown"
        if any(word in clean_name.lower() for word in ['take', 'pick', 'grab']):
            action_type = "take"
        elif any(word in clean_name.lower() for word in ['put', 'place', 'attach']):
            action_type = "put"
        elif any(word in clean_name.lower() for word in ['align', 'position', 'adjust']):
            action_type = "align"
        elif any(word in clean_name.lower() for word in ['screw', 'fasten', 'tighten']):
            action_type = "fasten"
        elif any(word in clean_name.lower() for word in ['check', 'verify', 'inspect']):
            action_type = "inspect"
        
        return {
            'full_action': clean_name,
            'components': words,
            'word_count': word_count,
            'action_type': action_type,
            'simplified': clean_name.replace('_', '').lower()
        }



    def record_action_execution(self, action_name: str, execution_time: float, operator_id: str, timestamp: datetime):
        clean_action = self.extract_action_components(action_name)['full_action']
        self.action_times[clean_action] = execution_time
        if clean_action not in self.action_base['live_actions']:
            self.action_base['live_actions'][clean_action] = {
                'execution_times': [],
                'operators': [],
                'first_seen': timestamp.isoformat(),
                'last_seen': timestamp.isoformat(),
                'total_executions': 0,
                'components': self.extract_action_components(clean_action),
                'operator_stats': {}
            }
        
        live_record = self.action_base['live_actions'][clean_action]
        live_record['execution_times'].append(execution_time)
        
        if operator_id not in live_record['operators']:
            live_record['operators'].append(operator_id)
        if operator_id not in live_record['operator_stats']:
            live_record['operator_stats'][operator_id] = {
                'execution_times': [],
                'execution_count': 0,
                'best_time': float('inf'),
                'average_time': 0.0
            }
        
        op_stats = live_record['operator_stats'][operator_id]
        op_stats['execution_times'].append(execution_time)
        op_stats['execution_count'] += 1
        op_stats['best_time'] = min(op_stats['best_time'], execution_time)
        op_stats['average_time'] = sum(op_stats['execution_times']) / len(op_stats['execution_times'])
        
        live_record['last_seen'] = timestamp.isoformat()
        live_record['total_executions'] += 1
        
        all_times = live_record['execution_times']
        live_record['average_time'] = sum(all_times) / len(all_times)
        live_record['best_time'] = min(all_times)
        live_record['worst_time'] = max(all_times)
        live_record['std_dev'] = np.std(all_times) if len(all_times) > 1 else 0

        self._update_best_performance(clean_action, execution_time, operator_id, timestamp)
        
        self.action_base['statistics']['total_actions_recorded'] += 1
        self._save_action_base()


    def _update_best_performance(self, action_name: str, execution_time: float, operator_id: str, timestamp: datetime):
        if action_name not in self.action_base['best_performance']:
            self.action_base['best_performance'][action_name] = {
                'execution_times': [execution_time],
                'operators': [operator_id],
                'best_operator': operator_id,
                'best_time': execution_time,
                'first_seen': timestamp.isoformat(),
                'last_seen': timestamp.isoformat(),
                'total_executions_considered': 1,
                'components': self.extract_action_components(action_name)
            }
        else:
            best_record = self.action_base['best_performance'][action_name]

            if execution_time < best_record['best_time']:
                best_record['execution_times'] = [execution_time]
                best_record['operators'] = [operator_id]
                best_record['best_operator'] = operator_id
                best_record['best_time'] = execution_time
                best_record['last_seen'] = timestamp.isoformat()
            
            best_record['total_executions_considered'] += 1
    
    def get_action_time_estimate(self, action_name: str, use_best: bool = False) -> float:
        clean_action = self.extract_action_components(action_name)['full_action']
        if use_best and clean_action in self.action_base['best_performance']:
            return self.action_base['best_performance'][clean_action]['best_time']
        elif clean_action in self.action_base['live_actions']:
            return self.action_base['live_actions'][clean_action]['average_time']
        else:
            components = self.extract_action_components(action_name)
            complexity = components.get('word_count', 1)
            return complexity * 2.0
    
    def get_action_statistics(self, action_name: str) -> Dict:
        clean_action = self.extract_action_components(action_name)['full_action']
        stats = {
            'action': clean_action,
            'exists_in_best': False,
            'exists_in_live': False,
            'recommended_time': self.get_action_time_estimate(clean_action),
            'best_possible_time': self.get_action_time_estimate(clean_action, use_best=True)
        }
        
        if clean_action in self.action_base['best_performance']:
            best = self.action_base['best_performance'][clean_action]
            stats.update({
                'exists_in_best': True,
                'best_time': best['best_time'],
                'best_operator': best['best_operator'],
                'executions_considered': best['total_executions_considered']
            })
        
        if clean_action in self.action_base['live_actions']:
            live = self.action_base['live_actions'][clean_action]
            stats.update({
                'exists_in_live': True,
                'average_time': live['average_time'],
                'total_executions': live['total_executions'],
                'unique_operators': len(live['operators']),
                'time_std_dev': live.get('std_dev', 0),
                'operator_performance': live['operator_stats']
            })
        
        return stats
    
    def get_performance_ranking(self) -> List[Dict]:
        ranking = []
        
        for action_name in set(list(self.action_base['best_performance'].keys()) + list(self.action_base['live_actions'].keys())):
            stats = self.get_action_statistics(action_name)
            
            if stats['exists_in_best'] and stats['exists_in_live']:
                improvement_potential = stats['average_time'] - stats['best_time']
                improvement_percentage = (improvement_potential / stats['average_time']) * 100
                
                ranking.append({
                    'action': action_name,
                    'average_time': stats['average_time'],
                    'best_time': stats['best_time'],
                    'improvement_potential': improvement_potential,
                    'improvement_percentage': improvement_percentage,
                    'best_operator': stats.get('best_operator', 'Unknown')
                })
        
        return sorted(ranking, key=lambda x: x['improvement_percentage'], reverse=True)


    def analyze_assembly_structure(self, all_video_sequences: Dict[str, List[str]], execution_data: Dict[str, List[Dict]] = None):

        self.unique_actions = set()

        if execution_data:
            for operator_id, executions in execution_data.items():
                for execution in executions:
                    action_name = execution['action_name']
                    duration = execution.get('duration', 0)
                    timestamp = datetime.fromtimestamp(execution.get('timestamp', 0))
                    
                    if duration > 0:
                        self.record_action_execution(action_name, duration, operator_id, timestamp)
                        self.unique_actions.add(action_name)

        self._build_assembly_dependencies()
        self._build_assembly_workflow()
        
        print(f" Performance Database Updated:")
        print(f"   Best performances: {len(self.action_base['best_performance'])} actions")
        print(f"   Live tracking: {len(self.action_base['live_actions'])} actions")
        print(f"   Total executions: {self.action_base['statistics']['total_actions_recorded']}")


    def add_operator_sequence(self, operator_id: str, action_name: str):

        if operator_id in self.operator_sequences:
            if not self.operator_sequences[operator_id] or self.operator_sequences[operator_id][-1] != action_name:
                self.operator_sequences[operator_id].append(action_name)
                self._build_sequence_graph(operator_id)
        else:
            self.operator_sequences[operator_id] = [action_name]




    def _clean_action_name(self, action_name: str) -> str:
        components = self.extract_action_components(action_name)
        return components['full_action']

    def _build_sequence_graph(self, operator_id: str):
        sequence = self.operator_sequences[operator_id]
        if not sequence:
            return
            
        G = nx.DiGraph()

        action_counts = Counter(sequence)
        for action, count in action_counts.items():
            G.add_node(action, count=count, frequency=count/len(sequence))

        transitions = defaultdict(int)
        for i in range(len(sequence) - 1):
            source = sequence[i]
            target = sequence[i + 1]
            transitions[(source, target)] += 1
        
        for (source, target), weight in transitions.items():
            G.add_edge(source, target, weight=weight, 
                      probability=weight/action_counts[source])
        
        self.sequence_graphs[operator_id] = G

    def predict_next_actions(self, operator_id: str, previous_actions: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        if operator_id not in self.sequence_graphs or not previous_actions:
            return []
        
        G = self.sequence_graphs[operator_id]
        current_action = previous_actions[-1]
        
        if current_action not in G:
            similar_actions = [action for action in G.nodes() 
                              if current_action in action or action in current_action]
            if similar_actions:
                current_action = similar_actions[0]
            else:
                return []

        next_actions = []
        for target in G.successors(current_action):
            probability = G[current_action][target].get('probability', 0)
            next_actions.append((target, probability))

        if not next_actions:
            if current_action in self.assembly_workflow:
                next_actions = [(action, 0.7) for action in self.assembly_workflow[current_action][:top_n]]
        
        # Sort by probability and return top N
        next_actions.sort(key=lambda x: x[1], reverse=True)
        return next_actions[:top_n]

    def find_optimal_path(self, operator_id: str) -> Dict:
        if operator_id not in self.sequence_graphs:
            return {}
        
        G = self.sequence_graphs[operator_id]

        cost_graph = nx.DiGraph()
        for u, v, data in G.edges(data=True):
            cost = 1 - data.get('probability', 0)
            cost_graph.add_edge(u, v, weight=cost)

        start_nodes = [node for node in cost_graph.nodes() if cost_graph.in_degree(node) == 0]
        end_nodes = [node for node in cost_graph.nodes() if cost_graph.out_degree(node) == 0]
        
        if not start_nodes or not end_nodes:
            return {}

        optimal_path = None
        min_cost = float('inf')
        
        for start in start_nodes:
            for end in end_nodes:
                try:
                    path = nx.shortest_path(cost_graph, start, end, weight='weight')
                    path_cost = nx.shortest_path_length(cost_graph, start, end, weight='weight')
                    
                    if path_cost < min_cost:
                        min_cost = path_cost
                        optimal_path = path
                except nx.NetworkXNoPath:
                    continue
        
        if not optimal_path:
            return {}
        
        actual_time = sum(self.action_times.get(action, 0) for action in self.operator_sequences[operator_id])
        optimal_time = sum(self.action_times.get(action, 0) for action in optimal_path)
        time_efficiency = (optimal_time / actual_time) * 100 if actual_time > 0 else 100
        
        return {
            'optimal_path': optimal_path,
            'actual_path': self.operator_sequences[operator_id],
            'time_efficiency': time_efficiency,
            'optimal_time': optimal_time,
            'actual_time': actual_time,
            'path_improvement': max(0, actual_time - optimal_time)
        }

    def generate_sequence_visualization(self, operator_id: str, output_dir: str) -> str:
        if operator_id not in self.sequence_graphs:
            return ""
        
        G = self.sequence_graphs[operator_id]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"action_transition_sequence_{timestamp}_{operator_id}.png"
        output_path = os.path.join(output_dir, filename)
        
        plt.figure(figsize=(16, 12))

        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)

        node_sizes = [G.nodes[node].get('count', 1) * 800 for node in G.nodes()]

        node_colors = []
        for node in G.nodes():
            action_type = self.extract_action_components(node)['action_type']
            color_map = {
                'take': '#2ecc71',      # Green for take actions
                'put': '#3498db',       # Blue for put actions  
                'align': '#f39c12',     # Orange for align actions
                'fasten': '#e74c3c',    # Red for fasten actions
                'inspect': '#9b59b6',   # Purple for inspect actions
                'unknown': '#95a5a6'    # Gray for unknown
            }
            node_colors.append(color_map.get(action_type, '#95a5a6'))
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, edgecolors='black', linewidths=2)

        edge_widths = [G[u][v].get('weight', 1) * 1.0 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color='#34495e', arrows=True, arrowsize=20)

        edge_labels = {(u, v): f"{G[u][v].get('weight', 0)}x\n({G[u][v].get('probability', 0):.0%})" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        node_labels = {node: node.replace('_', '\n').upper() for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='bold')
        
        plt.title(f'Action Transition Sequence - Operator {operator_id}\n'
                 f'Total Actions: {len(self.operator_sequences[operator_id])}, '
                 f'Unique Actions: {len(G.nodes())}', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    def generate_analysis_report(self, operator_id: str, output_dir: str) -> str:
        if operator_id not in self.sequence_graphs:
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sequence_analysis_{timestamp}_{operator_id}.txt"
        output_path = os.path.join(output_dir, filename)
        
        G = self.sequence_graphs[operator_id]
        sequence = self.operator_sequences[operator_id]
        optimal_data = self.find_optimal_path(operator_id)
        
        with open(output_path, 'w') as f:
            f.write(f"COMPREHENSIVE ACTION SEQUENCE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Operator: {operator_id}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("ASSEMBLY SEQUENCE OVERVIEW:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Actions: {len(sequence)}\n")
            f.write(f"Unique Actions: {len(G.nodes())}\n")
            f.write(f"Complete Sequence: {' -> '.join(sequence)}\n\n")
            
            f.write("ACTION FREQUENCY ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            action_counts = Counter(sequence)
            for action, count in action_counts.most_common():
                percentage = (count / len(sequence)) * 100
                estimated_time = self.action_times.get(action, 0)
                action_type = self.extract_action_components(action)['action_type']
                f.write(f"{action} [{action_type}]: {count} times ({percentage:.1f}%) - ~{estimated_time:.1f}s each\n")
            f.write("\n")
            
            f.write("TRANSITION ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            transitions = defaultdict(int)
            for i in range(len(sequence) - 1):
                transition = f"{sequence[i]} -> {sequence[i + 1]}"
                transitions[transition] += 1
            
            for transition, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True):
                source_action = transition.split(' -> ')[0]
                probability = (count / action_counts[source_action]) * 100
                f.write(f"{transition}: {count} times ({probability:.1f}% probability)\n")
            f.write("\n")
            
            if optimal_data:
                f.write("OPTIMAL PATH ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Actual Path Time: {optimal_data['actual_time']:.1f}s\n")
                f.write(f"Optimal Path Time: {optimal_data['optimal_time']:.1f}s\n")
                f.write(f"Time Efficiency: {optimal_data['time_efficiency']:.1f}%\n")
                f.write(f"Potential Time Saving: {optimal_data['path_improvement']:.1f}s\n")
                f.write(f"Optimal Sequence: {' -> '.join(optimal_data['optimal_path'])}\n\n")
            
            f.write("PREDICTION ANALYSIS AND RECOMMENDATIONS:\n")
            f.write("-" * 50 + "\n")

            correct_predictions = 0
            total_predictions = 0
            
            for i in range(2, len(sequence) - 1):
                previous_actions = sequence[i-2:i] 
                actual_next = sequence[i]
                predictions = self.predict_next_actions(operator_id, previous_actions, 5)
                
                if predictions:
                    total_predictions += 1
                    predicted_actions = [pred[0] for pred in predictions]
                    confidence_scores = [pred[1] for pred in predictions]
                    
                    if actual_next in predicted_actions:
                        correct_predictions += 1
                        status = " CORRECT"
                    else:
                        status = " INCORRECT"
                    
                    f.write(f"{status} Previous Actions: {previous_actions}: \n")
                    f.write(f"   Predicted top-3: {[(action, f'{score:.0%}') for action, score in zip(predicted_actions[:3], confidence_scores[:3])]}\n")
                    f.write(f"   Actual: [{actual_next}, timestamp: {i/len(sequence)*100:.1f}% of sequence]\n")

                    if actual_next not in predicted_actions and predictions:
                        f.write(f"   RECOMMENDATION: Consider {predicted_actions[0]} next time ({(predictions[0][1]*100):.1f}% confidence)\n")
                    f.write("\n")
            
            if total_predictions > 0:
                accuracy = (correct_predictions / total_predictions) * 100
                f.write(f"Overall Prediction Accuracy: {accuracy:.1f}%\n\n")

            f.write("STARTING ACTION RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            start_actions = [action for action in G.nodes() if G.in_degree(action) == 0]
            if start_actions:
                f.write("Recommended starting actions based on optimal paths:\n")
                for action in start_actions:
                    action_type = self.extract_action_components(action)['action_type']
                    f.write(f"  • {action} [{action_type}]\n")
            else:
                f.write("No specific starting recommendations available.\n")

            f.write("\nMOST EFFICIENT PATTERNS:\n")
            f.write("-" * 40 + "\n")
            efficient_transitions = sorted([(u, v, data['probability']) 
                                          for u, v, data in G.edges(data=True)], 
                                         key=lambda x: x[2], reverse=True)[:5]
            
            for u, v, prob in efficient_transitions:
                u_type = self.extract_action_components(u)['action_type']
                v_type = self.extract_action_components(v)['action_type']
                f.write(f"  {u} [{u_type}] -> {v} [{v_type}]: {prob:.1%} probability\n")
        
        return output_path

    def analyze_all_operators(self, output_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparative_sequence_analysis_{timestamp}.txt"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as f:
            f.write("COMPARATIVE ACTION SEQUENCE ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            efficiency_data = []
            for operator_id in self.operator_sequences.keys():
                optimal_data = self.find_optimal_path(operator_id)
                if optimal_data:
                    efficiency_data.append({
                        'operator': operator_id,
                        'efficiency': optimal_data['time_efficiency'],
                        'time_saving': optimal_data['path_improvement'],
                        'sequence_length': len(self.operator_sequences[operator_id])
                    })
            
            f.write("OPERATOR EFFICIENCY COMPARISON:\n")
            f.write("-" * 40 + "\n")
            for data in sorted(efficiency_data, key=lambda x: x['efficiency'], reverse=True):
                f.write(f"{data['operator']}: {data['efficiency']:.1f}% efficiency, ")
                f.write(f"Potential saving: {data['time_saving']:.1f}s, ")
                f.write(f"Sequence length: {data['sequence_length']}\n")
            
            f.write("\nMOST COMMON SEQUENCE PATTERNS:\n")
            f.write("-" * 40 + "\n")

            all_transitions = defaultdict(int)
            for operator_id, G in self.sequence_graphs.items():
                for u, v, data in G.edges(data=True):
                    transition = f"{u} -> {v}"
                    all_transitions[transition] += data.get('weight', 0)
            
            for transition, count in sorted(all_transitions.items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"{transition}: {count} times\n")

    def _build_assembly_dependencies(self):
        self.assembly_dependencies = {}

    def _build_assembly_workflow(self):
        self.assembly_workflow = {}
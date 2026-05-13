import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy import stats
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
from typing import Dict, List, Any, Optional

class PremAnalyzer:

    
    def __init__(self, engagement_data, hand_usage_data, progress_data, action_sequences, output_dir: str = "prem_analysis_results"):
        self.engagement_data = engagement_data
        self.hand_usage_data = hand_usage_data
        self.progress_data = progress_data
        self.action_sequences = action_sequences
        self.operators = list(action_sequences.keys())
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.primitive_categories = {
            'manipulation': ['take_', 'put_', 'align_', 'change_'],
            'fastening': ['screw_', 'tighten_', 'plug_', 'tigthen_'],
            'positioning': ['align_', 'put_', 'take_'],
            'tool_usage': ['screwdriver', 'screwdriver_bit'],
            'component_handling': ['take_bolt', 'take_nut', 'take_screw', 'take_camera', 'change_screwdriver_bit']
        }

        self.operator_colors = {
            'ID-1': '#1f77b4', 'ID-2': '#ff7f0e', 'ID-3': '#2ca02c',
            'ID-4': '#d62728', 'ID-5': '#9467bd', 'unknown_operator': '#7f7f7f'
        }

    def generate_prem_analysis(self, research_metrics: Dict) -> Dict:
        """Generate comprehensive micro-analysis with all visualizations and reports"""
        print("Starting Prem Micro-Analysis...")

        engagement_data = self._extract_engagement_data(research_metrics)
        hand_usage_data = self._extract_hand_usage_data(research_metrics)
        progress_data = self._extract_progress_data(research_metrics)

        primitive_analysis = self.extract_action_primitives()
        efficiency_analysis = self.analyze_primitive_efficiency()
        hand_usage_analysis = self.hand_usage_primitive_analysis()
        engagement_correlation = self.engagement_primitive_correlation()
        operator_clustering = self.cluster_operators_by_primitives()

        self.visualize_micro_analysis()
        self.generate_operator_wise_visualizations(primitive_analysis)
        self.generate_hrc_collaboration_analysis(research_metrics)

        csv_reports = self.export_prem_csv_reports(primitive_analysis, efficiency_analysis, hand_usage_analysis, engagement_correlation)
        insights_report = self.generate_operator_insights(primitive_analysis, efficiency_analysis)

        comprehensive_report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'title': 'Micro-analysis of Human Operator Primitives for HRC',
            'primitive_analysis': primitive_analysis,
            'efficiency_analysis': efficiency_analysis,
            'hand_usage_analysis': hand_usage_analysis,
            'engagement_correlation': engagement_correlation,
            'operator_clustering': operator_clustering,
            'summary_statistics': self._calculate_micro_summary_stats(primitive_analysis),
            'csv_reports': csv_reports,
            'operator_insights': insights_report
        }

        serializable_report = self._make_json_serializable(comprehensive_report)
        with open(f"{self.output_dir}/prem_comprehensive_report.json", 'w') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        print(f"Prem analysis completed! Results saved to {self.output_dir}")
        return comprehensive_report

    def _make_json_serializable(self, data):
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, dict):
            return {str(k): self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return str(data)

    def _extract_engagement_data(self, research_metrics: Dict) -> Dict:
        engagement_data = {}
        for operator_id, profile in research_metrics.get('operator_profiles', {}).items():
            engagement_data[operator_id] = profile.get('engagement', {})
        return engagement_data

    def _extract_hand_usage_data(self, research_metrics: Dict) -> Dict:
        hand_usage_data = {}
        for operator_id, profile in research_metrics.get('operator_profiles', {}).items():
            hand_usage_data[operator_id] = profile.get('hand_usage', {})
        return hand_usage_data

    def _extract_progress_data(self, research_metrics: Dict) -> Dict:
        progress_data = {}
        for operator_id, profile in research_metrics.get('operator_profiles', {}).items():
            progress_data[operator_id] = {
                'completion_times': profile.get('completion_times', []),
                'performance_scores': profile.get('performance_scores', [])
            }
        return progress_data

    def extract_action_primitives(self) -> Dict:
        primitive_analysis = {}
        
        for operator_id, sequence in self.action_sequences.items():
            unique_sequence = []
            prev_action = None
            for action in sequence:
                if action != prev_action:
                    unique_sequence.append(action)
                    prev_action = action
            
            primitives = {
                'raw_sequence': unique_sequence,
                'categorized_primitives': self._categorize_primitives(unique_sequence),
                'primitive_frequency': Counter(unique_sequence),
                'transition_patterns': self._analyze_primitive_transitions(unique_sequence),
                'temporal_patterns': self._analyze_temporal_distribution(unique_sequence)
            }
            primitive_analysis[operator_id] = primitives
        
        return primitive_analysis

    def _categorize_primitives(self, sequence: List[str]) -> Dict:
        categorized = {category: [] for category in self.primitive_categories.keys()}
        categorized['uncategorized'] = []
        
        for action in sequence:
            categorized_flag = False
            for category, keywords in self.primitive_categories.items():
                if any(keyword in action for keyword in keywords):
                    categorized[category].append(action)
                    categorized_flag = True
                    break
            if not categorized_flag:
                categorized['uncategorized'].append(action)
        
        return categorized

    def _analyze_primitive_transitions(self, sequence: List[str]) -> Dict:
        transitions = []
        primitive_types = []
        
        for action in sequence:
            action_type = 'uncategorized'
            for category, keywords in self.primitive_categories.items():
                if any(keyword in action for keyword in keywords):
                    action_type = category
                    break
            primitive_types.append(action_type)

        for i in range(len(primitive_types) - 1):
            transition = (primitive_types[i], primitive_types[i+1])
            transitions.append(transition)

        transitions_counter = Counter(transitions)
        transitions_str_keys = {f"{src}->{dst}": count for (src, dst), count in transitions_counter.items()}
        
        return {
            'transitions': transitions_str_keys,
            'primitive_sequence': primitive_types,
            'transition_matrix': self._create_transition_matrix(primitive_types)
        }

    def _create_transition_matrix(self, primitive_sequence: List[str]) -> np.ndarray:
        categories = list(self.primitive_categories.keys()) + ['uncategorized']
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        matrix = np.zeros((len(categories), len(categories)))
        
        for i in range(len(primitive_sequence) - 1):
            current = primitive_sequence[i]
            next_val = primitive_sequence[i+1]
            if current in cat_to_idx and next_val in cat_to_idx:
                matrix[cat_to_idx[current]][cat_to_idx[next_val]] += 1

        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)
        
        return matrix

    def _analyze_temporal_distribution(self, sequence: List[str]) -> Dict:
        action_positions = defaultdict(list)
        for idx, action in enumerate(sequence):
            action_type = 'uncategorized'
            for category, keywords in self.primitive_categories.items():
                if any(keyword in action for keyword in keywords):
                    action_type = category
                    break
            action_positions[action_type].append(idx)
        
        temporal_stats = {}
        for action_type, positions in action_positions.items():
            if positions:
                temporal_stats[action_type] = {
                    'mean_position': np.mean(positions),
                    'std_position': np.std(positions),
                    'first_occurrence': min(positions),
                    'last_occurrence': max(positions),
                    'density': len(positions) / len(sequence)
                }
        
        return temporal_stats

    def analyze_primitive_efficiency(self) -> Dict:
        efficiency_analysis = {}
        
        for operator_id, sequence in self.action_sequences.items():
            primitive_efficiency = {}
            
            for action in set(sequence):
                action_count = sequence.count(action)
                total_actions = len(sequence)
                
                primitive_efficiency[action] = {
                    'frequency': action_count,
                    'frequency_percentage': (action_count / total_actions) * 100,
                    'position_stats': self._calculate_position_stats(sequence, action),
                    'transition_efficiency': self._calculate_transition_efficiency(sequence, action)
                }
            
            efficiency_analysis[operator_id] = primitive_efficiency
        
        return efficiency_analysis

    def _calculate_position_stats(self, sequence: List[str], action: str) -> Dict:
        positions = [i for i, a in enumerate(sequence) if a == action]
        if not positions:
            return {}
        
        return {
            'mean_position': np.mean(positions),
            'std_position': np.std(positions),
            'first_occurrence': min(positions),
            'last_occurrence': max(positions)
        }

    def _calculate_transition_efficiency(self, sequence: List[str], action: str) -> float:
        positions = [i for i, a in enumerate(sequence) if a == action]
        if len(positions) < 2:
            return 1.0
        
        intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        return 1 / (np.std(intervals) + 1e-6)

    def hand_usage_primitive_analysis(self) -> Dict:
        hand_analysis = {}
        
        for operator_id in self.operators:
            primitive_hand_usage = {}
            
            for action in set(self.action_sequences[operator_id]):
                if operator_id in self.hand_usage_data:
                    hand_data = self.hand_usage_data[operator_id]
                    total_frames = sum(hand_data.values()) if hand_data else 1
                    
                    primitive_hand_usage[action] = {
                        'left_hand_ratio': hand_data.get('LEFT', 0) / total_frames,
                        'right_hand_ratio': hand_data.get('RIGHT', 0) / total_frames,
                        'bimanual_indicator': min(hand_data.get('LEFT', 0), hand_data.get('RIGHT', 0)) / total_frames,
                        'hand_preference': self._determine_hand_preference(hand_data)
                    }
            
            hand_analysis[operator_id] = primitive_hand_usage
        
        return hand_analysis

    def _determine_hand_preference(self, hand_data: Dict) -> str:
        left = hand_data.get('LEFT', 0)
        right = hand_data.get('RIGHT', 0)
        
        if left > right * 1.5:
            return 'LEFT_PREFERRED'
        elif right > left * 1.5:
            return 'RIGHT_PREFERRED'
        else:
            return 'AMBIDEXTROUS'

    def engagement_primitive_correlation(self) -> Dict:
        engagement_correlation = {}
        
        for operator_id in self.operators:
            primitive_engagement = {}
            
            for action in set(self.action_sequences[operator_id]):
                if operator_id in self.engagement_data:
                    eng_data = self.engagement_data[operator_id]
                    total_frames = sum(eng_data.values()) if eng_data else 1
                    
                    primitive_engagement[action] = {
                        'high_engagement_ratio': eng_data.get('HIGHLY_ENGAGED', 0) / total_frames,
                        'engagement_score': (eng_data.get('HIGHLY_ENGAGED', 0) * 2 + 
                                           eng_data.get('ENGAGED', 0)) / total_frames,
                        'preparation_time_ratio': eng_data.get('PREPARING', 0) / total_frames,
                        'idle_time_ratio': eng_data.get('IDLE', 0) / total_frames
                    }
            
            engagement_correlation[operator_id] = primitive_engagement
        
        return engagement_correlation

    def cluster_operators_by_primitives(self) -> Dict:
        features = []
        feature_names = []
        
        for operator_id in self.operators:
            sequence = self.action_sequences[operator_id]
            feature_vector = []

            for category in self.primitive_categories.keys():
                category_actions = [a for a in sequence if any(kw in a for kw in self.primitive_categories[category])]
                feature_vector.append(len(category_actions) / len(sequence))

            feature_vector.append(len(set(sequence)) / len(sequence))
            feature_vector.append(Counter(sequence).most_common(1)[0][1] / len(sequence))
            
            features.append(feature_vector)
        
        feature_names = list(self.primitive_categories.keys()) + ['action_diversity', 'action_concentration']
        
        if len(features) > 1:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            clustering = DBSCAN(eps=0.5, min_samples=1).fit(features_scaled)
            
            return {
                'labels': clustering.labels_.tolist(),
                'features': features_scaled.tolist(),
                'feature_names': feature_names,
                'operator_groups': {op_id: label for op_id, label in zip(self.operators, clustering.labels_)}
            }
        else:
            return {
                'labels': [0],
                'features': features,
                'feature_names': feature_names,
                'operator_groups': {self.operators[0]: 0}
            }

    def visualize_micro_analysis(self):
        try:
            primitive_data = self.extract_action_primitives()

            plt.figure(figsize=(15, 10))
            
            for i, operator_id in enumerate(self.operators):
                categorized = primitive_data[operator_id]['categorized_primitives']
                category_counts = [len(categorized[cat]) for cat in self.primitive_categories.keys()]
                plt.bar(np.arange(len(category_counts)) + (i * 0.2), 
                       category_counts, width=0.2, label=f'OP-{operator_id}',
                       color=self.operator_colors.get(operator_id, '#7f7f7f'))
            
            plt.title('Prem Analysis: Primitive Distribution by Operator', fontsize=16, fontweight='bold')
            plt.xlabel('Primitive Categories')
            plt.ylabel('Frequency')
            plt.xticks(np.arange(len(self.primitive_categories)), 
                      list(self.primitive_categories.keys()), rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/prem_primitive_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()

            self._plot_prem_transition_network(primitive_data)

            self._plot_prem_temporal_heatmap(primitive_data)

            self._plot_prem_hand_usage_analysis()

            self._plot_prem_engagement_correlation()

        except Exception as e:
            print(f"Warning: Some visualizations failed - {e}")
            

    def _plot_prem_transition_network(self, primitive_data: Dict):
        G = nx.DiGraph()
        
        for category in self.primitive_categories.keys():
            G.add_node(category)

        for operator_id in self.operators:
            transitions = primitive_data[operator_id]['transition_patterns']['transitions']
            for transition_key, count in transitions.items():
                if '->' in transition_key:
                    src, dst = transition_key.split('->')
                else:
                    continue
                
                if G.has_edge(src, dst):
                    G[src][dst]['weight'] += count
                else:
                    G.add_edge(src, dst, weight=count)

        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        pos = nx.spring_layout(G, seed=42)
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, edge_color=weights, width=3, 
                edge_cmap=plt.cm.Blues, arrows=True, arrowsize=20)
        plt.title('Prem Analysis: Primitive Transition Network', fontsize=16, fontweight='bold')

        plt.savefig(f"{self.output_dir}/prem_transition_network.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prem_temporal_heatmap(self, primitive_data: Dict):
        temporal_data = []
        
        for operator_id in self.operators:
            temp_patterns = primitive_data[operator_id]['temporal_patterns']
            row = []
            for category in self.primitive_categories.keys():
                if category in temp_patterns:
                    row.append(temp_patterns[category]['mean_position'])
                else:
                    row.append(0)
            temporal_data.append(row)

        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        plt.imshow(temporal_data, cmap='viridis', aspect='auto')
        plt.colorbar(label='Mean Temporal Position')
        plt.title('Prem Analysis: Temporal Distribution Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Primitive Categories')
        plt.ylabel('Operators')
        plt.xticks(np.arange(len(self.primitive_categories)), list(self.primitive_categories.keys()), rotation=45)
        plt.yticks(np.arange(len(self.operators)), self.operators)

        plt.savefig(f"{self.output_dir}/prem_temporal_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prem_hand_usage_analysis(self):
        hand_analysis = self.hand_usage_primitive_analysis()
        
        plt.figure(figsize=(14, 8))
        
        primitive_types = list(self.primitive_categories.keys())
        left_hand_ratios = []
        
        for category in primitive_types:
            category_ratio = []
            for operator_id in self.operators:
                for action, data in hand_analysis.get(operator_id, {}).items():
                    if any(kw in action for kw in self.primitive_categories[category]):
                        category_ratio.append(data['left_hand_ratio'])
            left_hand_ratios.append(np.mean(category_ratio) if category_ratio else 0)
        
        plt.bar(primitive_types, left_hand_ratios, color='skyblue', alpha=0.8)
        plt.title('Prem Analysis: Left Hand Usage by Primitive Type', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel('Left Hand Ratio')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/prem_hand_usage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prem_engagement_correlation(self):
        engagement_data = self.engagement_primitive_correlation()
        
        plt.figure(figsize=(12, 8))
        
        complexity_scores = []
        engagement_scores = []
        
        for operator_id in self.operators:
            for action, data in engagement_data.get(operator_id, {}).items():
                complexity = len(action.split('_'))
                engagement_scores.append(data['engagement_score'])
                complexity_scores.append(complexity)
        
        plt.scatter(complexity_scores, engagement_scores, alpha=0.6, color='coral')
        plt.title('Prem Analysis: Engagement vs Primitive Complexity', fontsize=16, fontweight='bold')
        plt.xlabel('Primitive Complexity (Word Count)')
        plt.ylabel('Engagement Score')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/prem_engagement_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_operator_wise_visualizations(self, primitive_analysis: Dict):
        for operator_id in self.operators:
            plt.figure(figsize=(10, 6))
            categorized = primitive_analysis[operator_id]['categorized_primitives']
            categories = list(self.primitive_categories.keys())
            counts = [len(categorized[cat]) for cat in categories]
            
            plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
            plt.title(f'Prem Analysis: Primitive Distribution - Operator {operator_id}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/prem_operator_{operator_id}_primitives.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def generate_hrc_collaboration_analysis(self, research_metrics: Dict):
        if not research_metrics.get('robot_data'):
            return
        
        robot_data = research_metrics['robot_data']
        if not robot_data:
            return
        
        collaboration_by_primitive = defaultdict(list)
        
        for record in robot_data:
            action = record.get('action_name', '')
            collaboration_score = record.get('collaboration_score', 0)

            primitive_type = 'uncategorized'
            for category, keywords in self.primitive_categories.items():
                if any(keyword in action for keyword in keywords):
                    primitive_type = category
                    break
            
            collaboration_by_primitive[primitive_type].append(collaboration_score)
        plt.figure(figsize=(12, 8))
        primitive_types = list(collaboration_by_primitive.keys())
        avg_collaboration = [np.mean(scores) for scores in collaboration_by_primitive.values()]
        
        plt.bar(primitive_types, avg_collaboration, color='lightgreen', alpha=0.8)
        plt.title('Prem Analysis: HRC Collaboration by Primitive Type', fontsize=16, fontweight='bold')
        plt.xlabel('Primitive Type')
        plt.ylabel('Average Collaboration Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/prem_hrc_collaboration.png", dpi=300, bbox_inches='tight')
        plt.close()

    def export_prem_csv_reports(self, primitive_analysis: Dict, efficiency_analysis: Dict, hand_usage_analysis: Dict, engagement_correlation: Dict) -> Dict:

        csv_files = {}

        prem_data = []
        for operator_id, analysis in primitive_analysis.items():
            for category, actions in analysis['categorized_primitives'].items():
                prem_data.append({
                    'operator_id': operator_id,
                    'primitive_category': category,
                    'action_count': len(actions),
                    'actions': '; '.join(actions)
                })
        
        prem_df = pd.DataFrame(prem_data)
        prem_df.to_csv(f"{self.output_dir}/prem_primitive_analysis.csv", index=False)
        csv_files['primitive_analysis'] = f"{self.output_dir}/prem_primitive_analysis.csv"

        eff_data = []
        for operator_id, analysis in efficiency_analysis.items():
            for action, metrics in analysis.items():
                eff_data.append({
                    'operator_id': operator_id,
                    'action': action,
                    'frequency': metrics['frequency'],
                    'frequency_percentage': metrics['frequency_percentage'],
                    'mean_position': metrics['position_stats'].get('mean_position', 0),
                    'transition_efficiency': metrics['transition_efficiency']
                })
        
        eff_df = pd.DataFrame(eff_data)
        eff_df.to_csv(f"{self.output_dir}/prem_efficiency_analysis.csv", index=False)
        csv_files['efficiency_analysis'] = f"{self.output_dir}/prem_efficiency_analysis.csv"

        hand_data = []
        for operator_id, analysis in hand_usage_analysis.items():
            for action, metrics in analysis.items():
                hand_data.append({
                    'operator_id': operator_id,
                    'action': action,
                    'left_hand_ratio': metrics['left_hand_ratio'],
                    'right_hand_ratio': metrics['right_hand_ratio'],
                    'bimanual_indicator': metrics['bimanual_indicator'],
                    'hand_preference': metrics['hand_preference']
                })
        
        hand_df = pd.DataFrame(hand_data)
        hand_df.to_csv(f"{self.output_dir}/prem_hand_usage_analysis.csv", index=False)
        csv_files['hand_usage_analysis'] = f"{self.output_dir}/prem_hand_usage_analysis.csv"

        eng_data = []
        for operator_id, analysis in engagement_correlation.items():
            for action, metrics in analysis.items():
                eng_data.append({
                    'operator_id': operator_id,
                    'action': action,
                    'high_engagement_ratio': metrics['high_engagement_ratio'],
                    'engagement_score': metrics['engagement_score'],
                    'preparation_time_ratio': metrics['preparation_time_ratio'],
                    'idle_time_ratio': metrics['idle_time_ratio']
                })
        
        eng_df = pd.DataFrame(eng_data)
        eng_df.to_csv(f"{self.output_dir}/prem_engagement_correlation.csv", index=False)
        csv_files['engagement_correlation'] = f"{self.output_dir}/prem_engagement_correlation.csv"
        
        return csv_files

    def generate_operator_insights(self, primitive_analysis: Dict, efficiency_analysis: Dict) -> Dict:
        insights = {}
        
        for operator_id in self.operators:
            operator_insights = {
                'primitive_usage': {},
                'efficiency_metrics': {},
                'recommendations': []
            }

            primitives = primitive_analysis[operator_id]['categorized_primitives']
            total_actions = sum(len(actions) for actions in primitives.values())
            
            for category, actions in primitives.items():
                percentage = (len(actions) / total_actions) * 100 if total_actions > 0 else 0
                operator_insights['primitive_usage'][category] = {
                    'count': len(actions),
                    'percentage': percentage
                }

            efficiency = efficiency_analysis.get(operator_id, {})
            if efficiency:
                avg_efficiency = np.mean([metrics['transition_efficiency'] for metrics in efficiency.values()])
                operator_insights['efficiency_metrics']['average_transition_efficiency'] = avg_efficiency

            recommendations = self._generate_operator_recommendations(operator_insights)
            operator_insights['recommendations'] = recommendations
            
            insights[operator_id] = operator_insights

        insights_data = []
        for operator_id, insight in insights.items():
            for category, usage in insight['primitive_usage'].items():
                insights_data.append({
                    'operator_id': operator_id,
                    'category': category,
                    'action_count': usage['count'],
                    'percentage': usage['percentage'],
                    'recommendations': '; '.join(insight['recommendations'])
                })
        
        insights_df = pd.DataFrame(insights_data)
        insights_df.to_csv(f"{self.output_dir}/prem_operator_insights.csv", index=False)
        
        return insights

    def _generate_operator_recommendations(self, insights: Dict) -> List[str]:
        recommendations = []
        
        primitive_usage = insights['primitive_usage']
        efficiency = insights['efficiency_metrics']

        if primitive_usage.get('fastening', {}).get('percentage', 0) > 40:
            recommendations.append("Consider balancing fastening tasks with other primitive types")
        
        if primitive_usage.get('manipulation', {}).get('percentage', 0) < 20:
            recommendations.append("Increase manipulation activities for better task variety")

        if efficiency.get('average_transition_efficiency', 1) < 0.8:
            recommendations.append("Work on smoother transitions between actions")
        
        if len(recommendations) == 0:
            recommendations.append("Maintain current efficient work patterns")
        
        return recommendations

    def _calculate_micro_summary_stats(self, primitive_analysis: Dict) -> Dict:
        stats = {
            'total_primitives_analyzed': sum(len(seq) for seq in self.action_sequences.values()),
            'unique_action_primitives': len(set().union(*[set(seq) for seq in self.action_sequences.values()])),
            'average_sequence_length': np.mean([len(seq) for seq in self.action_sequences.values()]),
            'primitive_category_distribution': {},
            'operator_variability': {}
        }

        for category in self.primitive_categories.keys():
            category_count = 0
            for operator_id in self.operators:
                category_count += len(primitive_analysis[operator_id]['categorized_primitives'][category])
            stats['primitive_category_distribution'][category] = category_count
        
        return stats

if __name__ == "__main__":
    print("Prem Analyzer Module Loaded Successfully")
    print("This module provides micro-analysis of human operator primitives")
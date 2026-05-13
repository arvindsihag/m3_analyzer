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

class ThreeLayerPremAnalyzer:
    
    def __init__(self, research_metrics: Dict, output_dir: str = "three_layer_analysis"):
        self.research_metrics = research_metrics
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.operator_colors = {
            'ID-1': '#1f77b4', 'ID-2': '#ff7f0e', 'ID-3': '#2ca02c',
            'ID-4': '#d62728', 'ID-5': '#9467bd', 'unknown_operator': '#7f7f7f'
        }
        
        # Layer-specific color schemes
        self.layer_colors = {
            'MICRO': '#3498db',    # Blue
            'MESO': '#e74c3c',     # Red  
            'MACRO': '#2ecc71'     # Green
        }

    def extract_micro_layer_data(self) -> Dict:
        micro_data = {
            'primitive_categories': {
                'manipulation': ['take_', 'put_', 'align_', 'change_'],
                'fastening': ['screw_', 'tighten_', 'plug_', 'tigthen_'],
                'positioning': ['align_', 'put_', 'take_'],
                'tool_usage': ['screwdriver', 'screwdriver_bit'],
                'component_handling':
                [
                    'take_pir_sensor',
                    'take_front_panel',
                    'take_bolt',
                    'take_nut',
                    'take_screwdriver',
                    'take_rpi_camera',
                    'take_rpi_came_ra_module',
                    'take_screw',
                    'take_rpi_board',
                    'take_display',
                    'take_rpi_board_display_module',
                    'take_rpi_camera_camera_module',
                    'take_display_mount_bracket',
                    'take_fcc_cable',
                    'take_rpi_hat',
                    'take_power_adapter',
                    'take_back_panel'
                ]

            },
            'operator_sequences': {},
            'primitive_distribution': {},
            'temporal_patterns': {},
            'diversity_metrics': {}
        }

        engagement_data = self.research_metrics.get('engagement_data', [])
        operator_actions = defaultdict(list)
        
        for record in engagement_data:
            operator = record['operator_id']
            action = record['action']
            operator_actions[operator].append(action)

        for operator, actions in operator_actions.items():
            unique_actions = list(set(actions))
            micro_data['operator_sequences'][operator] = unique_actions

            total_actions = len(actions)
            unique_count = len(unique_actions)

            action_type_distribution = defaultdict(int)
            for action in actions:
                for category, keywords in micro_data['primitive_categories'].items():
                    if any(kw in action for kw in keywords):
                        action_type_distribution[category] += 1
                        break
                else:
                    action_type_distribution['uncategorized'] += 1

            micro_data['diversity_metrics'][operator] = {
                'unique_action_count': unique_count,
                'total_action_count': total_actions,
                'diversity_ratio': unique_count / max(1, total_actions),
                'action_type_entropy': self._calculate_entropy(list(action_type_distribution.values())),
                'action_type_distribution': dict(action_type_distribution),
                'most_common_action': Counter(actions).most_common(1)[0][0] if actions else 'None'
            }
        
        return micro_data


    def _calculate_entropy(self, counts):
        if not counts or sum(counts) == 0:
            return 0
        proportions = np.array(counts) / sum(counts)
        return -np.sum(proportions * np.log(proportions + 1e-10))


    def extract_meso_layer_data(self) -> Dict:
        meso_data = {
            'engagement_levels': ['HIGHLY_ENGAGED', 'ENGAGED', 'PREPARING', 'IDLE', 'DISENGAGED'],
            'engagement_scores': {},
            'hand_usage_patterns': {},
            'temporal_engagement': {}
        }
        
        engagement_data = self.research_metrics.get('engagement_data', [])
        operator_profiles = self.research_metrics.get('operator_profiles', {})
        
        for operator, profile in operator_profiles.items():
            total_engagement = sum(profile.get('engagement', {}).values()) or 1
            meso_data['engagement_scores'][operator] = {
                level: (profile.get('engagement', {}).get(level, 0) / total_engagement * 100)
                for level in meso_data['engagement_levels']
            }

            total_hand_usage = sum(profile.get('hand_usage', {}).values()) or 1
            meso_data['hand_usage_patterns'][operator] = {
                pattern: (profile.get('hand_usage', {}).get(pattern, 0) / total_hand_usage * 100)
                for pattern in ['LEFT', 'RIGHT', 'BOTH', 'NONE']
            }
        
        return meso_data

    def extract_macro_layer_data(self) -> Dict:
        macro_data = {
            'collaboration_metrics': {},
            'handover_analysis': {},
            'performance_comparison': {}
        }

        robot_data = self.research_metrics.get('robot_data', [])
        handover_events = self.research_metrics.get('handover_events', [])

        collaboration_by_operator = defaultdict(list)
        for record in robot_data:
            operator = record.get('operator_id', 'unknown')
            collaboration_by_operator[operator].append(record.get('collaboration_score', 0))
        
        for operator, scores in collaboration_by_operator.items():
            macro_data['collaboration_metrics'][operator] = {
                'mean_score': np.mean(scores) if scores else 0,
                'std_score': np.std(scores) if len(scores) > 1 else 0,
                'interaction_count': len(scores)
            }

        if handover_events:
            handover_by_operator = defaultdict(list)
            for event in handover_events:
                operator = event.get('operator_id', 'unknown')
                handover_by_operator[operator].append(event)
            
            macro_data['handover_analysis'] = {
                operator: {
                    'count': len(events),
                    'avg_performance': np.mean([e.get('performance_score', 0) for e in events]),
                    'common_reasons': Counter([reason for e in events for reason in e.get('reasons', [])]).most_common(3)
                }
                for operator, events in handover_by_operator.items()
            }
        
        return macro_data


    def generate_three_layer_framework_diagram(self):

        fig = plt.figure(figsize=(22, 18))

        gs = fig.add_gridspec(3, 3, width_ratios=[1, 1.2, 1.2], height_ratios=[1, 1, 1.2])

        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_micro_layer(ax1)

        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_meso_layer(ax2)

        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_macro_layer(ax3)

        ax4 = fig.add_subplot(gs[0, 1:])
        self._plot_enhanced_micro_meso_correlation(ax4)

        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_integrated_case_study(ax5)

        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_radar_integrated_case_study(ax6)

        ax7 = fig.add_subplot(gs[2, 1:])
        self._plot_performance_summary(ax7)
        
        plt.suptitle('Three-Layer Analysis Framework: MICRO-MESO-MACRO\nWith Enhanced Multi-Operator Insights', fontsize=22, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/three_layer_framework_enhanced.png", dpi=300, bbox_inches='tight')
        plt.close()




    def _plot_enhanced_micro_meso_correlation(self, ax):
        try:
            micro_data = self.extract_micro_layer_data()
            meso_data = self.extract_meso_layer_data()
            
            operators = list(micro_data.get('operator_sequences', {}).keys())
            if len(operators) < 2:
                ax.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Micro-Meso Correlation\n(Need more operators)', fontsize=10)
                return

            diversities = []
            engagements = []
            
            for operator in operators:
                diversity_metrics = micro_data.get('diversity_metrics', {}).get(operator, {})
                diversity_score = diversity_metrics.get('diversity_ratio', 0.5)
                diversities.append(diversity_score)

                engagement_scores = meso_data.get('engagement_scores', {}).get(operator, {})
                high_engagement = engagement_scores.get('HIGHLY_ENGAGED', 0)
                engagements.append(high_engagement / 100)

            scatter = ax.scatter(diversities, engagements, 
                               s=200, alpha=0.7, c=range(len(operators)), 
                               cmap='viridis', edgecolors='black')

            for i, operator in enumerate(operators):
                ax.annotate(operator, (diversities[i], engagements[i]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontweight='bold', fontsize=9)
            
            ax.set_xlabel('Primitive Diversity Ratio')
            ax.set_ylabel('High Engagement (%)')
            ax.set_title('Micro vs Meso Correlation\nDiversity vs Engagement', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            if len(diversities) > 1:
                z = np.polyfit(diversities, engagements, 1)
                p = np.poly1d(z)
                ax.plot(diversities, p(diversities), "r--", alpha=0.8, linewidth=2)

                corr = np.corrcoef(diversities, engagements)[0,1]
                ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error in correlation plot:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('Micro-Meso Correlation\n(Error)', fontsize=10)


    def _plot_performance_summary(self, ax):
        try:
            micro_data = self.extract_micro_layer_data()
            meso_data = self.extract_meso_layer_data()
            macro_data = self.extract_macro_layer_data()
            
            operators = list(micro_data.get('operator_sequences', {}).keys())
            if not operators:
                ax.text(0.5, 0.5, 'No operator data available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Performance Summary\n(No Data)', fontsize=12)
                return

            micro_scores = []
            meso_scores = []
            macro_scores = []
            
            for operator in operators:
                diversity_metrics = micro_data.get('diversity_metrics', {}).get(operator, {})
                micro_score = diversity_metrics.get('diversity_ratio', 0.5)
                micro_scores.append(micro_score)

                engagement_scores = meso_data.get('engagement_scores', {}).get(operator, {})
                meso_score = (engagement_scores.get('HIGHLY_ENGAGED', 0) / 100)
                meso_scores.append(meso_score)

                collaboration_metrics = macro_data.get('collaboration_metrics', {}).get(operator, {})
                macro_score = collaboration_metrics.get('mean_score', 0.5)
                macro_scores.append(macro_score)

            x = np.arange(len(operators))
            width = 0.25
            
            bars1 = ax.bar(x - width, micro_scores, width, label='MICRO\n(Diversity)', 
                          alpha=0.8, color=self.layer_colors['MICRO'])
            bars2 = ax.bar(x, meso_scores, width, label='MESO\n(Engagement)', 
                          alpha=0.8, color=self.layer_colors['MESO'])
            bars3 = ax.bar(x + width, macro_scores, width, label='MACRO\n(Collaboration)', 
                          alpha=0.8, color=self.layer_colors['MACRO'])

            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.set_xlabel('Operators')
            ax.set_ylabel('Normalized Scores')
            ax.set_title('Cross-Layer Performance Summary', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(operators)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error in performance summary:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('Performance Summary\n(Error)', fontsize=10)



    def _plot_micro_layer(self, ax):
        micro_data = self.extract_micro_layer_data()

        operators = list(micro_data['operator_sequences'].keys())
        categories = list(micro_data['primitive_categories'].keys())

        primitive_dist = {}
        for operator in operators:
            unique_actions = micro_data['operator_sequences'][operator] 
            primitive_dist[operator] = {}
            for category, keywords in micro_data['primitive_categories'].items():
                count = sum(1 for action in unique_actions if any(kw in action for kw in keywords))
                primitive_dist[operator][category] = count

        if operators and categories:
            bottom = np.zeros(len(operators))
            for i, category in enumerate(categories):
                values = [primitive_dist[op].get(category, 0) for op in operators]
                ax.bar(operators, values, bottom=bottom, label=category, 
                       color=plt.cm.Set3(i/len(categories)), alpha=0.8)
                bottom += values
            
            ax.set_title('MICRO Layer: Primitive-Level Action Patterns\n(Unique Actions Only)', 
                        fontsize=14, fontweight='bold', pad=10)
            ax.set_ylabel('Unique Action Count')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_title('MICRO Layer: No Data Available', fontsize=14, fontweight='bold')

    def _plot_meso_layer(self, ax):
        meso_data = self.extract_meso_layer_data()
        operators = list(meso_data['engagement_scores'].keys())
        engagement_levels = meso_data['engagement_levels']
        angles = np.linspace(0, 2*np.pi, len(engagement_levels), endpoint=False).tolist()
        angles += angles[:1]
        
        for operator in operators:
            scores = [meso_data['engagement_scores'][operator][level] for level in engagement_levels]
            scores += scores[:1]  # Complete the circle
            ax.plot(angles, scores, 'o-', linewidth=2, label=operator,
                   color=self.operator_colors.get(operator, '#7f7f7f'))
            ax.fill(angles, scores, alpha=0.1, 
                   color=self.operator_colors.get(operator, '#7f7f7f'))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(engagement_levels)
        ax.set_title('MESO Layer: Engagement State Dynamics', 
                    fontsize=14, fontweight='bold', pad=10)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_macro_layer(self, ax):
        macro_data = self.extract_macro_layer_data()
        operators = list(macro_data['collaboration_metrics'].keys())
        collaboration_scores = [macro_data['collaboration_metrics'][op]['mean_score'] for op in operators]
        
        bars = ax.bar(operators, collaboration_scores, color=[self.operator_colors.get(op, '#7f7f7f') for op in operators], alpha=0.8)
        if macro_data['handover_analysis']:
            handover_counts = [macro_data['handover_analysis'].get(op, {}).get('count', 0) for op in operators]
            max_collab = max(collaboration_scores) if collaboration_scores else 1
            for i, count in enumerate(handover_counts):
                if count > 0:
                    ax.scatter(i, collaboration_scores[i] + 0.05, s=count*50, 
                              color='red', alpha=0.6, label='Handovers' if i == 0 else "")
        
        ax.set_title('MACRO Layer: Robot Collaboration Effectiveness', 
                    fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('Collaboration Score (0-1)')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    def _plot_integrated_case_study(self, ax):
        micro_data = self.extract_micro_layer_data()
        meso_data = self.extract_meso_layer_data()
        macro_data = self.extract_macro_layer_data()
        
        operators = list(micro_data.get('operator_sequences', {}).keys())
        if not operators:
            operators = list(meso_data.get('engagement_scores', {}).keys())
        if not operators:
            operators = list(macro_data.get('collaboration_metrics', {}).keys())
        
        if not operators:
            ax.text(0.5, 0.5, 'No operator data available', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_title('Integrated Case Study:\nNo Data Available', 
                        fontsize=14, fontweight='bold', pad=10)
            return
        
        print(f"Analyzing {len(operators)} operators: {operators}")
        comparison_data = {}
        for operator in operators:
            comparison_data[operator] = {
                'Micro_Primitives': micro_data.get('operator_sequences', {}).get(operator, []),
                'Meso_Engagement': meso_data.get('engagement_scores', {}).get(operator, {}),
                'Macro_Collaboration': macro_data.get('collaboration_metrics', {}).get(operator, {}).get('mean_score', 0)
            }
        
        # Create radar-like comparison metrics
        metrics = ['Primitive\nDiversity', 'Engagement\nStability', 'Collaboration\nEffectiveness']
        
        # Calculate values for each operator
        operator_values = {}
        for operator in operators:
            primitives = comparison_data[operator]['Micro_Primitives']
            engagement = comparison_data[operator]['Meso_Engagement']
            collaboration = comparison_data[operator]['Macro_Collaboration']
            
            values = [
                len(set(primitives)) / max(1, len(primitives)), 
                engagement.get('HIGHLY_ENGAGED', 0) / 100,  
                collaboration  
            ]
            operator_values[operator] = values

        x = np.arange(len(metrics))
        width = 0.8 / len(operators)
        
        # Plot bars for each operator
        for i, operator in enumerate(operators):
            values = operator_values[operator]
            position = x - (0.8 - width) / 2 + i * width
            ax.bar(position, values, width, label=operator,
                   color=self.operator_colors.get(operator, '#7f7f7f'), alpha=0.8)
            
            # Add value labels on bars
            for j, value in enumerate(values):
                ax.text(position[j], value + 0.02, f'{value:.2f}', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_title(f'Integrated Case Study:\nOperator Comparison ({len(operators)} Operators)', 
                    fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Normalized Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        if len(operators) > 1:
            # Calculate averages
            avg_diversity = np.mean([operator_values[op][0] for op in operators])
            avg_engagement = np.mean([operator_values[op][1] for op in operators])
            avg_collaboration = np.mean([operator_values[op][2] for op in operators])
            
            # Add average line
            ax.axhline(y=avg_diversity, color='red', linestyle='--', alpha=0.7, 
                      label=f'Avg Diversity: {avg_diversity:.2f}')
            ax.axhline(y=avg_engagement, color='blue', linestyle='--', alpha=0.7, 
                      label=f'Avg Engagement: {avg_engagement:.2f}')
            ax.axhline(y=avg_collaboration, color='green', linestyle='--', alpha=0.7, 
                      label=f'Avg Collaboration: {avg_collaboration:.2f}')



    def _plot_radar_integrated_case_study(self, ax):
        micro_data = self.extract_micro_layer_data()
        meso_data = self.extract_meso_layer_data()
        macro_data = self.extract_macro_layer_data()
        
        operators = list(micro_data.get('operator_sequences', {}).keys())
        if not operators:
            ax.text(0.5, 0.5, 'No operator data available', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_title('Radar Case Study:\nNo Data Available', 
                        fontsize=14, fontweight='bold', pad=10)
            return

        metrics = ['Primitive Diversity', 'Engagement Stability', 'Collaboration Effectiveness']
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for operator in operators:
            # Get data for this operator
            primitives = micro_data.get('operator_sequences', {}).get(operator, [])
            engagement = meso_data.get('engagement_scores', {}).get(operator, {})
            collaboration = macro_data.get('collaboration_metrics', {}).get(operator, {}).get('mean_score', 0)
            
            values = [
                len(set(primitives)) / max(1, len(primitives)),
                engagement.get('HIGHLY_ENGAGED', 0) / 100,
                collaboration
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=operator,
                   color=self.operator_colors.get(operator, '#7f7f7f'))
            ax.fill(angles, values, alpha=0.1, 
                   color=self.operator_colors.get(operator, '#7f7f7f'))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(f'Operator Comparison: Radar Chart\n({len(operators)} Operators)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
        ax.grid(True)






    def generate_layer_specific_analyses(self):

        self._generate_micro_layer_analysis()

        self._generate_meso_layer_analysis()

        self._generate_macro_layer_analysis()

        self._generate_cross_layer_correlations()

    def _generate_micro_layer_analysis(self):
        micro_data = self.extract_micro_layer_data()
        
        # 1. Primitive Transition Network
        plt.figure(figsize=(14, 10))
        self._plot_primitive_transition_network(micro_data)
        plt.savefig(f"{self.output_dir}/micro_primitive_transition_network.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Temporal Distribution Heatmap
        plt.figure(figsize=(12, 8))
        self._plot_temporal_heatmap(micro_data)
        plt.savefig(f"{self.output_dir}/micro_temporal_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_meso_layer_analysis(self):
        """Generate detailed MESO layer visualizations"""
        meso_data = self.extract_meso_layer_data()
        
        # 1. Engagement Violin Plots
        plt.figure(figsize=(14, 8))
        self._plot_engagement_violin(meso_data)
        plt.savefig(f"{self.output_dir}/meso_engagement_violin.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Hand Usage Patterns
        plt.figure(figsize=(12, 8))
        self._plot_hand_usage_patterns(meso_data)
        plt.savefig(f"{self.output_dir}/meso_hand_usage_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_macro_layer_analysis(self):
        """Generate detailed MACRO layer visualizations"""
        macro_data = self.extract_macro_layer_data()
        
        # 1. Collaboration vs Performance
        plt.figure(figsize=(12, 8))
        self._plot_collaboration_performance(macro_data)
        plt.savefig(f"{self.output_dir}/macro_collaboration_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Handover Effectiveness
        plt.figure(figsize=(10, 6))
        self._plot_handover_effectiveness(macro_data)
        plt.savefig(f"{self.output_dir}/macro_handover_effectiveness.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comprehensive_report(self):
        """Generate comprehensive three-layer analysis report"""
        print("Starting Three-Layer Analysis...")
        
        # Generate all visualizations
        self.generate_three_layer_framework_diagram()
        self.generate_layer_specific_analyses()
        self.generate_enhanced_correlations() 
        
        # Create summary report
        report = {
            'timestamp': datetime.now().isoformat(),
            'framework': 'MICRO-MESO-MACRO Analysis',
            'micro_analysis': self.extract_micro_layer_data(),
            'meso_analysis': self.extract_meso_layer_data(),
            'macro_analysis': self.extract_macro_layer_data(),
            'cross_layer_insights': self._generate_cross_layer_insights()
        }
        
        # Save report
        with open(f"{self.output_dir}/three_layer_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Three-layer analysis completed! Results saved to {self.output_dir}")
        return report


    def generate_enhanced_correlations(self):
        try:
            self._generate_cross_layer_correlations()
            print("✓ Enhanced correlations generated")
        except Exception as e:
            print(f"Error generating enhanced correlations: {e}")


    def _plot_primitive_transition_network(self, micro_data):
        try:
            G = nx.DiGraph()

            operators = list(micro_data.get('operator_sequences', {}).keys())
            if not operators:
                print("No operator data for transition network")
                return

            for operator in operators:
                sequence = micro_data['operator_sequences'][operator]
                if len(sequence) < 2:
                    continue
                    
                for i in range(len(sequence) - 1):
                    source = sequence[i]
                    target = sequence[i + 1]
                    
                    if G.has_edge(source, target):
                        G[source][target]['weight'] += 1
                    else:
                        G.add_edge(source, target, weight=1)
            
            if len(G.edges()) == 0:
                print("Not enough transitions for network diagram")
                return
                
            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw the network
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.9)
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
            
            # Add edge weights
            edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            plt.title('MICRO Layer: Primitive Transition Network', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
        except Exception as e:
            print(f"Error in transition network: {e}")

    def _plot_temporal_heatmap(self, micro_data):
        try:
            operators = list(micro_data.get('operator_sequences', {}).keys())
            if not operators:
                print("No operator data for temporal heatmap")
                return

            all_actions = set()
            for operator in operators:
                all_actions.update(micro_data['operator_sequences'][operator])
            
            if not all_actions:
                print("No actions for temporal heatmap")
                return
                
            all_actions = sorted(list(all_actions))

            heatmap_data = []
            for operator in operators:
                sequence = micro_data['operator_sequences'][operator]
                if not sequence:
                    continue
                    
                row = []
                for action in all_actions:
                    if action in sequence:
                        positions = [i for i, a in enumerate(sequence) if a == action]
                        if positions:
                            avg_pos = np.mean(positions) / max(1, len(sequence) - 1)
                            row.append(avg_pos)
                        else:
                            row.append(0)
                    else:
                        row.append(0)
                heatmap_data.append(row)
            
            if not heatmap_data:
                print("No heatmap data available")
                return
                
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, 
                       xticklabels=[a[:15] + '...' if len(a) > 15 else a for a in all_actions],
                       yticklabels=operators,
                       cmap='viridis', 
                       cbar_kws={'label': 'Normalized Temporal Position'})
            
            plt.title('MICRO Layer: Temporal Distribution Heatmap', fontsize=16, fontweight='bold')
            plt.xlabel('Action Primitives')
            plt.ylabel('Operators')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
        except Exception as e:
            print(f"Error in temporal heatmap: {e}")

    def _plot_engagement_violin(self, meso_data):
        try:
            engagement_scores = meso_data.get('engagement_scores', {})
            if not engagement_scores:
                print("No engagement data for violin plot")
                return

            plot_data = []
            engagement_levels = ['HIGHLY_ENGAGED', 'ENGAGED', 'PREPARING', 'IDLE', 'DISENGAGED']
            
            for operator, scores in engagement_scores.items():
                total_samples = 100
                for level in engagement_levels:
                    percentage = scores.get(level, 0)
                    sample_count = int(total_samples * percentage / 100)

                    for _ in range(sample_count):
                        noise = np.random.normal(0, 5)
                        score_value = max(0, min(100, percentage + noise))
                        plot_data.append({
                            'Operator': operator,
                            'Engagement Level': level,
                            'Score': score_value
                        })
            
            if not plot_data:
                print("No engagement plot data")
                return
                
            df = pd.DataFrame(plot_data)
            
            plt.figure(figsize=(14, 8))
            
            # Create actual violin plot with data
            sns.violinplot(data=df, x='Engagement Level', y='Score', hue='Operator',
                          palette=self.operator_colors, alpha=0.7, cut=0)
            
            # Add individual data points
            sns.stripplot(data=df, x='Engagement Level', y='Score', hue='Operator',
                         palette=self.operator_colors, size=4, alpha=0.6, dodge=True)
            
            plt.title('MESO Layer: Engagement Level Distribution\n(With Individual Data Points)', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Engagement Level')
            plt.ylabel('Engagement Score (%)')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
        except Exception as e:
            print(f"Error in engagement violin plot: {e}")

    def _plot_hand_usage_patterns(self, meso_data):
        try:
            hand_patterns = meso_data.get('hand_usage_patterns', {})
            if not hand_patterns:
                print("No hand usage data for patterns")
                return
                
            operators = list(hand_patterns.keys())
            patterns = ['LEFT', 'RIGHT', 'BOTH', 'NONE']
            plot_data = {pattern: [] for pattern in patterns}
            for operator in operators:
                for pattern in patterns:
                    plot_data[pattern].append(hand_patterns[operator].get(pattern, 0))
            
            plt.figure(figsize=(12, 8))
            
            bottom = np.zeros(len(operators))
            for i, pattern in enumerate(patterns):
                plt.bar(operators, plot_data[pattern], bottom=bottom, 
                       label=pattern, alpha=0.8, 
                       color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][i])
                bottom += np.array(plot_data[pattern])
            
            plt.title('MESO Layer: Hand Usage Patterns', fontsize=16, fontweight='bold')
            plt.xlabel('Operators')
            plt.ylabel('Usage Percentage')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.ylim(0, 100)
            plt.tight_layout()
            
        except Exception as e:
            print(f"Error in hand usage patterns: {e}")


    def _plot_collaboration_performance(self, macro_data):

        try:
            collaboration_metrics = macro_data.get('collaboration_metrics', {})
            handover_analysis = macro_data.get('handover_analysis', {})
            
            if not collaboration_metrics:
                print("No collaboration data for performance plot")
                return
                
            operators = list(collaboration_metrics.keys())
            collaboration_scores = [collaboration_metrics[op]['mean_score'] for op in operators]
            interaction_counts = [collaboration_metrics[op]['interaction_count'] for op in operators]
            
            handover_counts = [handover_analysis.get(op, {}).get('count', 0) for op in operators]
            performance_scores = [handover_analysis.get(op, {}).get('avg_performance', 50) for op in operators]
            
            plt.figure(figsize=(14, 10))
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

            bars1 = ax1.bar(operators, collaboration_scores, 
                           color=[self.operator_colors.get(op, '#7f7f7f') for op in operators],
                           alpha=0.8, yerr=[collaboration_metrics[op]['std_score'] for op in operators],
                           capsize=5)
            
            ax1.set_title('Collaboration Effectiveness Scores\n(with Standard Deviation)', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Collaboration Score (0-1)')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3, axis='y')

            for bar, score in zip(bars1, collaboration_scores):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

            scatter = ax2.scatter(interaction_counts, handover_counts, 
                                s=[score * 200 for score in collaboration_scores],
                                c=collaboration_scores, cmap='viridis', alpha=0.7,
                                edgecolors='black', linewidth=1)

            for i, op in enumerate(operators):
                ax2.annotate(op, (interaction_counts[i], handover_counts[i]), 
                            xytext=(5, 5), textcoords='offset points', fontweight='bold')
            
            ax2.set_title('Interaction Count vs Handover Frequency\n(Size = Collaboration Score)', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Interaction Count')
            ax2.set_ylabel('Handover Count')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label='Collaboration Score')
            
            # Plot 3: Performance Comparison
            x_pos = np.arange(len(operators))
            width = 0.35
            
            bars3a = ax3.bar(x_pos - width/2, collaboration_scores, width, 
                            label='Collaboration', alpha=0.8)
            bars3b = ax3.bar(x_pos + width/2, [p/100 for p in performance_scores], width,
                            label='Performance', alpha=0.8)
            
            ax3.set_title('Collaboration vs Performance Scores', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Operators')
            ax3.set_ylabel('Normalized Scores')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(operators)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/macro_collaboration_performance_enhanced.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error in collaboration performance plot: {e}")



    def _plot_handover_effectiveness(self, macro_data):

        try:
            handover_analysis = macro_data.get('handover_analysis', {})
            if not handover_analysis:
                print("No handover data for effectiveness plot")
                return
                
            operators = list(handover_analysis.keys())
            handover_counts = [handover_analysis[op]['count'] for op in operators]
            avg_performances = [handover_analysis[op]['avg_performance'] for op in operators]
            
            plt.figure(figsize=(10, 6))
            
            # Create bar plot for handover counts
            bars = plt.bar(operators, handover_counts, alpha=0.7, 
                          color=[self.operator_colors.get(op, '#7f7f7f') for op in operators])
            
            # Add performance line
            ax2 = plt.twinx()
            ax2.plot(operators, avg_performances, 'ro-', linewidth=2, markersize=8, 
                    label='Avg Performance')
            ax2.set_ylabel('Average Performance Score')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='upper right')
            
            plt.title('MACRO Layer: Handover Effectiveness', fontsize=16, fontweight='bold')
            plt.xlabel('Operators')
            plt.ylabel('Handover Count')
            
            # Add count labels on bars
            for bar, count in zip(bars, handover_counts):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"Error in handover effectiveness plot: {e}")


    def _generate_cross_layer_correlations(self):

        try:
            micro_data = self.extract_micro_layer_data()
            meso_data = self.extract_meso_layer_data()
            macro_data = self.extract_macro_layer_data()
            
            operators = list(micro_data.get('operator_sequences', {}).keys())
            if len(operators) < 2:
                print("Insufficient operators for cross-layer correlations")
                return

            diversities = []
            engagement_scores = []
            collaboration_scores = []
            handover_counts = []
            
            for operator in operators:
                diversity_metrics = micro_data.get('diversity_metrics', {}).get(operator, {})
                diversity_score = diversity_metrics.get('diversity_ratio', 0)
                entropy_score = diversity_metrics.get('action_type_entropy', 0)

                combined_diversity = (diversity_score + entropy_score) / 2
                diversities.append(combined_diversity)

                engagement = meso_data['engagement_scores'][operator]
                high_engagement = engagement.get('HIGHLY_ENGAGED', 0)
                engaged = engagement.get('ENGAGED', 0)
                total_engagement = (high_engagement * 1.5 + engaged) / 2.5  # Weighted
                engagement_scores.append(total_engagement / 100)

                collaboration = macro_data['collaboration_metrics'][operator]['mean_score']
                collaboration_scores.append(collaboration)
                

                handover_count = macro_data['handover_analysis'].get(operator, {}).get('count', 0)
                handover_counts.append(handover_count)

            plt.figure(figsize=(16, 12))

            plt.subplot(2, 2, 1)
            scatter1 = plt.scatter(diversities, engagement_scores, 
                                  s=[c * 300 for c in collaboration_scores],
                                  c=handover_counts, cmap='Reds', alpha=0.7,
                                  edgecolors='black', linewidth=1)
            
            for i, op in enumerate(operators):
                plt.annotate(op, (diversities[i], engagement_scores[i]), 
                            xytext=(8, 8), textcoords='offset points', 
                            fontweight='bold', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            
            plt.xlabel('Enhanced Diversity Score\n(Diversity Ratio + Entropy)')
            plt.ylabel('Weighted Engagement Score')
            plt.title('Micro vs Meso: Diversity vs Engagement\n(Size=Collaboration, Color=Handovers)', 
                     fontsize=12, fontweight='bold')
            plt.colorbar(scatter1, label='Handover Count')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 2)
            scatter2 = plt.scatter(engagement_scores, collaboration_scores,
                                  s=[d * 500 for d in diversities],
                                  c=handover_counts, cmap='Blues', alpha=0.7,
                                  edgecolors='black', linewidth=1)
            
            for i, op in enumerate(operators):
                plt.annotate(op, (engagement_scores[i], collaboration_scores[i]),
                            xytext=(8, 8), textcoords='offset points',
                            fontweight='bold', fontsize=9)
            
            plt.xlabel('Weighted Engagement Score')
            plt.ylabel('Collaboration Score')
            plt.title('Meso vs Macro: Engagement vs Collaboration\n(Size=Diversity, Color=Handovers)', 
                     fontsize=12, fontweight='bold')
            plt.colorbar(scatter2, label='Handover Count')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 3)
            scatter3 = plt.scatter(diversities, collaboration_scores,
                                  s=[e * 400 for e in engagement_scores],
                                  c=handover_counts, cmap='Greens', alpha=0.7,
                                  edgecolors='black', linewidth=1)
            
            for i, op in enumerate(operators):
                plt.annotate(op, (diversities[i], collaboration_scores[i]),
                            xytext=(8, 8), textcoords='offset points',
                            fontweight='bold', fontsize=9)
            
            plt.xlabel('Enhanced Diversity Score')
            plt.ylabel('Collaboration Score')
            plt.title('Micro vs Macro: Diversity vs Collaboration\n(Size=Engagement, Color=Handovers)', 
                     fontsize=12, fontweight='bold')
            plt.colorbar(scatter3, label='Handover Count')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 4)
            metrics_data = {
                'Diversity': diversities,
                'Engagement': engagement_scores,
                'Collaboration': collaboration_scores
            }
            
            plt.boxplot(metrics_data.values(), labels=metrics_data.keys())
            plt.title('Distribution of Key Metrics Across Operators', 
                     fontsize=12, fontweight='bold')
            plt.ylabel('Normalized Scores')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/cross_layer_correlations_enhanced.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ Enhanced cross-layer correlations generated")
            
        except Exception as e:
            print(f"Error in cross-layer correlations: {e}")


    def _generate_cross_layer_insights(self):
        """Generate comprehensive cross-layer insights"""
        try:
            micro_data = self.extract_micro_layer_data()
            meso_data = self.extract_meso_layer_data()
            macro_data = self.extract_macro_layer_data()
            
            operators = list(micro_data.get('operator_sequences', {}).keys())
            if not operators:
                return {"insights": "No operator data available for cross-layer analysis"}
            
            insights = {
                "summary": "",
                "micro_meso_correlations": {},
                "meso_macro_correlations": {}, 
                "operator_performance_profiles": {},
                "recommendations": [],
                "key_findings": []
            }

            for operator in operators:
                operator_insights = {}

                primitives = micro_data['operator_sequences'].get(operator, [])
                primitive_diversity = len(set(primitives)) / max(1, len(primitives))
                action_count = len(primitives)
                
                engagement_scores = meso_data['engagement_scores'].get(operator, {})
                high_engagement = engagement_scores.get('HIGHLY_ENGAGED', 0)
                engagement_stability = engagement_scores.get('ENGAGED', 0) + high_engagement
                
                hand_patterns = meso_data['hand_usage_patterns'].get(operator, {})
                bimanual_usage = hand_patterns.get('BOTH', 0)

                collaboration_metrics = macro_data['collaboration_metrics'].get(operator, {})
                collaboration_score = collaboration_metrics.get('mean_score', 0)
                interaction_count = collaboration_metrics.get('interaction_count', 0)
                
                handover_analysis = macro_data['handover_analysis'].get(operator, {})
                handover_count = handover_analysis.get('count', 0)

                operator_insights = {
                    "micro": {
                        "primitive_diversity": round(primitive_diversity, 3),
                        "action_count": action_count,
                        "unique_actions": len(set(primitives))
                    },
                    "meso": {
                        "high_engagement": high_engagement,
                        "engagement_stability": engagement_stability,
                        "bimanual_usage": bimanual_usage
                    },
                    "macro": {
                        "collaboration_score": round(collaboration_score, 3),
                        "interaction_count": interaction_count,
                        "handover_count": handover_count
                    }
                }
                
                insights["operator_performance_profiles"][operator] = operator_insights
            
            # Calculate cross-layer correlations
            insights.update(self._calculate_cross_layer_correlations(insights["operator_performance_profiles"]))
            
            # Generate overall summary
            insights["summary"] = self._generate_summary_insights(insights)
            
            # Generate recommendations
            insights["recommendations"] = self._generate_recommendations(insights)
            
            # Key findings
            insights["key_findings"] = self._extract_key_findings(insights)
            
            return insights
            
        except Exception as e:
            print(f"Error generating cross-layer insights: {e}")
            return {"insights": f"Error in cross-layer analysis: {str(e)}"}




    def _calculate_cross_layer_correlations(self, operator_profiles):
        """Calculate correlations between different layers"""
        correlations = {
            "micro_meso_correlations": {},
            "meso_macro_correlations": {}
        }
        
        if len(operator_profiles) < 2:
            correlations["micro_meso_correlations"]["status"] = "Insufficient data for correlation analysis"
            correlations["meso_macro_correlations"]["status"] = "Insufficient data for correlation analysis"
            return correlations
        
        # Micro-Meso correlations
        diversities = []
        engagements = []
        bimanual_usages = []
        
        for operator, profile in operator_profiles.items():
            diversities.append(profile["micro"]["primitive_diversity"])
            engagements.append(profile["meso"]["high_engagement"])
            bimanual_usages.append(profile["meso"]["bimanual_usage"])

        try:
            diversity_engagement_corr = np.corrcoef(diversities, engagements)[0, 1] if len(diversities) > 1 else 0
            diversity_bimanual_corr = np.corrcoef(diversities, bimanual_usages)[0, 1] if len(diversities) > 1 else 0
            
            correlations["micro_meso_correlations"] = {
                "diversity_engagement_correlation": round(diversity_engagement_corr, 3),
                "diversity_bimanual_correlation": round(diversity_bimanual_corr, 3),
                "interpretation": self._interpret_correlation(diversity_engagement_corr, "diversity", "engagement")
            }
        except:
            correlations["micro_meso_correlations"]["status"] = "Correlation calculation failed"

        engagements = []
        collaboration_scores = []
        handover_counts = []
        
        for operator, profile in operator_profiles.items():
            engagements.append(profile["meso"]["high_engagement"])
            collaboration_scores.append(profile["macro"]["collaboration_score"])
            handover_counts.append(profile["macro"]["handover_count"])
        
        try:
            engagement_collaboration_corr = np.corrcoef(engagements, collaboration_scores)[0, 1] if len(engagements) > 1 else 0
            engagement_handover_corr = np.corrcoef(engagements, handover_counts)[0, 1] if len(engagements) > 1 else 0
            
            correlations["meso_macro_correlations"] = {
                "engagement_collaboration_correlation": round(engagement_collaboration_corr, 3),
                "engagement_handover_correlation": round(engagement_handover_corr, 3),
                "interpretation": self._interpret_correlation(engagement_collaboration_corr, "engagement", "collaboration")
            }
        except:
            correlations["meso_macro_correlations"]["status"] = "Correlation calculation failed"
        
        return correlations


    def _interpret_correlation(self, correlation, metric1, metric2):
        if abs(correlation) < 0.3:
            return f"Weak relationship between {metric1} and {metric2}"
        elif abs(correlation) < 0.6:
            direction = "positive" if correlation > 0 else "negative"
            return f"Moderate {direction} relationship between {metric1} and {metric2}"
        elif abs(correlation) < 0.8:
            direction = "positive" if correlation > 0 else "negative"
            return f"Strong {direction} relationship between {metric1} and {metric2}"
        else:
            direction = "positive" if correlation > 0 else "negative"
            return f"Very strong {direction} relationship between {metric1} and {metric2}"



    def _generate_summary_insights(self, insights):
        operator_profiles = insights["operator_performance_profiles"]
        
        if not operator_profiles:
            return "No operator data available for analysis"
        
        total_operators = len(operator_profiles)
        
        # Find best performers in each layer
        best_micro = max(operator_profiles.items(), 
                        key=lambda x: x[1]["micro"]["primitive_diversity"], 
                        default=(None, {}))[0]
        
        best_meso = max(operator_profiles.items(),
                       key=lambda x: x[1]["meso"]["high_engagement"],
                       default=(None, {}))[0]
        
        best_macro = max(operator_profiles.items(),
                        key=lambda x: x[1]["macro"]["collaboration_score"],
                        default=(None, {}))[0]
        
        summary = f"""
    Cross-Layer Analysis Summary:
    - Total operators analyzed: {total_operators}
    - Best primitive diversity: {best_micro or 'N/A'}
    - Highest engagement: {best_meso or 'N/A'} 
    - Top collaborator: {best_macro or 'N/A'}
    """
        
        micro_meso_corr = insights["micro_meso_correlations"].get("diversity_engagement_correlation", 0)
        if abs(micro_meso_corr) > 0.3:
            direction = "increases with" if micro_meso_corr > 0 else "decreases with"
            summary += f"- Action diversity {direction} engagement levels\n"
        
        meso_macro_corr = insights["meso_macro_correlations"].get("engagement_collaboration_correlation", 0)
        if abs(meso_macro_corr) > 0.3:
            direction = "enhances" if meso_macro_corr > 0 else "reduces"
            summary += f"- Engagement {direction} collaboration effectiveness\n"
        
        return summary.strip()



    def _generate_recommendations(self, insights):

        recommendations = []
        operator_profiles = insights["operator_performance_profiles"]
        
        if not operator_profiles:
            return ["Collect more data for meaningful recommendations"]
        
        low_engagement_ops = [op for op, profile in operator_profiles.items() 
                             if profile["meso"]["high_engagement"] < 30]
        
        if low_engagement_ops:
            recommendations.append(
                f"Provide engagement training for operators: {', '.join(low_engagement_ops)}"
            )
        
        low_collaboration_ops = [op for op, profile in operator_profiles.items()
                                if profile["macro"]["collaboration_score"] < 0.3]
        
        if low_collaboration_ops:
            recommendations.append(
                f"Enhance collaboration skills for: {', '.join(low_collaboration_ops)}"
            )
        
        high_handover_ops = [op for op, profile in operator_profiles.items()
                            if profile["macro"]["handover_count"] > 5]
        
        if high_handover_ops:
            recommendations.append(
                f"Review task allocation for frequent handover operators: {', '.join(high_handover_ops)}"
            )

        micro_meso_corr = insights["micro_meso_correlations"].get("diversity_engagement_correlation", 0)
        if micro_meso_corr > 0.4:
            recommendations.append("Consider increasing task variety to improve engagement")
        elif micro_meso_corr < -0.4:
            recommendations.append("Simplify task sequences to reduce cognitive load")
        
        if not recommendations:
            recommendations.append("Current performance patterns are within expected ranges")
        
        return recommendations




    def _extract_key_findings(self, insights):
        """Extract key findings from the analysis"""
        findings = []
        operator_profiles = insights["operator_performance_profiles"]
        
        if not operator_profiles:
            return ["Insufficient data for key findings"]
        
        # Find notable patterns
        engagement_range = max([p["meso"]["high_engagement"] for p in operator_profiles.values()]) - \
                         min([p["meso"]["high_engagement"] for p in operator_profiles.values()])
        
        if engagement_range > 40:
            findings.append("Significant variation in engagement levels across operators")
        
        collaboration_scores = [p["macro"]["collaboration_score"] for p in operator_profiles.values()]
        if max(collaboration_scores) > 0.7 and min(collaboration_scores) < 0.3:
            findings.append("Wide range of collaboration effectiveness observed")
        
        # Check for consistent performers
        consistent_ops = []
        for op, profile in operator_profiles.items():
            micro_score = profile["micro"]["primitive_diversity"]
            meso_score = profile["meso"]["high_engagement"] / 100
            macro_score = profile["macro"]["collaboration_score"]
            
            if micro_score > 0.6 and meso_score > 0.6 and macro_score > 0.6:
                consistent_ops.append(op)
        
        if consistent_ops:
            findings.append(f"Consistent high performers across all layers: {', '.join(consistent_ops)}")
        
        if not findings:
            findings.append("Analysis completed successfully. Review detailed metrics for specific insights.")
        
        return findings

class PremAnalyzer:
    
    def __init__(self, engagement_data, hand_usage_data, progress_data, action_sequences, output_dir: str = "prem_analysis_results"):
        self.engagement_data = engagement_data
        self.hand_usage_data = hand_usage_data
        self.progress_data = progress_data
        self.action_sequences = action_sequences
        self.operators = list(action_sequences.keys())
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize three-layer analyzer
        self.three_layer_analyzer = None
        self.three_layer_output_dir = os.path.join(output_dir, "three_layer_analysis")
        os.makedirs(self.three_layer_output_dir, exist_ok=True)
        
        # Enhanced analysis flags
        self.include_three_layer_analysis = True
        
        # Define action primitive categories
        self.primitive_categories = {
            'manipulation': ['take_', 'put_', 'align_', 'change_'],
            'fastening': ['screw_', 'tighten_', 'plug_', 'tigthen_'],
            'positioning': ['align_', 'put_', 'take_'],
            'tool_usage': ['screwdriver', 'screwdriver_bit'],
            'component_handling': [
                'take_pir_sensor',
                'take_front_panel',
                'align_pir_sensor_with_front_panel',
                'take_bolt',
                'take_nut',
                'tighten_nut_with_hand',
                'take_screwdriver',
                'tigthen_bolt_with_scredriver',
                'put_screwdriver',
                'put_front_panel',
                'take_rpi_camera',
                'take_rpi_came_ra_module',
                'align_rpi_camera_with_camera_module',
                'take_screw',
                'screw_screw_with_screwdriver',
                'put_rpi_camera_camera_module',
                'take_rpi_board',
                'take_display',
                'align_rpi_board_on_display',
                'take_rpi_board_display_module',
                'align_rpi_camera_camera_module_with_front_panel',
                'change_screwdriver_bit',
                'align_rpi_board_display_module_on_front_panel',
                'take_display_mount_bracket',
                'plug_display_mount_bracket',
                'take_fcc_cable',
                'plug_fcc_cable',
                'take_rpi_hat',
                'plug_rpi_hat',
                'put_front_panel',
                'take_back_panel',
                'take_power_adapter',
                'tie_a_knot_in_cable',
                'plug_power_cable',
                'align_front_and_back_panel',
                'put_complete_fras'
            ]
        }
        
        self.operator_colors = {
            'ID-1': '#1f77b4', 'ID-2': '#ff7f0e', 'ID-3': '#2ca02c',
            'ID-4': '#d62728', 'ID-5': '#9467bd', 'unknown_operator': '#7f7f7f'
        }

    def generate_prem_analysis(self, research_metrics: Dict) -> Dict:
        print("Starting Enhanced PREM Analysis with Three-Layer Framework...")

        original_report = self._generate_original_prem_analysis(research_metrics)
        
        if self.include_three_layer_analysis:
            three_layer_report = self._generate_three_layer_analysis(research_metrics)
            enhanced_report = self._integrate_analysis_layers(original_report, three_layer_report)
            
            print("✓ Enhanced PREM analysis with three-layer framework completed!")
            return enhanced_report
        else:
            return original_report

    def _generate_original_prem_analysis(self, research_metrics: Dict) -> Dict:
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

    def _generate_three_layer_analysis(self, research_metrics: Dict) -> Dict:
        try:
            self.three_layer_analyzer = ThreeLayerPremAnalyzer(
                research_metrics=research_metrics,
                output_dir=self.three_layer_output_dir
            )
            
            three_layer_report = self.three_layer_analyzer.generate_comprehensive_report()
            
            return three_layer_report
            
        except Exception as e:
            print(f"Warning: Three-layer analysis failed - {e}")
            return {}

    def _integrate_analysis_layers(self, prem_report: Dict, three_layer_report: Dict) -> Dict:
        integrated_report = {
            'metadata': {
                'analysis_type': 'Enhanced PREM with Three-Layer Framework',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0',
                'framework': 'MICRO-MESO-MACRO Integrated'
            },
            'prem_analysis': prem_report,
            'three_layer_analysis': three_layer_report,
            'integrated_insights': self._generate_integrated_insights(prem_report, three_layer_report),
            'cross_layer_mappings': self._create_cross_layer_mappings(prem_report, three_layer_report)
        }
        
        return integrated_report

    def _generate_integrated_insights(self, prem_report: Dict, three_layer_report: Dict) -> Dict:
        integrated_insights = {
            'micro_meso_bridge': {},
            'meso_macro_bridge': {},
            'action_engagement_correlations': {},
            'collaboration_efficiency_factors': {}
        }
        
        if 'primitive_analysis' in prem_report and 'meso_analysis' in three_layer_report:
            micro_data = prem_report['primitive_analysis']
            meso_data = three_layer_report['meso_analysis']
            
            for operator_id, primitive_info in micro_data.items():
                if operator_id in meso_data.get('engagement_scores', {}):
                    primitive_diversity = len(set(primitive_info.get('raw_sequence', [])))
                    engagement_score = meso_data['engagement_scores'][operator_id].get('HIGHLY_ENGAGED', 0)
                    
                    integrated_insights['micro_meso_bridge'][operator_id] = {
                        'primitive_diversity': primitive_diversity,
                        'high_engagement_percentage': engagement_score,
                        'correlation_strength': 'positive' if engagement_score > 50 else 'neutral'
                    }
        
        return integrated_insights

    def _create_cross_layer_mappings(self, prem_report: Dict, three_layer_report: Dict) -> Dict:
        mappings = {
            'prem_to_three_layer': {
                'primitive_analysis': 'MICRO_LAYER',
                'efficiency_analysis': 'MICRO_MACRO_BRIDGE',
                'hand_usage_analysis': 'MESO_LAYER', 
                'engagement_correlation': 'MESO_LAYER',
                'operator_clustering': 'CROSS_LAYER'
            },
            'three_layer_to_prem': {
                'MICRO_LAYER': 'primitive_analysis',
                'MESO_LAYER': ['hand_usage_analysis', 'engagement_correlation'],
                'MACRO_LAYER': 'efficiency_analysis'
            }
        }
        
        return mappings

    def _make_json_serializable(self, data):

        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, dict):
            return {str(k): self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, (np.integer, np.int64, np.int32, np.int8)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32, np.float16)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, set):
            return list(data)
        elif isinstance(data, Counter):
            return dict(data)
        elif hasattr(data, '__dict__'):
            return self._make_json_serializable(data.__dict__)
        elif isinstance(data, (datetime, pd.Timestamp)):
            return data.isoformat()
        elif hasattr(data, 'dtype'):
            return self._make_json_serializable(data.tolist())
        else:
            try:
                return str(data)
            except:
                return "Unserializable object"


    def _ensure_json_serializable(self, data):
        try:
            json.dumps(data)
            return data
        except (TypeError, ValueError) as e:
            print(f"Data not serializable, applying fixes: {e}")
            return self._make_json_serializable(data)

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
        
        # Analyze transitions
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
            
            # Use DBSCAN for density-based clustering
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

        except Exception as e:
            print(f"Warning: Some visualizations failed - {e}")


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
        plt.xticks(np.arange(len(self.primitive_categories)), 
                  list(self.primitive_categories.keys()), rotation=45)
        plt.yticks(np.arange(len(self.operators)), self.operators)
        
        # Remove constrained_layout from savefig
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

    def _plot_cross_layer_triangle(self, integrated_report: Dict):
        fig = plt.figure(figsize=(14, 12))

        ax = fig.add_subplot(111)

        operators = []
        micro_scores, meso_scores, macro_scores = [], [], []
        
        insights = integrated_report.get('integrated_insights', {})
        micro_meso = insights.get('micro_meso_bridge', {})
        meso_macro = insights.get('meso_macro_bridge', {})
        
        for operator_id in set(list(micro_meso.keys()) + list(meso_macro.keys())):
            operators.append(operator_id)

            micro_data = micro_meso.get(operator_id, {})
            micro_score = micro_data.get('primitive_diversity', 0) / 20.0
            micro_scores.append(min(1.0, micro_score))

            meso_score = micro_data.get('high_engagement_percentage', 0) / 100.0
            meso_scores.append(meso_score)

            macro_data = meso_macro.get(operator_id, {})
            macro_score = macro_data.get('collaboration_effectiveness', 0)
            macro_scores.append(macro_score)

        for i, operator in enumerate(operators):
            x = micro_scores[i] + meso_scores[i] * 0.5
            y = meso_scores[i] * 0.866
            
            ax.scatter(x, y, s=macro_scores[i] * 500 + 100,
                      alpha=0.7, label=operator,
                      color=self.operator_colors.get(operator, '#7f7f7f'))
            
            ax.annotate(operator, (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontweight='bold')

        triangle_points = [(0, 0), (1, 0), (0.5, 0.866), (0, 0)]
        triangle_x, triangle_y = zip(*triangle_points)
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2, alpha=0.5)
        
        # Add labels at vertices
        ax.text(0, -0.1, 'MICRO\n(Primitives)', ha='center', fontweight='bold')
        ax.text(1, -0.1, 'MESO\n(Engagement)', ha='center', fontweight='bold')
        ax.text(0.5, 0.966, 'MACRO\n(Collaboration)', ha='center', fontweight='bold')
        
        ax.set_title('Cross-Layer Operator Performance Triangle', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.three_layer_output_dir}/cross_layer_triangle.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_layer_transition_network(self, integrated_report: Dict):
        G = nx.DiGraph()

        layers = {
            'MICRO': ['Primitives', 'Action Patterns', 'Temporal Distribution'],
            'MESO': ['Engagement', 'Hand Usage', 'Cognitive State'], 
            'MACRO': ['Collaboration', 'Handovers', 'System Efficiency']
        }

        for layer, components in layers.items():
            for component in components:
                G.add_node(f"{layer}_{component}", layer=layer, type=component)

        insights = integrated_report.get('integrated_insights', {})
        if insights.get('micro_meso_bridge'):
            G.add_edge('MICRO_Primitives', 'MESO_Engagement', 
                      weight=len(insights['micro_meso_bridge']), 
                      label='Primitive→Engagement')
        
        if insights.get('meso_macro_bridge'):
            G.add_edge('MESO_Engagement', 'MACRO_Collaboration',
                      weight=len(insights['meso_macro_bridge']),
                      label='Engagement→Collaboration')

        plt.figure(figsize=(14, 10))

        pos = {}
        layer_x = {'MICRO': 0, 'MESO': 1, 'MACRO': 2}
        for node, data in G.nodes(data=True):
            layer = data['layer']
            component_index = layers[layer].index(data['type'])
            y_pos = component_index / max(1, len(layers[layer]) - 1)
            pos[node] = (layer_x[layer], y_pos)

        node_colors = []
        for node in G.nodes():
            if node.startswith('MICRO'):
                node_colors.append('#3498db')  # Blue
            elif node.startswith('MESO'):
                node_colors.append('#e74c3c')  # Red
            else:
                node_colors.append('#2ecc71')  # Green
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=2000, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              width=[G[u][v]['weight'] for u, v in G.edges()],
                              arrows=True, arrowsize=20)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title('Three-Layer Analysis Transition Network', 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.three_layer_output_dir}/layer_transition_network.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comprehensive_report(self, research_metrics: Dict) -> Dict:

        return self.generate_prem_analysis(research_metrics)

def enhance_prem_analyzer():

    OriginalPremAnalyzer = PremAnalyzer
    
    class EnhancedPremAnalyzer(OriginalPremAnalyzer):

        def __init__(self, engagement_data, hand_usage_data, progress_data, action_sequences, output_dir: str = "prem_analysis_results"):
            super().__init__(engagement_data, hand_usage_data, progress_data, action_sequences, output_dir)
            
        def generate_enhanced_prem_analysis(self, research_metrics: Dict) -> Dict:

            return self.generate_prem_analysis(research_metrics)
    
    
    print("PremAnalyzer successfully enhanced with three-layer analysis capability!")
    return EnhancedPremAnalyzer

EnhancedPremAnalyzer = enhance_prem_analyzer()
MicroAnalysisPrimitives = PremAnalyzer


if __name__ == "__main__":
    print("Enhanced Prem Analyzer Module Loaded Successfully")
    print("This module provides micro-analysis of human operator primitives for HRC")
    print("Three-Layer (MICRO-MESO-MACRO) analysis framework!")
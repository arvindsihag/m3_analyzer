import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import random

class RobotSimulator:
    def __init__(self):
        self.handover_thresholds = {
            'fatigue': 65,          # Performance score below this triggers handover due to fatigue
            'safety': 70,           # Safety concerns threshold
            'inefficiency': 60,     # Inefficient technique threshold
            'difficulty': 55,       # Task difficulty threshold
            'engagement': 'IDLE'    # Engagement level that triggers handover
        }

        self.robot_capabilities = {
            'precision_tasks': ['align', 'position', 'measure', 'calibrate'],
            'force_tasks': ['screw', 'tighten', 'fasten', 'secure'],
            'repetitive_tasks': ['assemble', 'connect', 'install', 'mount'],
            'delicate_tasks': ['handle', 'place', 'adjust', 'orient']
        }

        self.robot_performance = {
            'precision': 0.95,      # 95% success rate for precision tasks
            'force': 0.92,          # 92% for force-required tasks
            'repetitive': 0.98,     # 98% for repetitive tasks
            'delicate': 0.85,       # 85% for delicate tasks (humans better)
            'general': 0.90         # 90% for general tasks
        }

    def assess_handover_need(self, performance_metrics: Dict, action_name: str, engagement_level: str, analysis: str) -> Dict:
        """Determine if robot handover is needed"""
        handover_decision = {
            'handover_recommended': False,
            'reasons': [],
            'urgency': 'LOW',  # LOW, MEDIUM, HIGH
            'robot_capability_score': 0.0,
            'expected_improvement': 0.0
        }
        
        score = performance_metrics.get('performance_score', 80)

        criteria_met = []

        if score < self.handover_thresholds['fatigue']:
            criteria_met.append(('performance', f"Low performance score: {score}/100"))
            handover_decision['urgency'] = 'HIGH' if score < 50 else 'MEDIUM'

        if performance_metrics.get('safety_concerns', False):
            criteria_met.append(('safety', "Safety concerns detected"))
            handover_decision['urgency'] = 'HIGH'

        if engagement_level in ['DISENGAGED', 'IDLE']:
            criteria_met.append(('engagement', f"Low engagement: {engagement_level}"))
            handover_decision['urgency'] = 'MEDIUM'

        if performance_metrics.get('difficulty_with_task', False):
            criteria_met.append(('difficulty', "Operator struggling with task"))
            handover_decision['urgency'] = 'MEDIUM'

        if performance_metrics.get('inefficient_technique', False):
            criteria_met.append(('inefficiency', "Inefficient technique detected"))
            handover_decision['urgency'] = 'MEDIUM'

        if criteria_met:
            handover_decision['handover_recommended'] = True
            handover_decision['reasons'] = [reason for _, reason in criteria_met]

            capability_score = self._calculate_robot_capability(action_name)
            handover_decision['robot_capability_score'] = capability_score

            expected_improvement = max(0, (capability_score * 100) - score)
            handover_decision['expected_improvement'] = expected_improvement

            if capability_score < 0.7 and handover_decision['urgency'] != 'HIGH':
                handover_decision['urgency'] = 'MEDIUM'
        
        return handover_decision

    def _calculate_robot_capability(self, action_name: str) -> float:
        action_lower = action_name.lower()
        capability_score = self.robot_performance['general']
        
        for category, keywords in self.robot_capabilities.items():
            if any(keyword in action_lower for keyword in keywords):
                capability_score = self.robot_performance[category]
                break

        variability = random.uniform(-0.05, 0.05)
        return max(0.6, min(0.99, capability_score + variability))

    def simulate_robot_performance(self, human_performance: Dict, action_name: str, handover_decision: Dict) -> Dict:

        capability_score = handover_decision.get('robot_capability_score', 0.85)
        base_score = capability_score * 100
        urgency_factor = {
            'LOW': 1.0,
            'MEDIUM': 0.95,
            'HIGH': 0.90
        }.get(handover_decision.get('urgency', 'LOW'), 1.0)

        variability = random.normalvariate(0, 3)  # ±3 points normally distributed
        
        final_score = base_score * urgency_factor + variability
        final_score = max(60, min(98, final_score))  # Keep within reasonable bounds
        
        return {
            'performance_score': final_score,
            'completion_time': human_performance.get('completion_time', 0) * 0.7,
            'error_rate': max(1, 10 - (final_score / 10)),  # Lower score = higher error rate
            'consistency': 90 + (final_score - 80) / 2,  # More consistent with higher scores
            'safety_score': min(100, 85 + (final_score - 70) / 2),
            'capability_utilization': capability_score
        }

    def generate_handover_scenarios(self, analysis_results: List[Dict]) -> List[Dict]:
        scenarios = []
        
        for i, result in enumerate(analysis_results):
            if i % 10 == 0:
                handover_need = self.assess_handover_need(
                    result['performance_metrics'],
                    result['action_name'],
                    result['engagement_level'],
                    result['raw_analysis']
                )
                
                if handover_need['handover_recommended']:
                    scenario = {
                        'frame_number': result['frame_number'],
                        'timestamp': result['timestamp'],
                        'action_name': result['action_name'],
                        'human_performance': result['performance_metrics']['performance_score'],
                        'handover_decision': handover_need,
                        'robot_performance': self.simulate_robot_performance(
                            result['performance_metrics'],
                            result['action_name'],
                            handover_need
                        )
                    }
                    scenarios.append(scenario)
        
        return scenarios
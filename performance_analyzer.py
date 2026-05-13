import numpy as np
from typing import Dict, List
import re
from collections import Counter

class PerformanceAnalyzer:
    def __init__(self):
        self.fatigue_patterns = [
            'tired', 'fatigue', 'slow.*movement', 'rubbing.*eyes', 'yawning',
            'sluggish', 'straining', 'struggling', 'frustrated', 'repeating.*mistakes',
            'shaking', 'trembling', 'unsteady'
        ]
        
        self.safety_patterns = [
            'unsafe', 'dangerous', 'risk', 'hazard', 'precarious', 'accident',
            'injury', 'harm', 'careless', 'reckless', 'improper.*grip',
            'wrong.*tool', 'incorrect.*posture'
        ]
        
        self.inefficiency_patterns = [
            'awkward', 'inefficient', 'poor.*technique', 'wrong.*method',
            'wasting.*time', 'unnecessary.*movement', 'confused',
            'hesitating', 'uncertain', 'searching'
        ]
        
        self.difficulty_patterns = [
            'clearly struggling', 'obvious difficulty', 'significant trouble',
            'multiple failed attempts', 'completely stuck', 'cannot proceed',
            'unable to continue', 'repeated failures', 'struggling with',
            'having difficulty with', 'failing to', 'unsuccessful in'
        ]

    def analyze_performance(self, analysis: str, engagement_level: str, hand_usage: str, action_name: str) -> Dict:
        a = analysis.lower()
        
        metrics = {
            'fatigue_detected': False,
            'safety_concerns': False,
            'inefficient_technique': False,
            'difficulty_with_task': False,
            'performance_score': 80,  
            'confidence_level': 0.8,  
            'improvement_suggestions': []
        }
        
        fatigue_score = self._pattern_match(a, self.fatigue_patterns)
        if fatigue_score > 0.3:
            metrics['fatigue_detected'] = True
            metrics['performance_score'] -= fatigue_score * 20

        safety_score = self._pattern_match(a, self.safety_patterns)
        if safety_score > 0.4:
            metrics['safety_concerns'] = True
            metrics['performance_score'] -= safety_score * 25

        inefficiency_score = self._pattern_match(a, self.inefficiency_patterns)
        if inefficiency_score > 0.3:
            metrics['inefficient_technique'] = True
            metrics['performance_score'] -= inefficiency_score * 15

        difficulty_score = self._pattern_match(a, self.difficulty_patterns)
        # if difficulty_score > 0.4:
        if difficulty_score > 0.6:
            metrics['difficulty_with_task'] = True
            metrics['performance_score'] -= difficulty_score * 20

        engagement_impact = {
            'HIGHLY_ENGAGED': 10,
            'ENGAGED': 0,
            'PREPARING': -5,
            'IDLE': -15,
            'DISENGAGED': -25
        }
        metrics['performance_score'] += engagement_impact.get(engagement_level, 0)

        if engagement_level == 'HIGHLY_ENGAGED':
            metrics['performance_score'] += 15
        elif engagement_level == 'ENGAGED':
            metrics['performance_score'] += 5

        if hand_usage == 'BOTH':
            metrics['performance_score'] += 8
        elif hand_usage in ['LEFT', 'RIGHT']:
            metrics['performance_score'] += 3

        if hand_usage in ['UNCERTAIN', 'NONE']:
            metrics['performance_score'] -= 10

        complexity = self._assess_task_complexity(action_name)
        metrics['performance_score'] -= (1 - complexity) * 5  

        metrics['performance_score'] = max(30, min(95, metrics['performance_score']))

        metrics['improvement_suggestions'] = self._generate_suggestions(metrics, a)
        
        return metrics


    def _pattern_match(self, text: str, patterns: List[str]) -> float:
        matches = 0
        for pattern in patterns:

            if re.search(r'\b' + pattern + r'\b', text, re.IGNORECASE):
                matches += 1
        
        return min(1.0, matches / max(1, len(patterns)))

    def _assess_task_complexity(self, action_name: str) -> float:
        """Assess task complexity (0-1, where 1 is most complex)"""
        complexity_scores = {
            'align': 0.8, 'screw': 0.7, 'tighten': 0.7, 'connect': 0.6,
            'assemble': 0.5, 'install': 0.5, 'position': 0.4, 'place': 0.3,
            'take': 0.2, 'put': 0.2, 'check': 0.3, 'adjust': 0.4 
        }
        
        for verb, score in complexity_scores.items():
            if verb in action_name.lower():
                return score
        
        return 0.5

    def _generate_suggestions(self, metrics: Dict, analysis: str) -> List[str]:
        suggestions = []
        
        if metrics['fatigue_detected']:
            suggestions.extend([
                "Consider taking short breaks to reduce fatigue",
                "Ensure proper ergonomic setup to minimize strain",
                "Rotate tasks to maintain focus and reduce monotony"
            ])
        
        if metrics['safety_concerns']:
            suggestions.extend([
                "Review safety procedures before continuing",
                "Use appropriate personal protective equipment",
                "Ensure workspace is clear of hazards"
            ])
        
        if metrics['inefficient_technique']:
            suggestions.extend([
                "Review optimal techniques for this task",
                "Consider tool alternatives or adjustments",
                "Practice the motion to improve efficiency"
            ])
        
        if metrics['difficulty_with_task']:
            suggestions.extend([
                "Break down the task into smaller steps",
                "Request assistance or guidance if available",
                "Review instructional materials for this task"
            ])

        if len(suggestions) < 2:
            suggestions.extend([
                "Maintain consistent pace and rhythm",
                "Double-check alignments before final assembly",
                "Keep tools organized and within reach"
            ])
        
        return suggestions[:3]
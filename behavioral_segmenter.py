from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import Counter

class BehavioralPattern(Enum):
    EFFICIENT_EXECUTION = "efficient_execution"
    STRUGGLING = "struggling"
    ROBOT_DEPENDENT = "robot_dependent"
    SAFETY_CONSCIOUS = "safety_conscious"
    DISTRACTED = "distracted"
    COLLABORATIVE = "collaborative"

@dataclass
class BehavioralSegment:
    pattern: BehavioralPattern
    time_window: Tuple[float, float]
    dominant_actions: List[str]
    performance_metrics: Dict
    intervention_triggers: List[str]
    improvement_suggestions: List[str]

class BehavioralSegmenter:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        
    def identify_behavioral_patterns(self, analysis_results: List[Dict]) -> List[BehavioralSegment]:
        patterns = []
        
        for i in range(len(analysis_results) - self.window_size + 1):
            window = analysis_results[i:i + self.window_size]
            pattern = self._analyze_behavioral_pattern(window)
            
            if pattern:
                patterns.append(BehavioralSegment(
                    pattern=pattern,
                    time_window=(window[0]['timestamp'], window[-1]['timestamp']),
                    dominant_actions=self._get_dominant_actions(window),
                    performance_metrics=self._calculate_performance_metrics(window),
                    intervention_triggers=self._identify_intervention_triggers(window),
                    improvement_suggestions=self._generate_suggestions(window, pattern)
                ))
        
        return patterns
    
    def _analyze_behavioral_pattern(self, window: List[Dict]) -> BehavioralPattern:
        avg_performance = np.mean([r['performance_metrics']['performance_score'] for r in window])
        avg_collaboration = np.mean([r.get('collaboration_score', 0) for r in window])
        safety_issues = any(r['safety_assessment'] in ['MEDIUM_RISK', 'HIGH_RISK'] for r in window)
        
        if avg_performance > 80 and avg_collaboration < 0.3:
            return BehavioralPattern.EFFICIENT_EXECUTION
        elif avg_performance < 60 and avg_collaboration < 0.4:
            return BehavioralPattern.STRUGGLING
        elif avg_collaboration > 0.6:
            return BehavioralPattern.COLLABORATIVE
        elif safety_issues:
            return BehavioralPattern.SAFETY_CONSCIOUS
        elif any(r['engagement_level'] in ['IDLE', 'DISENGAGED'] for r in window):
            return BehavioralPattern.DISTRACTED
        elif avg_performance < 70 and avg_collaboration > 0.5:
            return BehavioralPattern.ROBOT_DEPENDENT
        
        return None
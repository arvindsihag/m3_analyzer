import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class TemporalSegmentType(Enum):
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    TRANSITION = "transition"
    COLLABORATION = "collaboration"
    HANDOVER = "handover"

@dataclass
class TemporalSegment:
    segment_type: TemporalSegmentType
    start_time: float
    end_time: float
    duration: float
    dominant_actions: List[str]
    engagement_pattern: str
    collaboration_level: float
    metadata: Dict

class TemporalSegmenter:
    def __init__(self, min_segment_duration: float = 2.0):
        self.min_duration = min_segment_duration
        self.segments = []
        
    def segment_assembly_process(self, analysis_results: List[Dict]) -> List[TemporalSegment]:
        segments = []
        current_segment = None
        
        for i, result in enumerate(analysis_results):
            segment_type = self._classify_segment_type(result, current_segment)
            
            if current_segment is None or segment_type != current_segment.segment_type:
                if current_segment:
                    current_segment.end_time = result['timestamp']
                    current_segment.duration = current_segment.end_time - current_segment.start_time
                    if current_segment.duration >= self.min_duration:
                        segments.append(current_segment)
                current_segment = TemporalSegment(
                    segment_type=segment_type,
                    start_time=result['timestamp'],
                    end_time=result['timestamp'],
                    duration=0.0,
                    dominant_actions=[result['action_name']],
                    engagement_pattern=result['engagement_level'],
                    collaboration_level=result.get('collaboration_score', 0.0),
                    metadata={'frame_start': result['frame_number']}
                )
            else:
                current_segment.dominant_actions.append(result['action_name'])
                current_segment.end_time = result['timestamp']
                current_segment.collaboration_level = max(
                    current_segment.collaboration_level,
                    result.get('collaboration_score', 0.0)
                )
                current_segment.metadata['frame_end'] = result['frame_number']
        
        return segments
    
    def _classify_segment_type(self, result: Dict, current_segment: TemporalSegment) -> TemporalSegmentType:
        action_name = result['action_name'].lower()
        engagement = result['engagement_level']
        collaboration = result.get('collaboration_score', 0.0)
        
        if any(x in action_name for x in ['take', 'get', 'fetch']):
            return TemporalSegmentType.PREPARATION
        elif any(x in action_name for x in ['align', 'screw', 'connect', 'assemble']):
            if collaboration > 0.6:
                return TemporalSegmentType.COLLABORATION
            return TemporalSegmentType.EXECUTION
        elif any(x in action_name for x in ['check', 'verify', 'inspect']):
            return TemporalSegmentType.VERIFICATION
        elif engagement in ['IDLE', 'DISENGAGED'] and collaboration > 0.3:
            return TemporalSegmentType.HANDOVER
        else:
            return TemporalSegmentType.TRANSITION


    def _calculate_collaboration_efficiency(self, temporal_segments: List[TemporalSegment], behavioral_patterns: List) -> float:
        if not temporal_segments or not behavioral_patterns:
            return 0.0
        collaboration_segments = [seg for seg in temporal_segments 
                                if seg.segment_type == TemporalSegmentType.COLLABORATION]
        
        if not collaboration_segments:
            return 0.0

        total_collaboration_time = sum(seg.duration for seg in collaboration_segments)
        total_time = sum(seg.duration for seg in temporal_segments)
        
        if total_time == 0:
            return 0.0

        time_efficiency = total_collaboration_time / total_time

        behavioral_adjustment = 1.0
        for pattern in behavioral_patterns:
            if pattern.pattern.value in ['efficient_execution', 'collaborative']:
                behavioral_adjustment += 0.1
            elif pattern.pattern.value in ['struggling', 'distracted']:
                behavioral_adjustment -= 0.1
        
        final_efficiency = max(0.0, min(1.0, time_efficiency * behavioral_adjustment))
        return final_efficiency
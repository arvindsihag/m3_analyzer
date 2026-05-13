# sequence_learner.py
import json
import numpy as np
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from datetime import datetime

class SequenceLearner:
    def __init__(self, sequences_file: str = "learned_sequences.json"):
        self.sequences_file = sequences_file
        self.operator_sequences = defaultdict(list)
        self.transition_counts = defaultdict(Counter)
        self.action_frequencies = Counter()
        self.learned_sub_assemblies = {}
        self.efficient_patterns = {}
        self._load_sequences()

    
    def _load_sequences(self):
        """Load learned sequences from file"""
        if os.path.exists(self.sequences_file):
            try:
                with open(self.sequences_file, 'r') as f:
                    data = json.load(f)
                
                self.operator_sequences = defaultdict(list, data.get('operator_sequences', {}))
                self.transition_counts = defaultdict(Counter)
                for op, transitions in data.get('transition_counts', {}).items():
                    self.transition_counts[op] = Counter(transitions)
                self.action_frequencies = Counter(data.get('action_frequencies', {}))
                self.learned_sub_assemblies = data.get('learned_sub_assemblies', {})
                self.efficient_patterns = data.get('efficient_patterns', {})
                
                print(f"✓ Loaded {len(self.operator_sequences)} operator sequences")
            except Exception as e:
                print(f"Error loading sequences: {e}")


    def _save_sequences(self):
        try:
            data = {
                'operator_sequences': {},
                'transition_counts': {},
                'action_frequencies': {},
                'learned_sub_assemblies': [],
                'efficient_patterns': [],
                'last_updated': datetime.now().isoformat(),
                'timestamp': datetime.now().isoformat()
            }

            for op, sequences in self.operator_sequences.items():
                data['operator_sequences'][str(op)] = [str(action) for action in sequences]

            for action, freq in self.action_frequencies.items():
                data['action_frequencies'][str(action)] = int(freq) if hasattr(freq, '__int__') else 0

            for key, count in self.transition_counts.items():
                if isinstance(key, tuple):

                    serializable_key = "->".join(str(item) for item in key)
                else:
                    serializable_key = str(key)
                
                if hasattr(count, '__int__'):
                    data['transition_counts'][serializable_key] = int(count)
                elif isinstance(count, (int, float)):
                    data['transition_counts'][serializable_key] = int(count)
                else:

                    try:
                        if hasattr(count, 'values'):
                            data['transition_counts'][serializable_key] = sum(count.values())
                        else:
                            data['transition_counts'][serializable_key] = 1
                    except:
                        data['transition_counts'][serializable_key] = 1
            
            if hasattr(self, 'learned_sub_assemblies'):
                serializable_sub_assemblies = []
                for assembly in self.learned_sub_assemblies:
                    if isinstance(assembly, dict):
                        serializable_assembly = {}
                        for k, v in assembly.items():
                            serializable_assembly[str(k)] = str(v) if not isinstance(v, (int, float, bool)) else v
                        serializable_sub_assemblies.append(serializable_assembly)
                    else:
                        serializable_sub_assemblies.append(str(assembly))
                data['learned_sub_assemblies'] = serializable_sub_assemblies

            if hasattr(self, 'efficient_patterns'):
                serializable_patterns = []
                for pattern in self.efficient_patterns:
                    if isinstance(pattern, dict):
                        serializable_pattern = {}
                        for k, v in pattern.items():
                            serializable_pattern[str(k)] = str(v) if not isinstance(v, (int, float, bool)) else v
                        serializable_patterns.append(serializable_pattern)
                    elif isinstance(pattern, (str, int, float)):
                        serializable_patterns.append(str(pattern))
                    else:
                        serializable_patterns.append(str(pattern))
                data['efficient_patterns'] = serializable_patterns

            sequences_file = "learned_sequences.json"
            with open(sequences_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Sequences saved to: {sequences_file}")
            return sequences_file
            
        except Exception as e:
            print(f"Error saving sequences: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_action_components(self, action_name: str) -> tuple:
        clean_name = re.sub(r'^\d+_', '', action_name)
        clean_name = re.sub(r'\.mp4$', '', clean_name)

        parts = clean_name.split('_')
        if len(parts) >= 2:
            action_verb = parts[0]
            object_name = '_'.join(parts[1:])
            return action_verb, object_name
        return clean_name, ""
    
    def add_sequence(self, operator_id: str, action_name: str):
        if not action_name:
            return

        self.operator_sequences[operator_id].append(action_name)
        self.action_frequencies[action_name] += 1

        sequence = self.operator_sequences[operator_id]
        if len(sequence) >= 2:
            previous_action = sequence[-2]
            self.transition_counts[operator_id][(previous_action, action_name)] += 1

        self._learn_patterns(operator_id)

        if len(self.operator_sequences[operator_id]) % 10 == 0:
            self._save_sequences()


    def _learn_patterns(self, operator_id: str):
        sequence = self.operator_sequences[operator_id]
        
        if len(sequence) < 3:
            return

        for i in range(len(sequence) - 2):
            pattern = tuple(sequence[i:i+3])
            pattern_key = "->".join(pattern)
            
            if pattern_key not in self.efficient_patterns:
                self.efficient_patterns[pattern_key] = {
                    'count': 0,
                    'operators': set(),
                    'first_seen': datetime.now().isoformat()
                }
            
            self.efficient_patterns[pattern_key]['count'] += 1
            self.efficient_patterns[pattern_key]['operators'].add(operator_id)
    
    def predict_next_actions(self, previous_actions: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        if not previous_actions:
            return self._get_common_starting_actions(top_n)
        
        last_action = previous_actions[-1]
        predictions = []

        for operator_id, transitions in self.transition_counts.items():
            for (source, target), count in transitions.items():
                if source == last_action:
                    total_transitions = sum(transitions.values())
                    probability = count / total_transitions if total_transitions > 0 else 0
                    predictions.append((target, probability))
        
        if not predictions:
            return self._get_common_actions(top_n)
        unique_predictions = {}
        for action, prob in predictions:
            if action in unique_predictions:
                unique_predictions[action] = max(unique_predictions[action], prob)
            else:
                unique_predictions[action] = prob
        
        sorted_predictions = sorted(unique_predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:top_n]

    def _get_common_starting_actions(self, top_n: int) -> List[Tuple[str, float]]:
        starting_actions = []
        for operator_id, sequence in self.operator_sequences.items():
            if sequence:
                starting_actions.append(sequence[0])
        
        if not starting_actions:
            return [('start_assembly', 1.0)]
        
        action_counts = Counter(starting_actions)
        total = sum(action_counts.values())
        return [(action, count/total) for action, count in action_counts.most_common(top_n)]

    def _get_common_actions(self, top_n: int) -> List[Tuple[str, float]]:
        if not self.action_frequencies:
            return [('continue_assembly', 1.0)]
        
        total = sum(self.action_frequencies.values())
        return [(action, count/total) for action, count in self.action_frequencies.most_common(top_n)]
    
    def get_efficient_patterns(self, min_frequency: int = 2) -> Dict:
        return {pattern: data for pattern, data in self.efficient_patterns.items() 
                if data['count'] >= min_frequency}
    
    def get_operator_stats(self, operator_id: str) -> Dict:
        """Get statistics for a specific operator"""
        if operator_id not in self.operator_sequences:
            return {}

        sequence = self.operator_sequences[operator_id]
        return {
            'total_actions': len(sequence),
            'unique_actions': len(set(sequence)),
            'most_common_action': Counter(sequence).most_common(1)[0] if sequence else None,
            'action_frequency': dict(Counter(sequence))
        }

    
    def _generate_object_based_predictions(self, current_action: str) -> List[Tuple[str, float]]:
        current_verb, current_object = self.extract_action_components(current_action)
        
        if not current_verb:
            return [('continue_assembly', 1.0)]

        predictions = []

        if current_verb == 'take':
            predictions = self._get_actions_for_objects(['put', 'align', 'assemble'], current_object)
        
        elif current_verb == 'put':
            predictions = self._get_actions_for_objects(['take', 'align', 'check'], "")
        
        elif current_verb == 'align':
            predictions = self._get_actions_for_objects(['screw', 'tighten', 'check'], current_object)
        
        elif current_verb in ['screw', 'tighten']:
            predictions = self._get_actions_for_objects(['check', 'take', 'assemble'], "")
        
        elif current_verb == 'connect':
            predictions = self._get_actions_for_objects(['test', 'check', 'assemble'], current_object)
        
        else:
            predictions = [('continue_assembly', 0.5), ('check', 0.3), ('complete', 0.2)]
        
        return predictions
    
    def _get_actions_for_objects(self, verbs: List[str], current_object: str) -> List[Tuple[str, float]]:
        predictions = []
        total_verbs = len(verbs)
        
        for i, verb in enumerate(verbs):
            probability = 0.7 - (i * 0.2)
            if current_object and verb in self.action_objects and self.action_objects[verb]:
                action_name = f"{verb}_{current_object}"
                predictions.append((action_name, probability))
            else:
                predictions.append((verb, probability))
        
        return predictions
    
    def get_learning_status(self) -> str:
        total_transitions = sum(len(v) for v in self.transitions.values())
        total_actions = len(self.action_objects)
        if total_transitions == 0:
            return "LEARNING_MODE: No patterns learned yet"
        elif total_transitions < 10:
            return f"LEARNING_MODE: {total_transitions} transitions observed"
        else:
            return f"PREDICTION_MODE: {total_transitions} transitions learned"
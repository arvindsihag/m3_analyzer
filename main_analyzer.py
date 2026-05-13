import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
import torch
import json
import cv2
import random
import numpy as np
from collections import deque, Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_analyzer import LlavaAnalyzer, extract_engagement_level, extract_hand_usage, extract_safety_assessment
from object_detector import ObjectDetector
from report_generator import ReportGenerator

from gpu_config import gpu_config
from gpu_utils import print_gpu_status, clear_gpu_cache

from robot_simulator import RobotSimulator
from performance_analyzer import PerformanceAnalyzer
from action_sequence_analyzer import ActionSequenceAnalyzer
from sequence_learner import SequenceLearner
from behavioral_segmenter import BehavioralSegmenter
from temporal_segmenter import TemporalSegmenter
# from report_analyzer import PremAnalyzer
from report_analyzer import PremAnalyzer
# from prem_analyzer import MicroAnalysisPrimitives
from prem_analyzer import PremAnalyzer, ThreeLayerPremAnalyzer



class EnhancedVideoAnalyzer:
    ASSEMBLY_HIERARCHY = {
        'sub_assemblies': {}
    }


    def __init__(self, config_path: str = "analysis_config.json", gpu_id: Optional[int] = None):
        self.config = self._load_config(config_path)
        self.gpu_id = gpu_id
        

        self.num_gpus = gpu_config.get_num_gpus()
        print(f"Available GPUs: {self.num_gpus}")
        
        self.llava_analyzer = LlavaAnalyzer(
            model_path=self.config['llava_model_path'],
            gpu_id=gpu_id,
            use_fp16=self.config.get('use_fp16', True),
            use_amp=self.config.get('use_amp', True)
        )
        
        self.object_detector = ObjectDetector(model_path=self.config['yolo_model_path'], conf_threshold=self.config.get('yolo_conf_threshold', 0.6), gpu_id=0)
        self.report_generator = ReportGenerator(output_dir=self.config['output_dir'])
        
        self.object_memory = deque(maxlen=self.config.get('memory_size', 15))
        self.action_memory = deque(maxlen=self.config.get('memory_size', 15))
        self.sequence_context = defaultdict(list)

        self.sequence_analyzer = ActionSequenceAnalyzer()
        self.sequence_learner = SequenceLearner() 

        self.num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {self.num_gpus}")

        self.process_pool = None
        if self.num_gpus > 1:
            from concurrent.futures import ProcessPoolExecutor
            self.process_pool = ProcessPoolExecutor(max_workers=self.num_gpus)
        
        print("EnhancedVideoAnalyzer initialized successfully with multi-GPU support")



        self.robot_available = False 
        self.robot_assisted_actions = []

        self.robot_metrics = {
            'assistance_level': [],
            'collaboration_scores': [],
            'intervention_times': []
        }


        self.temporal_segmenter = TemporalSegmenter()
        self.behavioral_segmenter = BehavioralSegmenter()
        self.collaboration_metrics = {'temporal_segments': [], 'behavioral_patterns': [], 'collaboration_efficiency': 0.0, 'handover_effectiveness': 0.0}


        self.video_sequences = defaultdict(list)


        self.research_metrics = {
            'engagement_data': [],
            'hand_use_data': [],
            'safety_data': [],
            'action_times': defaultdict(list),
            'operator_profiles': defaultdict(lambda: {
                'hand_usage': Counter(),
                'engagement': Counter(),
                'completion_times': [],
                'actions_performed': set()
            })
        }

        self.sub_assembly_progress = {}
        self.current_sub_assembly = None
        
        print("EnhancedVideoAnalyzer initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        defaults = {
            'llava_model_path': '/workspace/raid/stu-1/BLIP/model_lm/llava-hf/llava-1.5-7b-hf',
            'yolo_model_path': '/workspace/raid/stu-1/BLIP/d_weight/fras_best.pt',
            'yolo_conf_threshold': 0.5,
            'yolo_fps': 1,
            'llava_fps': 1,
            'max_prompt_tokens': 400,
            'temperature': 0.7,
            'top_p': 0.9,
            'memory_size': 15,
            'dataset_root_path': '/workspace/raid/stu-1/BLIP/dataset_op',
            'output_dir': 'multi_operator_analysis_results',
            'batch_size': 16,
            'max_workers': 16,
            'use_fp16': True,
            'use_amp': True,
            'llava_max_tokens': 32,
            'operators': [],
            'efficient_sequences_file': 'efficient_sequences.json'
        }
        
        for k, v in defaults.items():
            config.setdefault(k, v)
            
        return config


    def extract_action_name(self, video_path: str, dataset_type: str = "c1") -> str:
        filename = os.path.basename(video_path)

        if dataset_type == "c2":
            if any(filename.startswith(f"{i:03d}_") for i in range(0, 1000)):
                filename = filename[4:]
            if filename.endswith('_rgb.mp4'):
                filename = filename[:-8]
        else:
            if filename.startswith('clip_'):
                filename = filename[5:]
            if any(filename.startswith(f"{i:03d}_") for i in range(0, 1000)):
                filename = filename[4:]

        if filename.endswith('.mp4'):
            filename = filename[:-4]
        # return filename
        return self.sequence_analyzer.extract_action_components(filename)['full_action']


    def _save_efficient_sequences(self):

        try:
            efficient_sequences_file = self.config.get('efficient_sequences_file', 'efficient_sequences.json')

            efficient_sequences = {
                'learned_sequences': {},
                'performance_data': {},
                'optimal_paths': {},
                'timestamp': datetime.now().isoformat()
            }

            for operator_id, sequence in self.sequence_analyzer.operator_sequences.items():
                efficient_sequences['learned_sequences'][operator_id] = sequence

                optimal_path = self.sequence_analyzer.find_optimal_path(operator_id)
                if optimal_path:
                    efficient_sequences['optimal_paths'][operator_id] = {
                        'optimal_sequence': optimal_path.get('optimal_path', []),
                        'efficiency': optimal_path.get('time_efficiency', 0),
                        'improvement': optimal_path.get('path_improvement', 0)
                    }

            for action_name, data in self.sequence_analyzer.action_base['best_performance'].items():
                efficient_sequences['performance_data'][action_name] = {
                    'best_time': data.get('best_time', 0),
                    'best_operator': data.get('best_operator', 'unknown'),
                    'executions_considered': data.get('total_executions_considered', 0)
                }

            with open(efficient_sequences_file, 'w') as f:
                json.dump(efficient_sequences, f, indent=2)
            
            print(f" Efficient sequences saved to: {efficient_sequences_file}")
            return efficient_sequences_file
            
        except Exception as e:
            print(f"Error saving efficient sequences: {e}")
            return None

    def _load_efficient_sequences(self):

        try:
            efficient_sequences_file = self.config.get('efficient_sequences_file', 'efficient_sequences.json')
            
            if os.path.exists(efficient_sequences_file):
                with open(efficient_sequences_file, 'r') as f:
                    efficient_sequences = json.load(f)
                
                print(f"✓ Loaded efficient sequences from: {efficient_sequences_file}")
                return efficient_sequences
            else:
                print("ℹ No existing efficient sequences file found")
                return None
                
        except Exception as e:
            print(f"Error loading efficient sequences: {e}")
            return None



    def extract_operator_id(self, video_path: str) -> str:
        """Extract operator ID from path"""
        path_parts = video_path.split(os.sep)
        for part in path_parts:
            if part in self.config['operators']:
                return part
        return "unknown_operator"

    def find_c2_video_for_action(self, action_name: str, operator_id: str) -> Optional[str]:
        c2_dir = os.path.join(self.config['dataset_root_path'], operator_id, 'c2_rgb_clips')
        if not os.path.isdir(c2_dir):
            return None
            
        for filename in os.listdir(c2_dir):
            if filename.endswith('.mp4'):
                c2_action = self.extract_action_name(os.path.join(c2_dir, filename), "c2")
                if c2_action == action_name:
                    return os.path.join(c2_dir, filename)
        return None

    def get_context_string(self) -> str:
        if not self.object_memory:
            return "No objects detected recently"
        object_counts = Counter(list(self.object_memory)[-10:])
        return "Recently seen: " + ", ".join([f"{obj}({count}x)" for obj, count in object_counts.most_common(5)])

    def update_memory(self, detected_objects: List[str], current_action: str):
        try:
            self.object_memory.extend(detected_objects)
        except Exception:
            pass
        self.action_memory.append(current_action)

    def _calculate_hierarchical_progress(self, action_name: str, frame_number: int, total_frames: int, analysis: str) -> int:
        base = min(100, max(0, int((frame_number / max(total_frames - 1, 1)) * 100)))
        if frame_number == 0:
            return min(10, base)
        if frame_number == total_frames - 1:
            return 100

        self._detect_sub_assembly_transitions(action_name, analysis)
        adjusted_progress = self._adjust_progress_for_hierarchy(action_name, base, analysis)
        return min(100, adjusted_progress)

    def _detect_sub_assembly_transitions(self, action_name: str, analysis: str):
        """Detect sub-assembly transitions"""
        subs = self.ASSEMBLY_HIERARCHY.get('sub_assemblies', {})
        for sub_assembly, actions in subs.items():
            if action_name in actions and self.current_sub_assembly != sub_assembly:
                self.current_sub_assembly = sub_assembly
                self.sub_assembly_progress[sub_assembly] = 0
                print(f"Starting sub-assembly: {sub_assembly}")
                break

    def _adjust_progress_for_hierarchy(self, action_name: str, base_progress: int, analysis: str) -> int:
        """Adjust progress based on learned hierarchy and analysis"""
        a = analysis.lower()
        action_type = self.sequence_analyzer.extract_action_components(action_name)['action_type']
        
        # Use learned action types for progress adjustment
        if self.current_sub_assembly:
            if any(w in a for w in ['complete', 'finished', 'done', 'assembled', 'secured', 'fastened']):
                return min(100, base_progress + 20)

        progress_weights = {
            'fasten': 5,    # Fastening actions are significant
            'align': 3,     # Alignment actions are moderate
            'take': 1,      # Taking actions are basic
            'put': 1,       # Putting actions are basic
            'inspect': 2    # Inspection actions are moderate
        }
        
        weight = progress_weights.get(action_type, 1)
        return min(100, base_progress + weight)



    def _update_research_metrics(self, research_data: Dict, operator_id: str):

        try:
            research_data['sub_assembly'] = self.current_sub_assembly or 'main_assembly'
            
            numeric_timestamp = float(research_data['frame_number'])
            string_timestamp = str(research_data['timestamp'])

            profile = self.research_metrics['operator_profiles'][operator_id]
            profile['hand_usage'][research_data['hand_usage']] += 1
            profile['engagement'][research_data['engagement_level']] += 1
            profile['actions_performed'].add(research_data['action_name'])

            if 'performance_scores' not in profile:
                profile['performance_scores'] = []
            profile['performance_scores'].append(float(research_data['performance_metrics']['performance_score']))

            self.research_metrics['engagement_data'].append({
                'action': research_data['action_name'],
                'sub_assembly': research_data['sub_assembly'],
                'timestamp': numeric_timestamp, 
                'timestamp_str': string_timestamp, 
                'level': research_data['engagement_level'],
                'operator_id': operator_id
            })
            
            self.research_metrics['hand_use_data'].append({
                'action': research_data['action_name'],
                'sub_assembly': research_data['sub_assembly'],
                'timestamp': numeric_timestamp,  
                'timestamp_str': string_timestamp, 
                'pattern': research_data['hand_usage'],
                'operator_id': operator_id
            })
            
            self.research_metrics['safety_data'].append({
                'action': research_data['action_name'],
                'sub_assembly': research_data['sub_assembly'],
                'timestamp': numeric_timestamp, 
                'timestamp_str': string_timestamp, 
                'assessment': research_data['safety_assessment'],
                'operator_id': operator_id
            })

            robot_data = {
                'operator_id': operator_id,
                'action_name': research_data['action_name'],
                'action_category': str(self._categorize_action(research_data['action_name'])),
                'timestamp': numeric_timestamp, 
                'timestamp_str': string_timestamp, 
                'robot_present': bool(research_data.get('robot_present', False)),
                'assistance_level': str(research_data.get('assistance_level', 'NONE')),
                'collaboration_score': float(research_data.get('collaboration_score', 0.0)),
                'handover_recommended': bool(research_data['handover_decision'].get('handover_recommended', False)),
                'performance_score': float(research_data.get('performance_metrics', {}).get('performance_score', 0))
            }
      
            if 'robot_data' not in self.research_metrics:
                self.research_metrics['robot_data'] = []
            
            self.research_metrics['robot_data'].append(robot_data)

            if research_data['task_progress'] >= 95:
                self.research_metrics['action_times'][research_data['action_name']].append(numeric_timestamp)
                
                if 'completion_times' not in profile:
                    profile['completion_times'] = []
                
                profile['completion_times'].append({
                    'action': research_data['action_name'],
                    'time': numeric_timestamp,  # NUMERIC value
                    'progress': research_data['task_progress'],
                    'timestamp_str': string_timestamp
                })

            if self.current_sub_assembly:
                self.sub_assembly_progress[self.current_sub_assembly] = int(research_data['task_progress'])

        except Exception as e:
            print(f"Error updating research metrics: {e}")
            import traceback
            traceback.print_exc()






    def generate_three_layer_analysis(self):

        try:
            # Initialize three-layer analyzer
            three_layer_analyzer = ThreeLayerPremAnalyzer(
                research_metrics=self.research_metrics,
                output_dir=os.path.join(self.config['output_dir'], "three_layer_analysis")
            )

            report = three_layer_analyzer.generate_comprehensive_report()
            
            print("✓ Three-layer analysis completed successfully!")
            return report
            
        except Exception as e:
            print(f"Error in three-layer analysis: {e}")
            import traceback
            traceback.print_exc()
            return None



    def _categorize_action(self, action_name):
        if not isinstance(action_name, str):
            return 'unknown'
        
        action_name_lower = action_name.lower()
        if any(word in action_name_lower for word in ['screw', 'fasten', 'tighten']):
            return 'fastening'
        elif any(word in action_name_lower for word in ['align', 'position', 'adjust']):
            return 'alignment' 
        elif any(word in action_name_lower for word in ['take', 'pick', 'grab']):
            return 'taking'
        elif any(word in action_name_lower for word in ['put', 'place', 'attach']):
            return 'placing'
        elif any(word in action_name_lower for word in ['check', 'verify', 'inspect']):
            return 'inspection'
        else:
            return 'general'



    def _print_learned_insights(self):
        ranking = self.sequence_analyzer.get_performance_ranking()[:5]
        
        print("\n Enhanced Performance Insights:")
        print("-" * 60)
        print(f"Unique actions discovered: {len(self.sequence_analyzer.unique_actions)}")
        print(f"Best performances recorded: {len(self.sequence_analyzer.action_base['best_performance'])}")
        print(f"Live actions tracked: {len(self.sequence_analyzer.action_base['live_actions'])}")
        
        if ranking:
            print(f"\n Top Performance Improvement Opportunities:")
            for i, item in enumerate(ranking, 1):
                print(f"  {i}. {item['action']}:")
                print(f"     Average: {item['average_time']:.1f}s, Best: {item['best_time']:.1f}s")
                print(f"     Improvement: {item['improvement_percentage']:.1f}% (by {item['best_operator']})")
        
        # Show some best performances
        best_actions = list(self.sequence_analyzer.action_base['best_performance'].items())[:3]
        if best_actions:
            print(f"\n Best Performances:")
            for action, data in best_actions:
                print(f"  - {action}: {data['best_time']:.1f}s by {data['best_operator']}")

    def get_performance_recommendations(self, operator_id: str) -> List[Dict]:
        recommendations = []
        for action_name, live_data in self.sequence_analyzer.action_base['live_actions'].items():
            if operator_id in live_data['operator_stats']:
                op_stats = live_data['operator_stats'][operator_id]
                best_time = live_data['best_time']
                op_avg_time = op_stats['average_time']
                
                if op_avg_time > best_time * 1.1:
                    improvement = ((op_avg_time - best_time) / op_avg_time) * 100
                    best_operator = self.sequence_analyzer.action_base['best_performance'][action_name]['best_operator']
                    
                    recommendations.append({
                        'action': action_name,
                        'current_time': op_avg_time,
                        'best_time': best_time,
                        'improvement_potential': improvement,
                        'best_operator': best_operator,
                        'suggestion': f"Learn from {best_operator}'s technique"
                    })
        
        return sorted(recommendations, key=lambda x: x['improvement_potential'], reverse=True)

    def generate_prem_analysis_report(self):
        try:
            # Check if we have sequence data
            if not hasattr(self, 'sequence_analyzer') or not self.sequence_analyzer.operator_sequences:
                print("No sequence data available for prem analysis")
                return None

            if not self.research_metrics or not self.research_metrics.get('operator_profiles'):
                print("No research metrics available for prem analysis")
                return None
            
            print("Generating prem micro-analysis report...")

            action_sequences = dict(self.sequence_analyzer.operator_sequences)

            engagement_data = self._extract_prem_engagement_data(self.research_metrics)
            hand_usage_data = self._extract_prem_hand_usage_data(self.research_metrics)
            progress_data = self._extract_prem_progress_data(self.research_metrics)

            prem_analyzer = PremAnalyzer(
                engagement_data=engagement_data,
                hand_usage_data=hand_usage_data,
                progress_data=progress_data,
                action_sequences=action_sequences,
                output_dir=os.path.join(self.config['output_dir'], "prem_analysis_results")
            )

            prem_report = prem_analyzer.generate_prem_analysis(self.research_metrics)
            
            if prem_report:
                print(f" Prem analysis report generated: {prem_report.get('title', 'Unknown')}")

                if 'csv_reports' in prem_report:
                    print("✓ Generated CSV reports:")
                    for report_type, path in prem_report['csv_reports'].items():
                        print(f"  - {report_type}: {os.path.basename(path)}")
                
                if 'summary_statistics' in prem_report:
                    stats = prem_report['summary_statistics']
                    print(f" Analyzed {stats.get('total_primitives_analyzed', 0)} primitives")
                    print(f" Found {stats.get('unique_action_primitives', 0)} unique actions")
            
            return prem_report
            
        except Exception as e:
            print(f"Error generating prem analysis report: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_prem_engagement_data(self, research_metrics: Dict) -> Dict:
        engagement_data = {}
        for operator_id, profile in research_metrics.get('operator_profiles', {}).items():
            engagement_data[operator_id] = profile.get('engagement', {})
        return engagement_data

    def _extract_prem_hand_usage_data(self, research_metrics: Dict) -> Dict:
        hand_usage_data = {}
        for operator_id, profile in research_metrics.get('operator_profiles', {}).items():
            hand_usage_data[operator_id] = profile.get('hand_usage', {})
        return hand_usage_data

    def _extract_prem_progress_data(self, research_metrics: Dict) -> Dict:
        progress_data = {}
        for operator_id, profile in research_metrics.get('operator_profiles', {}).items():
            progress_data[operator_id] = {
                'completion_times': profile.get('completion_times', []),
                'performance_scores': profile.get('performance_scores', [])
            }
        return progress_data

    def generate_sequence_analysis(self, operator_id: str = None):
        if operator_id:
            self.sequence_analyzer.generate_sequence_visualization(
                operator_id, self.config['output_dir']
            )
            self.sequence_analyzer.generate_analysis_report(
                operator_id, self.config['output_dir']
            )
        else:
            for op in self.sequence_analyzer.operator_sequences.keys():
                self.sequence_analyzer.generate_sequence_visualization(
                    op, self.config['output_dir']
                )
                self.sequence_analyzer.generate_analysis_report(
                    op, self.config['output_dir']
                )
            self.sequence_analyzer.analyze_all_operators(self.config['output_dir'])




    def predict_next_actions(self, previous_actions: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        if not previous_actions:
            return [('starting_analysis', 1.0)]
        last_action = previous_actions[-1]
        return self.sequence_learner.predict_next_actions(previous_actions, top_n)




    def analyze_single_frame(self, frame_data: Dict) -> Optional[Dict]:
        try:
            frame = frame_data['frame']
            frame_number = frame_data['frame_number']
            timestamp = frame_data['timestamp']
            action_name = frame_data['action_name']
            detected_objects = frame_data['detected_objects']
            safety_status = frame_data['safety_status']
            operator_id = frame_data['operator_id']

            if not isinstance(frame, np.ndarray) or frame.size == 0:
                return None

            context = self.get_context_string()
            prompt = self.llava_analyzer.generate_prompt(action_name, detected_objects, safety_status, context)
            analysis = self.llava_analyzer.analyze_frame(frame, prompt)

            engagement_level = extract_engagement_level(analysis)
            hand_usage = extract_hand_usage(analysis, detected_objects)
            safety_assessment = extract_safety_assessment(analysis, detected_objects)
            task_progress = self._calculate_hierarchical_progress(action_name, frame_number, frame_data['total_frames'], analysis)

            try:
                performance_metrics = self.llava_analyzer.assess_operator_performance(analysis, detected_objects, engagement_level, hand_usage)
            except Exception as e:
                print(f"Error in performance assessment: {e}")
                performance_metrics = {
                    'fatigue_detected': False,
                    'safety_concerns': False,
                    'inefficient_technique': False,
                    'difficulty_with_task': False,
                    'performance_score': 80,
                    'confidence_level': 0.8,
                    'improvement_suggestions': []
                }

            try:
                handover_decision = self.llava_analyzer.assess_handover_need(performance_metrics, action_name, engagement_level, analysis)
            except Exception as e:
                print(f"Error in handover assessment: {e}")
                handover_decision = {
                    'handover_recommended': False,
                    'reasons': [],
                    'urgency': 'LOW',
                    'robot_capability_score': 0.0,
                    'expected_improvement': 0.0
                }

            robot_analysis = self.llava_analyzer.analyze_robot_presence(frame_data, analysis, handover_decision)

            objs_for_memory = (detected_objects.get('assembly_components', []) + 
                              detected_objects.get('tools', []))
            
            try:
                self.object_memory.extend(objs_for_memory)
            except Exception:
                pass

            if not hasattr(self, 'action_memory'):
                self.action_memory = deque(maxlen=self.config.get('memory_size', 15))

            if not self.action_memory or self.action_memory[-1] != action_name:
                self.action_memory.append(action_name)

            previous_actions = list(self.action_memory)[-2:] if len(self.action_memory) >= 2 else []
            predicted_next = []

            if len(self.action_memory) >= 2:
                predictions = self.sequence_analyzer.predict_next_actions(
                    operator_id, previous_actions, 5
                )
                
                predicted_next = [(pred[0], f"{pred[1]*100:.1f}%") for pred in predictions if pred[1] > 0.1]

                if not predicted_next and previous_actions and hasattr(self, 'sequence_learner'):
                    try:
                        predictions = self.sequence_learner.predict_next_actions(previous_actions, 3)
                        predicted_next = [(pred[0], f"{pred[1]*100:.1f}%") for pred in predictions]
                    except Exception as e:
                        print(f"Sequence learner prediction error: {e}")
            else:
                if hasattr(self, 'sequence_learner'):
                    try:
                        predictions = self.sequence_learner.predict_next_actions([action_name], 3)
                        predicted_next = [(pred[0], f"{pred[1]*100:.1f}%") for pred in predictions]
                    except Exception as e:
                        print(f"Sequence learner prediction error: {e}")

            # Package results
            research_data = {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'action_name': action_name,
                'engagement_level': engagement_level,
                'hand_usage': hand_usage,
                'safety_assessment': safety_assessment,
                'task_progress': task_progress,
                'detected_objects': detected_objects,
                'raw_analysis': analysis,
                'performance_metrics': performance_metrics,
                'handover_decision': handover_decision,
                'robot_present': robot_analysis['robot_present'],
                'assistance_level': robot_analysis['assistance_level'],
                'collaboration_score': robot_analysis['collaboration_score'],
                'previous_actions': previous_actions,
                'predicted_next': predicted_next
            }

            self._update_research_metrics(research_data, operator_id)

            self._log_analysis_results(research_data, frame_number)

            return research_data

        except Exception as e:
            print(f"Error analyzing frame: {e}")
            import traceback
            traceback.print_exc()
            return None



    def _log_analysis_results(self, research_data: Dict, frame_number: int):
        current_video = "Unknown Video"
        if hasattr(self, 'current_video_path'):
            current_video = os.path.basename(self.current_video_path)
        
        print(f"\nAction Videos Process: {current_video}")
        print(f"Frame {frame_number} - Sequence Context:")

        formatted_previous_actions = []
        for action in research_data.get('previous_actions', ['N/A']):
            action_weight = self._calculate_action_completion_time(action)
            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_previous_actions.append(
                f"('{action}', 'weight: {action_weight:.2f}s', 'timestamp: {current_timestamp}')"
            )
        
        print(f"Previous Actions: [{', '.join(formatted_previous_actions)}]")

        if 'predicted_next' in research_data and research_data['predicted_next']:
            print(f"Predicted top-3: {research_data['predicted_next']}")
        else:
            print("Predicted top-3: No predictions available")

        current_action = research_data['action_name']
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Actual Action: [{current_action}, timestamp: {current_timestamp}]")

        print(f"Engagement: {research_data['engagement_level']}")
        print(f"Hand Use: {research_data['hand_usage']}")
        print(f"Safety: {research_data['safety_assessment']}")
        print(f"Progress: {research_data['task_progress']}%")
        print(f"Performance: {research_data['performance_metrics']['performance_score']}/100")
        
        if research_data['handover_decision']['handover_recommended']:
            print("ROBOT HANDOVER RECOMMENDED!")
            reasons = research_data['handover_decision'].get('reasons', [])
            if not isinstance(reasons, list):
                reasons = [str(reasons)] if reasons else []
            
            print(f"   Reasons: {', '.join(reasons)}")
            print(f"   Urgency: {research_data['handover_decision'].get('urgency', 'LOW')}")
            print(f"   Expected improvement: {research_data['handover_decision'].get('expected_improvement', 0.0):.1f}%")


    def _calculate_action_completion_time(self, action_name: str) -> float:

        if hasattr(self, 'sequence_analyzer'):
            try:
                return self.sequence_analyzer.get_action_time_estimate(action_name)
            except:
                pass

        action_type = self.sequence_analyzer.extract_action_components(action_name)['action_type']
        time_mapping = {
            'take': 3.0, 'put': 2.0, 'align': 8.0, 'fasten': 5.0, 'inspect': 3.0
        }
        
        return time_mapping.get(action_type, 5.0) + random.uniform(-1.0, 1.0)



    def _simulate_robot_performance(self, human_performance: Dict, action_name: str) -> Dict:
        base_score = 85

        if any(x in action_name for x in ['screw', 'tighten', 'fasten']):
            base_score = 95 
        elif any(x in action_name for x in ['align', 'precision', 'measure']):
            base_score = 90 
        elif any(x in action_name for x in ['delicate', 'sensitive', 'fragile']):
            base_score = 80 

        variability = np.random.normal(0, 5)
        robot_score = max(60, min(100, base_score + variability))
        
        return {
            'performance_score': robot_score,
            'completion_time': human_performance.get('completion_time', 0) * 0.7,
            'consistency_score': 90,
            'error_rate': 2,
            'safety_score': 95
        }

    def _determine_task_handover(self, performance_metrics: Dict, analysis: str, action_name: str, frame_number: int) -> Dict:
        handover_decision = {
            'handover_recommended': performance_metrics['recommend_robot_handover'],
            'reason': [],
            'human_performance_score': performance_metrics['performance_score'],
            'frame_number': frame_number,
            'action_name': action_name
        }

        if performance_metrics['fatigue_detected']:
            handover_decision['reason'].append('Operator fatigue detected')
        if performance_metrics['difficulty_with_task']:
            handover_decision['reason'].append('Operator struggling with task')
        if performance_metrics['safety_concerns']:
            handover_decision['reason'].append('Safety concerns')
        if performance_metrics['inefficient_technique']:
            handover_decision['reason'].append('Inefficient technique')

        if handover_decision['handover_recommended']:
            handover_decision['robot_performance'] = self._simulate_robot_performance(
                performance_metrics, action_name
            )
            handover_decision['handover_message'] = (
                f"Task '{action_name}' transferred to robot at frame {frame_number}. "
                f"Reason: {', '.join(handover_decision['reason'])}. "
                f"Human performance score: {performance_metrics['performance_score']}/100"
            )
            print(f"ROBOT HANDOVER: {handover_decision['handover_message']}")
        
        return handover_decision



    def _calculate_collaboration_score(self, assistance_level: str, analysis: str) -> float:
        assistance_scores = {
            "ACTIVE_ASSISTANCE": 0.9,
            "MONITORING": 0.6, 
            "PRESENT": 0.3,
            "NONE": 0.0
        }
        
        base_score = assistance_scores.get(assistance_level, 0.0)

        analysis_lower = analysis.lower()

        if any(word in analysis_lower for word in ['handing', 'passing', 'assisting', 'helping', 'together']):
            base_score += 0.1
        elif any(word in analysis_lower for word in ['waiting', 'delayed', 'slow', 'hesitat']):
            base_score -= 0.1

        return max(0.0, min(1.0, base_score))



    def _analyze_robot_presence(self, frame: np.ndarray, analysis: str, detected_objects: Dict) -> Dict:
        robot_detected = any(obj in detected_objects.get('tools', []) 
                           for obj in ['robotic_arm', 'cobot', 'robot_gripper'])
        
        assistance_level = "NONE"
        if robot_detected:
            a = analysis.lower()
            if any(word in a for word in ['handing', 'passing', 'assisting', 'helping']):
                assistance_level = "ACTIVE_ASSISTANCE"
            elif any(word in a for word in ['waiting', 'standing', 'idle', 'monitoring']):
                assistance_level = "MONITORING"
            else:
                assistance_level = "PRESENT"
        
        return {
            'robot_present': robot_detected,
            'assistance_level': assistance_level,
            'collaboration_score': self._calculate_collaboration_score(assistance_level, analysis)
        }


    def analyze_frame_batch(self, frames_batch: List[Dict]) -> List[Dict]:
        results = []
        for frame_data in frames_batch:
            result = self.analyze_single_frame(frame_data)
            if result:
                results.append(result)

                operator_id = frame_data.get('operator_id', 'unknown')
                action_name = result.get('action_name')
                if action_name and hasattr(self, 'sequence_analyzer'):
                    try:
                        self.sequence_analyzer.add_operator_sequence(operator_id, action_name)
                        if hasattr(self, 'sequence_learner'):
                            self.sequence_learner.add_sequence(operator_id, action_name)
                    except Exception as e:
                        print(f"Error adding to sequence analyzers: {e}")       
        return sorted(results, key=lambda x: x['frame_number'])

    def analyze_assembly(self, c1_video_path: str) -> Tuple[Optional[List], Optional[Dict]]:
        self.sub_assembly_progress = {}
        self.current_sub_assembly = None
        self.object_memory.clear()
        self.current_video_path = c1_video_path

        action_name = self.extract_action_name(c1_video_path, "c1")
        operator_id = self.extract_operator_id(c1_video_path)
        print(f"Action: {action_name}, Operator: {operator_id}")
        self.video_sequences[operator_id].append(action_name)
        c2_video_path = self.find_c2_video_for_action(action_name, operator_id)
        if c2_video_path:
            print(f"Analyzing C2 video for object detection: {os.path.basename(c2_video_path)}")
            detection_results = self.object_detector.detect_objects_from_video(
                c2_video_path, self.config.get('yolo_fps', 1)
            )
        else:
            print("No C2 video found for object detection")
            detection_results = {
                'assembly_components': [], 'tools': [], 'hands': [],
                'safety_hazards': [], 'unknown_objects': [], 'safety_status': 'UNKNOWN'
            }

        detected_objects = {k: v for k, v in detection_results.items()
                            if k in ['assembly_components', 'tools', 'hands', 'safety_hazards', 'unknown_objects']}
        safety_status = detection_results.get('safety_status', 'UNKNOWN')

        cap = cv2.VideoCapture(c1_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {c1_video_path}")
            return None, None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        duration = total_frames / max(fps, 1)

        target_frames = [0, max(total_frames - 1, 0)]
        for t in range(1, int(duration)):
            pos = int(min(t * fps, total_frames - 1))
            if pos not in target_frames:
                target_frames.append(pos)
        target_frames = sorted(set(target_frames))

        print(f"Analyzing at {self.config['llava_fps']} FPS")
        print(f"Target frames: {len(target_frames)}")

        analysis_results = []
        frame_batch = []

        for frame_number in target_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp = frame_number / max(fps, 1)
            current_frame_data = {
                'frame': frame.copy(),
                'frame_number': frame_number,
                'timestamp': timestamp,
                'total_frames': total_frames,
                'action_name': action_name,
                'detected_objects': detected_objects,
                'safety_status': safety_status,
                'operator_id': operator_id
            }

            frame_batch.append(current_frame_data)

            if len(frame_batch) >= self.config.get('batch_size', 4):
                batch_results = self.analyze_frame_batch(frame_batch)
                analysis_results.extend(batch_results)
                frame_batch = []

        if frame_batch:
            batch_results = self.analyze_frame_batch(frame_batch)
            analysis_results.extend(batch_results)

        cap.release()

        if not analysis_results:
            return None, None

        summary = self._generate_research_summary(analysis_results, duration, action_name, operator_id)

        temporal_segments = []
        behavioral_patterns = []
        collaboration_efficiency = 0.0
        handover_effectiveness = 0.0

        try:
            temporal_segments = self.temporal_segmenter.segment_assembly_process(analysis_results)
            behavioral_patterns = self.behavioral_segmenter.identify_behavioral_patterns(analysis_results)

            if hasattr(self.temporal_segmenter, '_calculate_collaboration_efficiency'):
                collaboration_efficiency = self.temporal_segmenter._calculate_collaboration_efficiency(
                    temporal_segments, behavioral_patterns
                )

            handover_effectiveness = self._calculate_handover_effectiveness(analysis_results)
        except Exception as e:
            print(f"Error in segmentation analysis: {e}")

        summary.update({
            'temporal_segments': [segment.__dict__ for segment in temporal_segments] if temporal_segments else [],
            'behavioral_patterns': [pattern.__dict__ for pattern in behavioral_patterns] if behavioral_patterns else [],
            'collaboration_efficiency': collaboration_efficiency,
            'handover_effectiveness': handover_effectiveness
        })

        self.collaboration_metrics.update({
            'temporal_segments': temporal_segments,
            'behavioral_patterns': behavioral_patterns,
            'collaboration_efficiency': collaboration_efficiency,
            'handover_effectiveness': handover_effectiveness
        })

        self._save_research_results(analysis_results, summary, c1_video_path)

        return analysis_results, summary


    def _calculate_handover_effectiveness(self, analysis_results: List[Dict]) -> float:
        if not analysis_results:
            return 0.0
        
        handover_events = 0
        successful_handovers = 0
        
        for result in analysis_results:
            if result.get('handover_decision', {}).get('handover_recommended', False):
                handover_events += 1
                robot_perf = result.get('robot_analysis', {}).get('performance_score', 0)
                human_perf = result.get('performance_metrics', {}).get('performance_score', 0)
                if robot_perf > human_perf:
                    successful_handovers += 1
        
        if handover_events == 0:
            return 0.0
        
        return successful_handovers / handover_events


    def _generate_research_summary(self, analysis_results: List[Dict], duration: float, action_name: str, operator_id: str) -> Dict:
        engagement_counts = Counter()
        hand_usage_counts = Counter()
        safety_counts = Counter()
        progress_values = []

        for r in analysis_results:
            engagement_counts[r['engagement_level']] += 1
            hand_usage_counts[r['hand_usage']] += 1
            safety_counts[r['safety_assessment']] += 1
            progress_values.append(min(100, r['task_progress']))

        avg_progress = float(sum(progress_values) / len(progress_values)) if progress_values else 0.0
        completion_time = None
        
        if self.research_metrics['action_times'].get(action_name):
            completion_time = max(self.research_metrics['action_times'][action_name])

        return {
            'action_name': action_name,
            'operator_id': operator_id,
            'video_duration': float(duration),
            'completion_time': float(completion_time) if completion_time is not None else None,
            'engagement_distribution': dict(engagement_counts),
            'hand_usage_distribution': dict(hand_usage_counts),
            'safety_distribution': dict(safety_counts),
            'average_progress': float(avg_progress),
            'max_progress': int(max(progress_values) if progress_values else 0),
            'min_progress': int(min(progress_values) if progress_values else 0),
            'detected_objects': analysis_results[0]['detected_objects'] if analysis_results else {},
            'sub_assembly': self.current_sub_assembly or 'main_assembly'
        }

    def _save_research_results(self, analysis_results: List[Dict], summary: Dict, video_path: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        action_name = summary['action_name']
        operator_id = summary['operator_id']
        filename = f"{operator_id}_{action_name}_{timestamp}"

        with open(f"{self.config['output_dir']}/{filename}_detailed_report.txt", 'w') as f:
            f.write(f"COMPREHENSIVE HRI RESEARCH ANALYSIS REPORT\n{'='*80}\n")
            f.write(f"Action: {action_name}\nOperator: {operator_id}\n")
            f.write(f"Video: {os.path.basename(video_path)}\n")
            f.write(f"Duration: {summary['video_duration']:.1f}s\n")
            
            if summary['completion_time']:
                f.write(f"Completion Time: {summary['completion_time']:.1f}s\n")
            
            f.write(f"Total frames analyzed: {len(analysis_results)}\n\n")
            f.write("METRICS SUMMARY:\n")
            f.write(f"Engagement: {summary['engagement_distribution']}\n")
            f.write(f"Hand Usage: {summary['hand_usage_distribution']}\n")
            f.write(f"Safety: {summary['safety_distribution']}\n")
            f.write(f"Progress: Avg {summary['average_progress']:.1f}%\n")

        print(f"Research results saved for {action_name}")


    def analyze_all_operators(self) -> List:
        dataset_root = self.config['dataset_root_path']
        operators = self.config['operators']
        
        if not operators:
            if os.path.isdir(dataset_root):
                operators = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
                self.config['operators'] = operators

        all_results = []
        existing_sequences = self._load_efficient_sequences()
        if existing_sequences:
            print("Using pre-loaded efficient sequences")

        for op in operators:
            c1_dir = os.path.join(dataset_root, op, "c1_clips")
            if not os.path.isdir(c1_dir):
                continue

            for fn in sorted(os.listdir(c1_dir)):
                if fn.endswith(".mp4"):
                    video_path = os.path.join(c1_dir, fn)
                    print(f"\nProcessing: {op} - {fn}")
                    
                    try:
                        results, summary = self.analyze_assembly(video_path)
                        if results and summary:
                            all_results.append((f"{op}:{summary['action_name']}", summary))
                            print(f"✓ Completed: {fn}")
                        else:
                            print(f"✗ Failed: {fn}")
                    except Exception as e:
                        print(f"✗ Error: {fn} - {e}")

        if any(self.sequence_analyzer.operator_sequences.values()):
            print("\n" + "="*60)
            print("GENERATING SEQUENCE ANALYSIS")
            print("="*60)
            self.generate_sequence_analysis()

            efficient_file = self._save_efficient_sequences()
            if efficient_file:
                print(f"✓ Efficient sequences analysis completed and saved to: {efficient_file}")

                if hasattr(self, 'sequence_learner'):
                    self.sequence_learner._save_sequences()
            else:
                print("✗ Failed to save efficient sequences")

        if all_results:
            report_path = self.report_generator.generate_comprehensive_report(all_results, self.research_metrics)
            print(f"\nComprehensive report generated: {report_path}")
            
            operator_report = self.report_generator.generate_operator_comparison_report(self.research_metrics)
            print(f"Operator comparison report: {operator_report}")

        print("\n" + "="*60)
        print("GENERATING PREM MICRO-ANALYSIS")
        print("="*60)
        self.generate_prem_analysis_report()

        print("\n" + "="*60)
        print("GENERATING THREE-LAYER ANALYSIS")
        print("="*60)

        three_layer_report = self.generate_three_layer_analysis()
        if three_layer_report:
            print("✓ Three-layer analysis integrated successfully")


        return all_results



    def __del__(self):
        try:
            if hasattr(self, 'llava_analyzer') and hasattr(self.llava_analyzer, 'gpu_id'):
                with torch.cuda.device(f"cuda:{self.llava_analyzer.gpu_id}"):
                    torch.cuda.empty_cache()
        except Exception:
            pass

if __name__ == "__main__":
    try:
        print_gpu_status()

        analyzer = EnhancedVideoAnalyzer(config_path="analysis_config.json")
        print("Starting multi-operator analysis...")
        results = analyzer.analyze_all_operators()
        
        if results:
            print(f"\nAnalysis completed! Processed {len(results)} actions.")
            
            # Final save of efficient sequences
            efficient_file = analyzer._save_efficient_sequences()
            if efficient_file:
                print(f"Final efficient sequences saved to: {efficient_file}")

            print("\n" + "="*60)
            print("STARTING PREM MICRO-ANALYSIS")
            print("="*60)
            
            prem_report = analyzer.generate_prem_analysis_report()
            if prem_report:
                print(f"Prem analysis completed successfully!")
                print(f"Title: {prem_report.get('title', 'Unknown')}")
                print(f"Results saved to: prem_analysis_results/")
            else:
                print("Prem analysis failed or no data available")

        else:
            print("No actions processed.")

        # Cleanup
        clear_gpu_cache()

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
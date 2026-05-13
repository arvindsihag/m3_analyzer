import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
import torch
import gc
import numpy as np
import re
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import cv2
from typing import Dict, List, Optional, Tuple
import time
from collections import Counter
import random
from gpu_config import gpu_config
from robot_simulator import RobotSimulator
from performance_analyzer import PerformanceAnalyzer

class LlavaAnalyzer:
    def __init__(self, model_path: str = '/workspace/raid/stu-1/BLIP/model_lm/llava-hf/llava-1.5-7b-hf', 
                 gpu_id: Optional[int] = None, use_fp16: bool = True, use_amp: bool = True):
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self.use_amp = use_amp
        

        # Use centralized GPU config
        self.num_gpus = gpu_config.get_num_gpus()
        
        if self.num_gpus > 0:
            self.device = "cuda"
            
            # Load processor
            self.processor = LlavaProcessor.from_pretrained(model_path)
            if self.use_fp16:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            
            print(f"LLaVA model distributed across {self.num_gpus} GPUs")
            
        else:
            self.device = "cpu"
            self.processor = LlavaProcessor.from_pretrained(model_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(model_path)
        
        self.model.eval()
        print(f"LLaVA model loaded successfully on {self.num_gpus} GPUs")


        self.robot_available = True 
        self.robot_assisted_actions = [
            'screw', 'tighten', 'align', 'connect', 'assemble', 'install',
            'position', 'fasten', 'secure', 'calibrate'
        ]
        
        # Performance thresholds for robot handover
        self.handover_thresholds = {
            'fatigue': 45,          # Performance score below this triggers handover due to fatigue
            'safety': 50,           # Safety concerns threshold
            'inefficiency': 50,     # Inefficient technique threshold
            'difficulty': 45,       # Task difficulty threshold
            'engagement': 'IDLE'    # Engagement level that triggers handover
        }
        
        # Performance degradation patterns
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
            'unable to continue', 'repeated failures'
        ]



    def generate_prompt(self, action_name: str, detected_objects: Dict, safety_status: str, context_string: str = "No objects detected recently") -> str:

        if '_' in action_name:
            action_verb = action_name.split('_')[0]
        else:
            action_verb = action_name
        
        action_descriptions = {
            "take": "retrieving components", "put": "placing components",
            "align": "aligning components", "plug": "connecting or inserting components",
            "screw": "fastening with screws", "tighten": "securing with force or tool",
            "connect": "making electrical or mechanical connections",
            "check": "inspecting the assembly", "adjust": "making adjustments",
            "change": "changing tool bits or orientation"
        }
        action_desc = action_descriptions.get(action_verb, "performing assembly task")
        
        hand_context = ""
        if detected_objects.get('hands'):
            hand_count = len(detected_objects['hands'])
            hand_context = f"Hands detected: {hand_count} hand{'s' if hand_count != 1 else ''}"
        
        object_context = []
        if detected_objects.get('assembly_components'):
            object_context.append(f"Components: {', '.join(detected_objects['assembly_components'])}")
        if detected_objects.get('tools'):
            object_context.append(f"Tools: {', '.join(detected_objects['tools'])}")
        if hand_context:
            object_context.append(hand_context)

        safety_context = f"Workspace safety: {safety_status}"
        if detected_objects.get('safety_hazards'):
            safety_context += f" - Hazards detected: {', '.join(detected_objects['safety_hazards'])}"
        if detected_objects.get('unknown_objects'):
            safety_context += f" - Unknown objects: {', '.join(detected_objects['unknown_objects'])}"

        robot_context = "ROBOT ASSISTANT: Present and monitoring the assembly process. "
        if action_name in self.robot_assisted_actions:
            robot_context += "Robot is actively assisting with this task. "

        prompt = f"""USER: <image>
Analyze this assembly workstation where the operator is {action_desc} for {action_name.replace('_', ' ')}.
{robot_context}

CONTEXT:
{context_string}
{'; '.join(object_context) if object_context else 'No specific components detected'}
{safety_context}

                    PAY SPECIAL ATTENTION TO HAND USAGE. Be very specific about:
                    - Which hand is holding tools or components (LEFT, RIGHT, or BOTH)
                    - The exact hand performing the main action
                    - Whether hands are visible and actively engaged
                    - If you can't see hands clearly, say so explicitly

                    Focus your analysis on these specific metrics for human-robot interaction research:
                    1. OPERATOR ENGAGEMENT LEVEL: 
                       - HIGHLY_ENGAGED: Actively manipulating components with focused attention
                       - ENGAGED: Performing task with normal attention
                       - PREPARING: Setting up or organizing for the next action
                       - IDLE: Not actively engaged in task (waiting, thinking)
                       - DISENGAGED: Distracted, not paying attention to task

                    2. HAND USAGE PATTERN (BE SPECIFIC):
                       - LEFT: Clearly using left hand for the main action
                       - RIGHT: Clearly using right hand for the main action  
                       - BOTH: Clearly using both hands cooperatively
                       - NONE: No hands actively engaged in the task
                       - UNCERTAIN: Hand usage not clearly visible

                    3. SAFETY ASSESSMENT:
                       - SAFE_WORKSPACE: No visible risks
                       - LOW_RISK: Minor potential risks
                       - MEDIUM_RISK: Noticeable safety concerns
                       - HIGH_RISK: Significant safety issues
                       - HAZARD_DETECTED: Immediate danger present

                    4. TASK PROGRESS: Estimate completion percentage (0-100%) of {action_name.replace('_', ' ')}

                    5. PERFORMANCE INDICATORS: Note any signs of fatigue, difficulty, or inefficiency

                    Provide specific, quantitative observations for each metric. Be explicit about hand usage.
                    ASSISTANT:"""
        return prompt

    def analyze_frame(self, frame: np.ndarray, prompt: str) -> str:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Process inputs
            inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt", padding=True)

            moved_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(self.device, non_blocking=True)
                    if v.is_floating_point():
                        target_dtype = torch.float16 if self.use_fp16 else torch.float32
                        v = v.to(dtype=target_dtype)
                moved_inputs[k] = v

            # Generate response
            with torch.cuda.device(self.device):
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    with torch.inference_mode():
                        output = self.model.generate(
                            **moved_inputs,
                            max_new_tokens=128,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            num_return_sequences=1,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                        )

            generated_tokens = output[0]
            if generated_tokens.dim() > 1:
                generated_tokens = generated_tokens[0]
                
            analysis = self.processor.decode(generated_tokens, skip_special_tokens=True)
            
            if "ASSISTANT:" in analysis:
                analysis = analysis.split("ASSISTANT:")[-1].strip()

            return analysis

        except Exception as e:
            print(f"Error in frame analysis: {e}")
            return f"Analysis error: {str(e)}"
        finally:
            self._cleanup_memory()

    def _cleanup_memory(self):
        try:
            gc.collect()
            if self.device.startswith('cuda'):
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        except Exception as e:
            print(f"Memory cleanup error: {e}")

    def __del__(self):
        try:
            if hasattr(self, 'gpu_id') and self.gpu_id is not None:
                with torch.cuda.device(f"cuda:{self.gpu_id}"):
                    torch.cuda.empty_cache()
        except Exception:
            pass


    def _pattern_match(self, text: str, patterns: List[str]) -> float:
        """Calculate pattern match score (0-1)"""
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        return min(1.0, matches / len(patterns)) if patterns else 0.0

    def _assess_task_complexity(self, action_name: str) -> float:
        """Assess task complexity (between 0-1, where 1 is most complex)"""
        complexity_scores = {
            'align': 0.8,
            'screw': 0.7, 
            'tighten': 0.7, 
            'connect': 0.6,
            'assemble': 0.5, 
            'install': 0.5, 
            'position': 0.4, 
            'place': 0.3,
            'take': 0.2, 
            'put': 0.2, 
            'check': 0.3, 
            'adjust': 0.4
        }
        
        # Find the best matching action verb
        for verb, score in complexity_scores.items():
            if verb in action_name.lower():
                return score
        
        return 0.5 

    def assess_operator_performance(self, analysis: str, detected_objects: Dict, engagement_level: str, hand_usage: str, action_name: str) -> Dict:
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
        
        # Detect fatigue
        fatigue_score = self._pattern_match(a, self.fatigue_patterns)
        if fatigue_score > 0.3:
            metrics['fatigue_detected'] = True
            metrics['performance_score'] -= fatigue_score * 20
        
        # Detect safety issues
        safety_score = self._pattern_match(a, self.safety_patterns)
        if safety_score > 0.4:
            metrics['safety_concerns'] = True
            metrics['performance_score'] -= safety_score * 25
        
        # Detect inefficiency
        inefficiency_score = self._pattern_match(a, self.inefficiency_patterns)
        if inefficiency_score > 0.3:
            metrics['inefficient_technique'] = True
            metrics['performance_score'] -= inefficiency_score * 15
        
        # Detect difficulty
        difficulty_score = self._pattern_match(a, self.difficulty_patterns)
        if difficulty_score > 0.4:
            metrics['difficulty_with_task'] = True
            metrics['performance_score'] -= difficulty_score * 20
        
        # Engagement impact
        engagement_impact = {
            'HIGHLY_ENGAGED': 10,
            'ENGAGED': 0,
            'PREPARING': -5,
            'IDLE': -15,
            'DISENGAGED': -25
        }
        metrics['performance_score'] += engagement_impact.get(engagement_level, 0)
        
        # Hand usage impact
        if hand_usage in ['UNCERTAIN', 'NONE']:
            metrics['performance_score'] -= 10
        
        # Task complexity adjustment
        complexity = self._assess_task_complexity(action_name)
        metrics['performance_score'] -= (1 - complexity) * 5  # Harder tasks get slight bonus
        
        # Ensure score is within bounds
        metrics['performance_score'] = max(30, min(95, metrics['performance_score']))
        
        return metrics

    def assess_handover_need(self, performance_metrics: Dict, action_name: str, engagement_level: str, analysis: str) -> Dict:
        """Determine if robot handover is needed"""
        handover_decision = {
            'handover_recommended': False,
            'reasons': [],
            'urgency': 'LOW',  # LOW, MEDIUM, HIGH
            'robot_capability_score': self._calculate_robot_capability(action_name),
            'expected_improvement': 0.0
        }
        
        simple_tasks = ['take', 'put', 'position', 'place']
        if any(task in action_name.lower() for task in simple_tasks):
            return {
                'handover_recommended': False,
                'reasons': ['Simple task - no handover needed'],
                'urgency': 'LOW',
                'robot_capability_score': 0.0,
                'expected_improvement': 0.0
            }


        score = performance_metrics.get('performance_score', 80)
        
        # Check handover criteria
        criteria_met = []
        
        # 1. Performance-based handover
        if score < self.handover_thresholds['fatigue']:
            criteria_met.append(('performance', f"Low performance score: {score}/100"))
            handover_decision['urgency'] = 'HIGH' if score < 50 else 'MEDIUM'
        
        # 2. Safety concerns
        if performance_metrics.get('safety_concerns', False):
            criteria_met.append(('safety', "Safety concerns detected"))
            handover_decision['urgency'] = 'HIGH'
        
        # 3. Engagement issues
        if engagement_level in ['DISENGAGED', 'IDLE']:
            criteria_met.append(('engagement', f"Low engagement: {engagement_level}"))
            handover_decision['urgency'] = 'MEDIUM'
        
        # 4. Task-specific difficulties
        if performance_metrics.get('difficulty_with_task', False):
            criteria_met.append(('difficulty', "Operator struggling with task"))
            handover_decision['urgency'] = 'MEDIUM'
        
        # 5. Inefficient techniques
        if performance_metrics.get('inefficient_technique', False):
            criteria_met.append(('inefficiency', "Inefficient technique detected"))
            handover_decision['urgency'] = 'MEDIUM'
        
        # Determine if handover is recommended
        if criteria_met:
            handover_decision['handover_recommended'] = True
            handover_decision['reasons'] = [reason for _, reason in criteria_met]
            
            # Calculate expected improvement
            expected_improvement = max(0, (handover_decision['robot_capability_score'] * 100) - score)
            handover_decision['expected_improvement'] = expected_improvement
            
            # Adjust urgency based on capability
            if handover_decision['robot_capability_score'] < 0.7 and handover_decision['urgency'] != 'HIGH':
                handover_decision['urgency'] = 'MEDIUM'

        # Ensure reasons is always a list, even if empty
        if not isinstance(handover_decision.get('reasons'), list):
            handover_decision['reasons'] = []

        return handover_decision

    def _calculate_robot_capability(self, action_name: str) -> float:
        """Calculate robot's capability for a specific action"""
        robot_capabilities = {
            'precision_tasks': ['align', 'position', 'measure', 'calibrate'],
            'force_tasks': ['screw', 'tighten', 'fasten', 'secure'],
            'repetitive_tasks': ['assemble', 'connect', 'install', 'mount'],
            'delicate_tasks': ['handle', 'place', 'adjust', 'orient']
        }
        
        robot_performance = {
            'precision': 0.95,      # 95% success rate for precision tasks
            'force': 0.92,          # 92% for force-required tasks
            'repetitive': 0.98,     # 98% for repetitive tasks
            'delicate': 0.85,       # 85% for delicate tasks (humans better)
            'general': 0.90         # 90% for general tasks
        }
        
        action_lower = action_name.lower()
        capability_score = robot_performance['general']
        
        # Check which category the action belongs to
        for category, keywords in robot_capabilities.items():
            if any(keyword in action_lower for keyword in keywords):
                capability_score = robot_performance[category.split('_')[0]]
                break
        
        # Add some randomness to simulate real-world variability
        variability = random.uniform(-0.05, 0.05)
        return max(0.6, min(0.99, capability_score + variability))

    def simulate_robot_performance(self, human_performance: Dict, action_name: str, handover_decision: Dict) -> Dict:
        """Simulate robot performance after handover"""
        capability_score = handover_decision.get('robot_capability_score', 0.85)
        
        # Base robot performance
        base_score = capability_score * 100
        
        # Adjust based on task complexity and urgency
        urgency_factor = {
            'LOW': 1.0,
            'MEDIUM': 0.95,
            'HIGH': 0.90
        }.get(handover_decision.get('urgency', 'LOW'), 1.0)
        
        # Add some variability
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

    def analyze_robot_presence(self, frame_data: Dict, analysis: str, handover_decision: Dict) -> Dict:

        robot_present = handover_decision.get('handover_recommended', False)
        
        assistance_level = "NONE"
        if robot_present:
            urgency = handover_decision.get('urgency', 'LOW')
            assistance_level = {
                'LOW': 'MONITORING',
                'MEDIUM': 'PARTIAL_ASSISTANCE', 
                'HIGH': 'ACTIVE_ASSISTANCE'
            }.get(urgency, 'MONITORING')
        
        return {
            'robot_present': robot_present,
            'assistance_level': assistance_level,
            'collaboration_score': self._calculate_collaboration_score(assistance_level, analysis),
            'handover_recommended': handover_decision.get('handover_recommended', False),
            'handover_urgency': handover_decision.get('urgency', 'LOW')
        }


    def _calculate_collaboration_score(self, assistance_level: str, analysis: str) -> float:
        """Calculate collaboration score"""
        assistance_scores = {
            "ACTIVE_ASSISTANCE": 0.85,
            "PARTIAL_ASSISTANCE": 0.65,
            "MONITORING": 0.45,
            "NONE": 0.0
        }
        
        base_score = assistance_scores.get(assistance_level, 0.0)
        
        # Adjust based on analysis content
        analysis_lower = analysis.lower()
        if any(word in analysis_lower for word in ['cooperat', 'together', 'assist', 'help']):
            base_score += 0.1
        elif any(word in analysis_lower for word in ['waiting', 'delay', 'hesitat']):
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))


    def assess_operator_performance(self, analysis: str, detected_objects: Dict, engagement_level: str, hand_usage: str) -> Dict:
        """Comprehensive operator performance assessment with detailed metrics"""
        a = analysis.lower()
        
        # Performance indicators with enhanced metrics
        performance_metrics = {
            'fatigue_detected': False,
            'difficulty_with_task': False,
            'safety_concerns': False,
            'inefficient_technique': False,
            'focus_issues': False,
            'tool_usage_issues': False,
            'recommend_robot_handover': False,
            'performance_score': 80,
            'confidence_level': 0.8,
            'improvement_suggestions': [],
            'performance_category': 'GOOD',  # POOR, FAIR, GOOD, EXCELLENT
            'risk_level': 'LOW'  # LOW, MEDIUM, HIGH
        }
        
        fatigue_patterns = {
            'severe': ['tired', 'fatigue', 'exhausted', 'weary', 'drowsy', 'sleepy'],
            'moderate': ['slow movement', 'sluggish', 'rubbing eyes', 'yawning', 'stretching'],
            'mild': ['straining', 'struggling', 'frustrated', 'sighing', 'rubbing face']
        }
        
        fatigue_detected = False
        fatigue_severity = 0
        
        for severity, keywords in fatigue_patterns.items():
            for keyword in keywords:
                if keyword in a:
                    fatigue_detected = True
                    if severity == 'severe':
                        fatigue_severity += 3
                    elif severity == 'moderate':
                        fatigue_severity += 2
                    else:
                        fatigue_severity += 1
        
        if fatigue_detected:
            performance_metrics['fatigue_detected'] = True
            performance_metrics['performance_score'] -= min(30, fatigue_severity * 5)
        
        # Enhanced difficulty detection
        difficulty_patterns = {
            'severe': ['cannot', 'unable to', 'failed attempt', 'unsuccessful', 'stuck'],
            'moderate': ['struggling', 'difficulty', 'hard time', 'having trouble', 'challenging'],
            'mild': ['taking time', 'careful with', 'precision required', 'complex']
        }
        
        difficulty_detected = False
        difficulty_severity = 0
        
        for severity, keywords in difficulty_patterns.items():
            for keyword in keywords:
                if keyword in a:
                    difficulty_detected = True
                    if severity == 'severe':
                        difficulty_severity += 3
                    elif severity == 'moderate':
                        difficulty_severity += 2
                    else:
                        difficulty_severity += 1
        
        if difficulty_detected:
            performance_metrics['difficulty_with_task'] = True
            performance_metrics['performance_score'] -= min(25, difficulty_severity * 4)
        
        # Enhanced safety assessment
        safety_keywords = {
            'high_risk': ['high_risk', 'hazard_detected', 'dangerous', 'unsafe', 'precarious'],
            'medium_risk': ['medium_risk', 'risk', 'caution', 'careful', 'warning'],
            'safety_positive': ['safe', 'secure', 'protected', 'proper', 'correct']
        }
        
        safety_score = 0
        for risk_level, keywords in safety_keywords.items():
            for keyword in keywords:
                if keyword in a:
                    if risk_level == 'high_risk':
                        safety_score -= 3
                    elif risk_level == 'medium_risk':
                        safety_score -= 2
                    else:
                        safety_score += 1
        
        if safety_score < -2:
            performance_metrics['safety_concerns'] = True
            performance_metrics['performance_score'] -= 25
            performance_metrics['risk_level'] = 'HIGH'
        elif safety_score < 0:
            performance_metrics['safety_concerns'] = True
            performance_metrics['performance_score'] -= 15
            performance_metrics['risk_level'] = 'MEDIUM'
        
        # Inefficient technique detection with patterns
        inefficiency_patterns = [
            'awkward', 'inefficient', 'poor technique', 'wrong tool',
            'incorrect method', 'improper grip', 'wasting time',
            'unnecessary movement', 'repositioning', 'adjusting frequently'
        ]
        
        inefficiency_count = sum(1 for pattern in inefficiency_patterns if pattern in a)
        if inefficiency_count > 0:
            performance_metrics['inefficient_technique'] = True
            performance_metrics['performance_score'] -= min(20, inefficiency_count * 3)
        
        # Focus and attention issues
        focus_keywords = ['distracted', 'looking away', 'not focused', 'unfocused', 'divided attention']
        if any(keyword in a for keyword in focus_keywords):
            performance_metrics['focus_issues'] = True
            performance_metrics['performance_score'] -= 10
        
        # Tool usage issues
        tool_keywords = ['wrong tool', 'improper tool', 'missing tool', 'tool difficulty']
        if any(keyword in a for keyword in tool_keywords):
            performance_metrics['tool_usage_issues'] = True
            performance_metrics['performance_score'] -= 8
        
        # Engagement level impact with graduated penalties
        engagement_impact = {
            'HIGHLY_ENGAGED': 5,    # Bonus for high engagement
            'ENGAGED': 0,
            'PREPARING': -5,
            'IDLE': -20,
            'DISENGAGED': -35
        }
        performance_metrics['performance_score'] += engagement_impact.get(engagement_level, 0)
        
        # Hand usage impact with detailed assessment
        hand_usage_impact = {
            'BOTH': 2,       # Bonus for using both hands effectively
            'RIGHT': 0,
            'LEFT': 0,
            'UNCERTAIN': -12,
            'NONE': -25
        }
        performance_metrics['performance_score'] += hand_usage_impact.get(hand_usage, -15)
        
        # Object detection impact (if hands are detected but not used properly)
        if detected_objects.get('hands') and hand_usage in ['UNCERTAIN', 'NONE']:
            performance_metrics['performance_score'] -= 5
        
        # Determine performance category
        score = performance_metrics['performance_score']
        if score >= 85:
            performance_metrics['performance_category'] = 'EXCELLENT'
        elif score >= 70:
            performance_metrics['performance_category'] = 'GOOD'
        elif score >= 55:
            performance_metrics['performance_category'] = 'FAIR'
        else:
            performance_metrics['performance_category'] = 'POOR'

        performance_metrics['improvement_suggestions'] = self._generate_improvement_suggestions(performance_metrics, a)
        
        handover_conditions = [
            performance_metrics['performance_score'] < 55,  # Lower threshold for handover
            performance_metrics['safety_concerns'] and performance_metrics['risk_level'] == 'HIGH',
            performance_metrics['fatigue_detected'] and performance_metrics['performance_score'] < 65,
            performance_metrics['difficulty_with_task'] and performance_metrics['performance_score'] < 60,
            performance_metrics['performance_category'] == 'POOR'
        ]
        
        if any(handover_conditions):
            performance_metrics['recommend_robot_handover'] = True
        
        performance_metrics['performance_score'] = max(10, min(98, performance_metrics['performance_score']))
        
        analysis_quality = len(a.split())
        if analysis_quality < 20:
            performance_metrics['confidence_level'] = max(0.5, performance_metrics['confidence_level'] - 0.2)
        elif analysis_quality > 50:
            performance_metrics['confidence_level'] = min(0.95, performance_metrics['confidence_level'] + 0.1)
        
        return performance_metrics

    def _generate_improvement_suggestions(self, metrics: Dict, analysis: str) -> List[str]:
        suggestions = []
        a = analysis.lower()
        
        if metrics['fatigue_detected']:
            suggestions.extend([
                "Take short breaks to reduce fatigue and maintain focus",
                "Ensure proper ergonomic setup to minimize physical strain",
                "Consider task rotation to maintain engagement and reduce monotony"
            ])
        
        if metrics['safety_concerns']:
            risk_level = metrics.get('risk_level', 'LOW')
            if risk_level == 'HIGH':
                suggestions.extend([
                    "Immediately address safety hazards before continuing",
                    "Review safety procedures and use appropriate protective equipment",
                    "Ensure workspace is clear of immediate dangers"
                ])
            else:
                suggestions.extend([
                    "Review safety protocols for this specific task",
                    "Double-check tool and equipment safety before use",
                    "Maintain awareness of workspace safety throughout the task"
                ])
        
        if metrics['difficulty_with_task']:
            suggestions.extend([
                "Break down complex tasks into smaller, manageable steps",
                "Request guidance or review instructional materials if available",
                "Practice the specific motion or technique to build proficiency"
            ])
        
        if metrics['inefficient_technique']:
            suggestions.extend([
                "Review and practice optimal techniques for this task",
                "Consider alternative tools or approaches that might be more efficient",
                "Observe expert demonstrations if available for this specific action"
            ])
        
        if metrics.get('focus_issues', False):
            suggestions.extend([
                "Minimize distractions in the work environment",
                "Use focused attention techniques for precision tasks",
                "Take mental breaks to maintain concentration during long tasks"
            ])
        
        if metrics.get('tool_usage_issues', False):
            suggestions.extend([
                "Ensure you're using the correct tool for this specific task",
                "Familiarize yourself with tool features and proper usage",
                "Keep tools organized and within easy reach during assembly"
            ])
        
        # Add general best practices if few specific suggestions
        if len(suggestions) < 3:
            suggestions.extend([
                "Maintain consistent rhythm and pace throughout the task",
                "Double-check alignments and connections before final assembly",
                "Keep work area organized to minimize search and setup time",
                "Use both hands cooperatively when possible for efficiency",
                "Follow a systematic approach to complex assembly tasks"
            ])
        
        return suggestions[:4]  # Return top 4 most relevant suggestions


# Utility functions for parsing LLaVA responses
def extract_engagement_level(analysis: str) -> str:
    a = analysis.lower()
    disengaged_keywords = [
        'using mobile', 'on mobile', 'using phone', 'on the phone',
        'talking on phone', 'talking on mobile', 'sleeping', 'dozing', 'distracted', 'not paying attention', 'looking away', 'looking at phone', 'texting', 'checking phone', 'scrolling']
    strong_disengaged = any(kw in a for kw in [
        'using mobile', 'on mobile', 'using phone', 'sleeping', 'dozing'])
    if strong_disengaged:
        return "DISENGAGED"

    assembly_keywords = [
        'assembling', 'aligning', 'screwing', 'tightening', 'connecting',
        'plugging', 'inserting', 'placing', 'adjusting', 'handling',
        'manipulating', 'working on', 'operating', 'using tool'
    ]
    if any(kw in a for kw in assembly_keywords):
        if any(w in a for w in [
            'highly engaged', 'very engaged', 'completely focused', 'fully engaged', 'intensely focused', 'deeply concentrated', 'carefully', 'precisely']):
            return "HIGHLY_ENGAGED"
        if any(w in a for w in ['actively', 'engaged', 'focusing', 'concentrating', 'attentive']):
            return "ENGAGED"
        return "ENGAGED"

    if any(w in a for w in ['preparing', 'organizing', 'setting up', 'arranging', 'getting ready', 'selecting', 'reaching for', 'positioning']):
        return "PREPARING"

    if any(w in a for w in ['idle', 'waiting', 'resting', 'paused', 'not active', 'not working', 'standing by', 'inactive', 'stationary']):
        return "IDLE"

    weak_disengaged = any(kw in a for kw in ['looking away', 'distracted', 'not paying attention'])
    if weak_disengaged:
        count = sum(1 for kw in disengaged_keywords if kw in a)
        return "DISENGAGED" if count >= 2 else "ENGAGED"

    return "ENGAGED"



def extract_hand_usage(analysis: str, detected_objects: Dict = None) -> str:
    a = analysis.lower()

    # Debug: Print analysis to understand what LLaVA is saying
    # print(f"LLaVA Analysis:=========================================== {analysis}")
    
    # Use object detection data if available
    if detected_objects and detected_objects.get('hands'):
        hand_count = len(detected_objects['hands'])
        # print(f"Detected hands: {hand_count}")  # Added this line
        if hand_count == 1:
            # Try to determine which hand based on analysis
            if 'left' in a or 'left hand' in a:
                return "LEFT"
            elif 'right' in a or 'right hand' in a:
                return "RIGHT"
            else:
                # Default to UNCERTAIN if we can't determine which hand used
                return "UNCERTAIN"
        elif hand_count >= 2:
            return "BOTH"
    
    # More comprehensive hand detection patterns with priority
    both_hands_patterns = [
        'both hands', 'two hands', 'using both', 'with both', 'each hand',
        'hands together', 'cooperatively', 'simultaneously', 'together',
        'left and right', 'right and left', 'both sides'
    ]
    
    left_hand_patterns = [
        'left hand', 'left-hand', 'left_hand', 'using left', 'with left',
        'left only', 'left primarily', 'left mainly', 'left mostly',
        'holding with left', 'gripping with left', 'manipulating with left',
        'left side', 'left part', 'on the left'
    ]
    
    right_hand_patterns = [
        'right hand', 'right-hand', 'right_hand', 'using right', 'with right',
        'right only', 'right primarily', 'right mainly', 'right mostly',
        'holding with right', 'gripping with right', 'manipulating with right',
        'right side', 'right part', 'on the right'
    ]
    
    no_hands_patterns = [
        'no hands', 'without hands', 'not using hands', 'hands free',
        'hands not visible', 'hands not used', 'hands not involved',
        'using tool only', 'automated', 'mechanical assistance'
    ]
    
    # Check for specific patterns with priority (BOTH first)
    for pattern in both_hands_patterns:
        if pattern in a:
            return "BOTH"
    
    # Check for left hand patterns
    for pattern in left_hand_patterns:
        if pattern in a:
            # Make sure it's not mentioning right hand in the same context
            if not any(r_pattern in a for r_pattern in right_hand_patterns):
                return "LEFT"
    
    # Check for right hand patterns
    for pattern in right_hand_patterns:
        if pattern in a:
            # Make sure it's not mentioning left hand in the same context
            if not any(l_pattern in a for l_pattern in left_hand_patterns):
                return "RIGHT"
    
    # Check for no hands patterns
    for pattern in no_hands_patterns:
        if pattern in a:
            return "NONE"
    
    # Fallback: Look for action verbs that imply hand usage
    hand_action_verbs = {
        'holding': 'BOTH',
        'gripping': 'BOTH', 
        'manipulating': 'BOTH',
        'assembling': 'BOTH',
        'screwing': 'BOTH',
        'tightening': 'BOTH',
        'fastening': 'BOTH',
        'connecting': 'BOTH',
        'adjusting': 'BOTH',
        'positioning': 'BOTH'
    }
    
    for verb, default_hand in hand_action_verbs.items():
        if verb in a:
            return default_hand
    
    # Look for tool usage that typically requires hands
    tool_words = ['screwdriver', 'wrench', 'pliers', 'tool', 'device', 'instrument']
    if any(tool_word in a for tool_word in tool_words):
        return "BOTH"
    
    # Check for spatial references that might indicate hand usage
    if ('left' in a and any(action in a for action in ['holding', 'using', 'with'])) or \
       ('on the left' in a and 'hand' in a):
        return "LEFT"
        
    if ('right' in a and any(action in a for action in ['holding', 'using', 'with'])) or \
       ('on the right' in a and 'hand' in a):
        return "RIGHT"
    
    return "UNCERTAIN"


def extract_safety_assessment(analysis: str, detected_objects: Dict) -> str:
    a = analysis.lower()
    has_hazards = bool(detected_objects.get('safety_hazards')) or bool(detected_objects.get('unknown_objects'))
    if not has_hazards:
        return "SAFE_WORKSPACE"

    if any(w in a for w in ['hazard', 'danger', 'unsafe', 'risk', 'dangerous', 'precarious', 'accident', 'injury', 'harm']):
        if any(w in a for w in ['high', 'significant', 'serious', 'severe']):
            return "HIGH_RISK"
        if any(w in a for w in ['medium', 'moderate', 'some']):
            return "MEDIUM_RISK"
        return "LOW_RISK"
    if any(w in a for w in ['safe', 'secure', 'protected', 'no risk']):
        return "SAFE_WORKSPACE"

    return "MEDIUM_RISK" if detected_objects.get('safety_hazards') else "SAFE_WORKSPACE"

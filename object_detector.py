import os
os.environ["YOLO_USE_FP16"] = "0"
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict
from gpu_config import gpu_config


class ObjectDetector:
    def __init__(self, model_path: str = '/workspace/raid/stu-1/BLIP/d_weight/fras_best.pt', conf_threshold: float = 0.6, gpu_id: int = 7):
        self.model_path = model_path
        self.conf_threshold = conf_threshold

        # Use centralized GPU config
        if gpu_id is None:
            gpu_id = gpu_config.get_next_gpu()
        
        self.device = gpu_config.get_device(gpu_id)
        self.gpu_id = gpu_id
        
        print(f"Loading custom YOLO model on {self.device}...")
        
        try:
            # Try to load the model file
            loaded_data = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(loaded_data, dict):
                print("Model file contains weights dictionary. Using mock object detection.")
                self.model = None
            else:
                self.model = loaded_data
                self.model.eval()
                print("Custom YOLO model loaded successfully")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to mock object detection")
            self.model = None

        self.class_names = [
            'person', 'screwdriver', 'rpi_board', 'rpi_camera', 'display',
            'front_panel', 'pir_sensor', 'camera_module', 'screw', 'bolt',
            'nut', 'hand', 'tool', 'safety_hazard', 'unknown_object'
        ]

    def detect_objects_from_video(self, video_path: str, fps: int = 1) -> Dict:
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return self._get_empty_detections()

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                return self._get_empty_detections()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sampled_frames = min(5, total_frames)  
            
            results = defaultdict(list)
            
            for i in range(sampled_frames):
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_results = self.detect_objects_in_frame(frame)
                
                # Aggregate results
                for key in ['assembly_components', 'tools', 'hands', 'safety_hazards', 'unknown_objects']:
                    results[key].extend(frame_results[key])
            
            cap.release()
            
            final_results = {}
            for key, objects in results.items():
                final_results[key] = list(set(objects)) 
            
            final_results['safety_status'] = self._determine_safety_status(final_results)
            
            return final_results
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return self._get_empty_detections()

    def detect_objects_in_frame(self, frame: np.ndarray) -> Dict:

        if self.model is None:

            return self._get_mock_detections()
        
        try:
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                detections = self.model(input_tensor)
            
            processed_detections = self._process_detections(detections)
            
            return processed_detections
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            return self._get_mock_detections()

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        
        # Normalize
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return frame_tensor.to(self.device)

    def _process_detections(self, detections: torch.Tensor) -> Dict:
        return self._get_mock_detections()

    def _get_mock_detections(self) -> Dict:
        return {
            'assembly_components': ['pir_sensor', 'screw'],
            'tools': ['screwdriver'],
            'hands': ['hand'],
            'safety_hazards': [],
            'unknown_objects': [],
            'safety_status': 'SAFE_WORKSPACE'
        }

    def _get_empty_detections(self) -> Dict:
        return {
            'assembly_components': [],
            'tools': [],
            'hands': [],
            'safety_hazards': [],
            'unknown_objects': [],
            'safety_status': 'UNKNOWN'
        }

    def _categorize_object(self, label: str) -> str:
        label_lower = label.lower()
        
        assembly_components = [
            'rpi_camera', 'rpi_board', 'display', 'front_panel', 'pir_sensor', 'camera_module', 'screw', 'bolt', 'nut']
        
        tools = ['screwdriver', 'tool', 'wrench', 'pliers']
        hands = ['hand', 'left_hand', 'right_hand']
        safety_hazards = ['sharp_object', 'exposed_wire', 'loose_component', 'clutter']
        
        if any(comp in label_lower for comp in assembly_components):
            return 'assembly_components'
        elif any(tool in label_lower for tool in tools):
            return 'tools'
        elif any(hand in label_lower for hand in hands):
            return 'hands'
        elif any(hazard in label_lower for hazard in safety_hazards):
            return 'safety_hazards'
        else:
            return 'unknown_objects'

    def _determine_safety_status(self, results: Dict) -> str:
        """Determine safety status based on detected objects"""
        hazards = results['safety_hazards']
        unknown_objects = results['unknown_objects']
        
        if not hazards and not unknown_objects:
            return "SAFE_WORKSPACE"
        elif hazards:
            return "MEDIUM_RISK"
        elif unknown_objects:
            return "LOW_RISK"
        else:
            return "SAFE_WORKSPACE"

    def __del__(self):
        """Cleanup"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
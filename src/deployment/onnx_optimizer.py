"""
ONNX Runtime Optimization for Real-Time Game Agent

Author: Minsuk Kim (mk4434)
Purpose: Achieve 60 FPS target through optimized model inference

Key optimizations:
- YOLOv8-nano to ONNX conversion with optimization
- MLP policy network ONNX export
- Inference session configuration for maximum speed
- Memory management and batch processing
- Hardware-specific optimizations (CPU/GPU)
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import time
import json
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for ONNX optimization settings."""
    
    # Target performance
    target_fps: float = 60.0
    target_latency_ms: float = 16.7  # 1000ms / 60fps
    
    # ONNX optimization levels
    graph_optimization_level: str = "ORT_ENABLE_ALL"  # ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
    
    # Execution providers (in order of preference)
    execution_providers: List[str] = None
    
    # Session configuration
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 2
    enable_cpu_mem_arena: bool = True
    enable_mem_pattern: bool = True
    
    # Model-specific settings
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 480)  # NCHW format
    dynamic_axes: Dict[str, Dict[int, str]] = None
    
    def __post_init__(self):
        if self.execution_providers is None:
            # Auto-detect best execution providers
            available_providers = ort.get_available_providers()
            
            # Prefer GPU if available, fallback to optimized CPU
            preferred_order = [
                'CUDAExecutionProvider',
                'CoreMLExecutionProvider', 
                'CPUExecutionProvider'
            ]
            
            self.execution_providers = [
                provider for provider in preferred_order 
                if provider in available_providers
            ]
        
        if self.dynamic_axes is None:
            self.dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }


class ONNXModelOptimizer:
    """
    Optimizes PyTorch models for real-time inference via ONNX Runtime.
    
    Handles both YOLOv8 detection models and MLP policy networks.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """
        Initialize ONNX optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        
        # Performance tracking
        self.benchmark_results = {}
        
        logger.info(f"ONNX Optimizer initialized")
        logger.info(f"Available providers: {ort.get_available_providers()}")
        logger.info(f"Selected providers: {self.config.execution_providers}")
    
    def export_yolo_model(self, 
                         model: torch.nn.Module,
                         output_path: Path,
                         input_shape: Tuple[int, int, int, int] = None) -> Path:
        """
        Export YOLOv8 model to optimized ONNX format.
        
        Args:
            model: YOLOv8 PyTorch model
            output_path: Path to save ONNX model
            input_shape: Input tensor shape (batch, channels, height, width)
            
        Returns:
            Path to exported ONNX model
        """
        if input_shape is None:
            input_shape = self.config.input_shape
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        logger.info(f"Exporting YOLOv8 model to {output_path}")
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=self.config.dynamic_axes,
            verbose=False
        )
        
        # Optimize the exported model
        optimized_path = self._optimize_onnx_model(output_path)
        
        logger.info(f"YOLOv8 model exported and optimized: {optimized_path}")
        return optimized_path
    
    def export_policy_model(self,
                           model: torch.nn.Module,
                           output_path: Path,
                           input_size: int = 8) -> Path:
        """
        Export MLP policy network to optimized ONNX format.
        
        Args:
            model: Policy network PyTorch model
            output_path: Path to save ONNX model
            input_size: Size of state vector input
            
        Returns:
            Path to exported ONNX model
        """
        model.eval()
        
        # Create dummy input (batch_size=1, state_vector_size)
        dummy_input = torch.randn(1, input_size)
        
        logger.info(f"Exporting policy model to {output_path}")
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['state_vector'],
            output_names=['action_logits'],
            dynamic_axes={
                'state_vector': {0: 'batch_size'},
                'action_logits': {0: 'batch_size'}
            },
            verbose=False
        )
        
        # Optimize the exported model
        optimized_path = self._optimize_onnx_model(output_path)
        
        logger.info(f"Policy model exported and optimized: {optimized_path}")
        return optimized_path
    
    def _optimize_onnx_model(self, model_path: Path) -> Path:
        """
        Apply ONNX graph optimizations to reduce inference time.
        
        Args:
            model_path: Path to original ONNX model
            
        Returns:
            Path to optimized ONNX model
        """
        # Load model
        model = onnx.load(str(model_path))
        
        # Apply optimizations
        from onnxruntime.tools import optimizer
        
        optimized_path = model_path.with_suffix('.optimized.onnx')
        
        # Create optimization session
        opt_session = optimizer.optimize_model(
            str(model_path),
            model_type='bert',  # Generic optimization
            num_heads=0,
            hidden_size=0,
            optimization_options=None
        )
        
        # Save optimized model
        opt_session.save_model_to_file(str(optimized_path))
        
        return optimized_path
    
    def create_inference_session(self, 
                                model_path: Path,
                                providers: List[str] = None) -> ort.InferenceSession:
        """
        Create optimized ONNX Runtime inference session.
        
        Args:
            model_path: Path to ONNX model
            providers: Execution providers to use
            
        Returns:
            Configured inference session
        """
        if providers is None:
            providers = self.config.execution_providers
        
        # Session options for optimization
        sess_options = ort.SessionOptions()
        
        # Graph optimization
        if self.config.graph_optimization_level == "ORT_ENABLE_ALL":
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        elif self.config.graph_optimization_level == "ORT_ENABLE_EXTENDED":
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        elif self.config.graph_optimization_level == "ORT_ENABLE_BASIC":
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        else:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        # Threading configuration
        sess_options.intra_op_num_threads = self.config.intra_op_num_threads
        sess_options.inter_op_num_threads = self.config.inter_op_num_threads
        
        # Memory optimizations
        sess_options.enable_cpu_mem_arena = self.config.enable_cpu_mem_arena
        sess_options.enable_mem_pattern = self.config.enable_mem_pattern
        
        # Create session
        session = ort.InferenceSession(
            str(model_path),
            sess_options,
            providers=providers
        )
        
        logger.info(f"Created inference session for {model_path}")
        logger.info(f"Using providers: {session.get_providers()}")
        
        return session
    
    def benchmark_model(self, 
                       session: ort.InferenceSession,
                       input_data: np.ndarray,
                       num_runs: int = 100,
                       warmup_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark model inference performance.
        
        Args:
            session: ONNX Runtime inference session
            input_data: Sample input data for benchmarking
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs (not counted)
            
        Returns:
            Performance statistics
        """
        input_name = session.get_inputs()[0].name
        
        # Warmup runs
        logger.info(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            _ = session.run(None, {input_name: input_data})
        
        # Benchmark runs
        logger.info(f"Running {num_runs} benchmark iterations...")
        times = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = session.run(None, {input_name: input_data})
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Calculate statistics
        times = np.array(times)
        stats = {
            'mean_latency_ms': float(np.mean(times)),
            'std_latency_ms': float(np.std(times)),
            'min_latency_ms': float(np.min(times)),
            'max_latency_ms': float(np.max(times)),
            'p50_latency_ms': float(np.percentile(times, 50)),
            'p95_latency_ms': float(np.percentile(times, 95)),
            'p99_latency_ms': float(np.percentile(times, 99)),
            'mean_fps': 1000.0 / float(np.mean(times)),
            'target_fps': self.config.target_fps,
            'meets_target': float(np.mean(times)) <= self.config.target_latency_ms
        }
        
        logger.info(f"Benchmark Results:")
        logger.info(f"  Mean latency: {stats['mean_latency_ms']:.2f}ms")
        logger.info(f"  Mean FPS: {stats['mean_fps']:.1f}")
        logger.info(f"  Target: {self.config.target_fps} FPS ({self.config.target_latency_ms:.1f}ms)")
        logger.info(f"  Meets target: {stats['meets_target']}")
        
        return stats


class RealTimeInferencePipeline:
    """
    Real-time inference pipeline combining YOLO detection and policy networks.
    
    Optimized for 60 FPS game agent performance.
    """
    
    def __init__(self, 
                 yolo_model_path: Path,
                 policy_model_path: Path,
                 config: OptimizationConfig = None):
        """
        Initialize real-time inference pipeline.
        
        Args:
            yolo_model_path: Path to optimized YOLO ONNX model
            policy_model_path: Path to optimized policy ONNX model
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        
        # Create inference sessions
        optimizer = ONNXModelOptimizer(self.config)
        
        self.yolo_session = optimizer.create_inference_session(yolo_model_path)
        self.policy_session = optimizer.create_inference_session(policy_model_path)
        
        # Get input/output names
        self.yolo_input_name = self.yolo_session.get_inputs()[0].name
        self.yolo_output_name = self.yolo_session.get_outputs()[0].name
        
        self.policy_input_name = self.policy_session.get_inputs()[0].name
        self.policy_output_name = self.policy_session.get_outputs()[0].name
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        logger.info("Real-time inference pipeline initialized")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess game frame for YOLO inference.
        
        Args:
            frame: Raw game frame (H, W, C)
            
        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        # Resize to model input size
        target_h, target_w = self.config.input_shape[2], self.config.input_shape[3]
        
        if frame.shape[:2] != (target_h, target_w):
            import cv2
            frame = cv2.resize(frame, (target_w, target_h))
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Convert HWC to CHW
        frame = np.transpose(frame, (2, 0, 1))
        
        # Add batch dimension
        frame = np.expand_dims(frame, axis=0)
        
        return frame
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLO object detection on frame.
        
        Args:
            frame: Preprocessed frame tensor
            
        Returns:
            List of detected objects
        """
        # Run inference
        outputs = self.yolo_session.run([self.yolo_output_name], {self.yolo_input_name: frame})
        detections = outputs[0]
        
        # Post-process detections (simplified)
        # In practice, you'd apply NMS and confidence thresholding
        processed_detections = []
        
        # This is a placeholder - actual YOLO output processing would be more complex
        if len(detections.shape) >= 2 and detections.shape[1] >= 6:
            for detection in detections[0]:  # Assuming batch size 1
                if detection[4] > 0.5:  # Confidence threshold
                    processed_detections.append({
                        'bbox': detection[:4].tolist(),
                        'confidence': float(detection[4]),
                        'class_id': int(detection[5]) if len(detection) > 5 else 0,
                        'class_name': 'object'
                    })
        
        return processed_detections
    
    def extract_state_vector(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert detections to structured state vector for policy network.
        
        Args:
            detections: List of detected objects
            frame_shape: Original frame dimensions (H, W)
            
        Returns:
            State vector as numpy array
        """
        # Initialize state vector with default values
        state = np.zeros(8, dtype=np.float32)
        
        # Find player and obstacles
        player_detection = None
        obstacle_detections = []
        
        for det in detections:
            if det['class_id'] == 0:  # Player
                player_detection = det
            elif det['class_id'] == 1:  # Obstacle
                obstacle_detections.append(det)
        
        if player_detection:
            bbox = player_detection['bbox']
            # Normalize coordinates
            state[0] = (bbox[0] + bbox[2]) / 2 / frame_shape[1]  # player_x
            state[1] = (bbox[1] + bbox[3]) / 2 / frame_shape[0]  # player_y
        
        if obstacle_detections:
            # Find nearest obstacle
            nearest_obstacle = min(obstacle_detections, 
                                 key=lambda x: x['bbox'][0])  # Leftmost obstacle
            
            bbox = nearest_obstacle['bbox']
            state[2] = (bbox[0] + bbox[2]) / 2 / frame_shape[1]  # obstacle_x
            state[3] = (bbox[1] + bbox[3]) / 2 / frame_shape[0]  # obstacle_y
            
            # Calculate distance and time to collision (simplified)
            if player_detection:
                dx = state[2] - state[0]
                state[4] = max(0, dx)  # distance_to_obstacle
                state[5] = dx / 0.1 if dx > 0 else 0  # time_to_collision (assuming speed)
        
        # Add batch dimension
        return np.expand_dims(state, axis=0)
    
    def predict_action(self, state_vector: np.ndarray) -> Tuple[str, float]:
        """
        Predict action using policy network.
        
        Args:
            state_vector: State vector tensor
            
        Returns:
            Tuple of (action_name, confidence)
        """
        # Run policy inference
        outputs = self.policy_session.run([self.policy_output_name], 
                                        {self.policy_input_name: state_vector})
        logits = outputs[0][0]  # Remove batch dimension
        
        # Convert logits to action
        action_names = ['stay', 'flap', 'left', 'right']
        action_probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
        
        action_idx = np.argmax(action_probs)
        action_name = action_names[action_idx] if action_idx < len(action_names) else 'stay'
        confidence = float(action_probs[action_idx])
        
        return action_name, confidence
    
    def process_frame(self, frame: np.ndarray) -> Tuple[str, float, List[Dict], np.ndarray]:
        """
        Complete frame processing pipeline: detection → state → action.
        
        Args:
            frame: Raw game frame
            
        Returns:
            Tuple of (action, confidence, detections, state_vector)
        """
        start_time = time.perf_counter()
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Object detection
        detections = self.detect_objects(processed_frame)
        
        # Extract state vector
        state_vector = self.extract_state_vector(detections, frame.shape[:2])
        
        # Predict action
        action, confidence = self.predict_action(state_vector)
        
        # Update performance tracking
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        
        self.frame_count += 1
        self.total_inference_time += inference_time
        
        return action, confidence, detections, state_vector[0]  # Remove batch dim from state
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if self.frame_count == 0:
            return {}
        
        avg_inference_time = self.total_inference_time / self.frame_count
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'avg_inference_time_ms': avg_inference_time * 1000,
            'avg_fps': avg_fps,
            'target_fps': self.config.target_fps,
            'meets_target': avg_fps >= self.config.target_fps * 0.95,
            'total_frames': self.frame_count
        }


if __name__ == "__main__":
    # Example usage and testing
    config = OptimizationConfig(target_fps=60.0)
    optimizer = ONNXModelOptimizer(config)
    
    print("ONNX Runtime Optimizer ready!")
    print(f"Target performance: {config.target_fps} FPS ({config.target_latency_ms:.1f}ms)")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Selected providers: {config.execution_providers}")

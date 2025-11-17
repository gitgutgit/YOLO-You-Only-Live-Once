"""
Visualization and Debugging Tools for Game Agent

Author: Minsuk Kim (mk4434)
Purpose: Real-time visualization of detections, state vectors, and policy decisions

Key features:
- Bounding box rendering with confidence scores
- State vector logging and interpretation
- Policy decision reasoning display
- Performance metrics visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


class GameVisualizer:
    """
    Real-time visualization system for debugging the game agent.
    
    Provides overlay rendering for detections, state information,
    and policy decisions directly on game frames.
    """
    
    def __init__(self, 
                 frame_size: Tuple[int, int] = (640, 480),
                 font_scale: float = 0.6,
                 line_thickness: int = 2):
        """
        Initialize visualization system.
        
        Args:
            frame_size: Game frame dimensions (width, height)
            font_scale: Text size for overlays
            line_thickness: Thickness for bounding boxes and lines
        """
        self.frame_size = frame_size
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        
        # Color scheme for different object classes
        self.class_colors = {
            0: (0, 255, 0),    # Player - Green
            1: (0, 0, 255),    # Obstacle - Red  
            2: (255, 255, 0),  # Gap/Safe zone - Cyan
            3: (255, 0, 255),  # Item/Collectible - Magenta
        }
        
        # Class names for display
        self.class_names = {
            0: "Player",
            1: "Obstacle", 
            2: "Gap",
            3: "Item"
        }
        
        # State logging for analysis
        self.state_history = []
        self.action_history = []
        self.decision_log = []
    
    def draw_detections(self, 
                       frame: np.ndarray,
                       detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels for detected objects.
        
        Args:
            frame: Input game frame
            detections: List of detection dictionaries with keys:
                       'bbox', 'class_id', 'confidence', 'class_name'
        
        Returns:
            Frame with detection overlays
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            class_id = detection['class_id']
            confidence = detection['confidence']
            class_name = detection.get('class_name', self.class_names.get(class_id, 'Unknown'))
            
            # Get color for this class
            color = self.class_colors.get(class_id, (128, 128, 128))
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.line_thickness)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)[0]
            
            # Background rectangle for text
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Text label
            cv2.putText(annotated_frame, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       self.font_scale,
                       (255, 255, 255),
                       1)
        
        return annotated_frame
    
    def draw_state_vector(self, 
                         frame: np.ndarray,
                         state_vector: Dict[str, float],
                         position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Draw structured state vector information on frame.
        
        Args:
            frame: Input game frame
            state_vector: Dictionary of state variables
            position: Top-left position for state display
            
        Returns:
            Frame with state information overlay
        """
        annotated_frame = frame.copy()
        x, y = position
        line_height = 25
        
        # Background panel for state info
        panel_height = len(state_vector) * line_height + 20
        cv2.rectangle(annotated_frame,
                     (x - 5, y - 15),
                     (x + 300, y + panel_height),
                     (0, 0, 0, 128), -1)  # Semi-transparent black
        
        # Draw each state variable
        for i, (key, value) in enumerate(state_vector.items()):
            text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            cv2.putText(annotated_frame, text,
                       (x, y + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       self.font_scale,
                       (255, 255, 255),
                       1)
        
        return annotated_frame
    
    def draw_policy_decision(self,
                           frame: np.ndarray,
                           action: str,
                           reasoning: str = "",
                           confidence: float = 0.0,
                           position: Tuple[int, int] = None) -> np.ndarray:
        """
        Draw policy decision and reasoning on frame.
        
        Args:
            frame: Input game frame
            action: Chosen action (e.g., "flap", "stay", "left", "right")
            reasoning: Human-readable explanation of decision
            confidence: Action confidence/probability
            position: Position for decision display (auto if None)
            
        Returns:
            Frame with decision overlay
        """
        annotated_frame = frame.copy()
        
        if position is None:
            position = (frame.shape[1] - 350, 30)  # Top-right corner
        
        x, y = position
        
        # Action color coding
        action_colors = {
            "flap": (0, 255, 255),    # Yellow - Active action
            "jump": (0, 255, 255),    # Yellow - Active action
            "left": (255, 0, 0),      # Blue - Movement
            "right": (255, 0, 0),     # Blue - Movement  
            "stay": (0, 255, 0),      # Green - Passive
            "nothing": (0, 255, 0),   # Green - Passive
        }
        
        action_color = action_colors.get(action.lower(), (255, 255, 255))
        
        # Decision panel background
        panel_width = 340
        panel_height = 80 if reasoning else 50
        cv2.rectangle(annotated_frame,
                     (x - 5, y - 15),
                     (x + panel_width, y + panel_height),
                     (0, 0, 0, 128), -1)
        
        # Action text
        action_text = f"Action: {action.upper()}"
        if confidence > 0:
            action_text += f" ({confidence:.1%})"
        
        cv2.putText(annotated_frame, action_text,
                   (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   self.font_scale + 0.1,
                   action_color,
                   2)
        
        # Reasoning text (if provided)
        if reasoning:
            # Split long reasoning into multiple lines
            words = reasoning.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                test_line = " ".join(current_line)
                if len(test_line) > 45:  # Approximate character limit
                    if len(current_line) > 1:
                        current_line.pop()
                        lines.append(" ".join(current_line))
                        current_line = [word]
                    else:
                        lines.append(test_line)
                        current_line = []
            
            if current_line:
                lines.append(" ".join(current_line))
            
            # Draw reasoning lines
            for i, line in enumerate(lines[:2]):  # Max 2 lines
                cv2.putText(annotated_frame, line,
                           (x, y + 25 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           self.font_scale - 0.1,
                           (200, 200, 200),
                           1)
        
        return annotated_frame
    
    def log_decision(self,
                    frame_id: int,
                    state_vector: Dict[str, float],
                    action: str,
                    reasoning: str = "",
                    confidence: float = 0.0,
                    timestamp: Optional[datetime] = None) -> None:
        """
        Log decision for later analysis and debugging.
        
        Args:
            frame_id: Unique frame identifier
            state_vector: Current game state
            action: Chosen action
            reasoning: Decision reasoning
            confidence: Action confidence
            timestamp: Decision timestamp (auto if None)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        decision_entry = {
            'frame_id': frame_id,
            'timestamp': timestamp.isoformat(),
            'state': state_vector.copy(),
            'action': action,
            'reasoning': reasoning,
            'confidence': confidence
        }
        
        self.decision_log.append(decision_entry)
        self.state_history.append(state_vector.copy())
        self.action_history.append(action)
    
    def save_decision_log(self, output_path: Path) -> None:
        """Save decision log to JSON file for analysis."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.decision_log, f, indent=2)
        
        print(f"Saved {len(self.decision_log)} decisions to {output_path}")
    
    def create_performance_dashboard(self, 
                                   output_dir: Path,
                                   survival_times: List[float],
                                   action_counts: Dict[str, int],
                                   state_stats: Dict[str, List[float]]) -> None:
        """
        Create comprehensive performance analysis dashboard.
        
        Args:
            output_dir: Directory to save dashboard plots
            survival_times: List of survival times across episodes
            action_counts: Count of each action type
            state_stats: Statistics for each state variable
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Survival time progression
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(survival_times, alpha=0.7)
        plt.plot(pd.Series(survival_times).rolling(window=10).mean(), 
                color='red', linewidth=2, label='10-episode average')
        plt.xlabel('Episode')
        plt.ylabel('Survival Time (s)')
        plt.title('Learning Progress: Survival Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Action distribution
        plt.subplot(1, 2, 2)
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        plt.pie(counts, labels=actions, autopct='%1.1f%%', colors=colors)
        plt.title('Action Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_overview.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. State variable analysis
        if state_stats:
            n_vars = len(state_stats)
            fig, axes = plt.subplots(2, (n_vars + 1) // 2, figsize=(15, 8))
            axes = axes.flatten() if n_vars > 1 else [axes]
            
            for i, (var_name, values) in enumerate(state_stats.items()):
                if i < len(axes):
                    axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{var_name} Distribution')
                    axes[i].set_xlabel(var_name)
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(state_stats), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'state_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Performance dashboard saved to {output_dir}")


class PerformanceProfiler:
    """
    Real-time performance profiling for the game agent pipeline.
    
    Tracks inference times, FPS, and bottlenecks to ensure
    60 FPS target is maintained.
    """
    
    def __init__(self, target_fps: float = 60.0):
        """
        Initialize performance profiler.
        
        Args:
            target_fps: Target frames per second
        """
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps  # seconds per frame
        
        # Timing measurements
        self.frame_times = []
        self.detection_times = []
        self.policy_times = []
        self.total_times = []
        
        # Current frame timing
        self.current_frame_start = None
        self.current_detection_start = None
        self.current_policy_start = None
    
    def start_frame(self) -> None:
        """Mark start of frame processing."""
        import time
        self.current_frame_start = time.perf_counter()
    
    def start_detection(self) -> None:
        """Mark start of detection inference."""
        import time
        self.current_detection_start = time.perf_counter()
    
    def end_detection(self) -> None:
        """Mark end of detection inference."""
        import time
        if self.current_detection_start is not None:
            detection_time = time.perf_counter() - self.current_detection_start
            self.detection_times.append(detection_time)
    
    def start_policy(self) -> None:
        """Mark start of policy inference."""
        import time
        self.current_policy_start = time.perf_counter()
    
    def end_policy(self) -> None:
        """Mark end of policy inference."""
        import time
        if self.current_policy_start is not None:
            policy_time = time.perf_counter() - self.current_policy_start
            self.policy_times.append(policy_time)
    
    def end_frame(self) -> Dict[str, float]:
        """
        Mark end of frame processing and return timing stats.
        
        Returns:
            Dictionary with timing information
        """
        import time
        if self.current_frame_start is not None:
            total_time = time.perf_counter() - self.current_frame_start
            self.total_times.append(total_time)
            self.frame_times.append(total_time)
            
            current_fps = 1.0 / total_time if total_time > 0 else 0
            
            return {
                'frame_time': total_time,
                'fps': current_fps,
                'target_fps': self.target_fps,
                'meets_target': current_fps >= self.target_fps * 0.95,  # 5% tolerance
                'detection_time': self.detection_times[-1] if self.detection_times else 0,
                'policy_time': self.policy_times[-1] if self.policy_times else 0
            }
        
        return {}
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get overall performance statistics."""
        if not self.total_times:
            return {}
        
        avg_frame_time = np.mean(self.total_times)
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        return {
            'avg_fps': avg_fps,
            'avg_frame_time_ms': avg_frame_time * 1000,
            'target_frame_time_ms': self.target_frame_time * 1000,
            'avg_detection_time_ms': np.mean(self.detection_times) * 1000 if self.detection_times else 0,
            'avg_policy_time_ms': np.mean(self.policy_times) * 1000 if self.policy_times else 0,
            'fps_target_achievement': (avg_fps / self.target_fps) * 100,
            'total_frames': len(self.total_times)
        }


if __name__ == "__main__":
    # Example usage
    visualizer = GameVisualizer()
    profiler = PerformanceProfiler(target_fps=60)
    
    print("Visualization and profiling tools ready!")
    print(f"Target FPS: {profiler.target_fps}")
    print(f"Target frame time: {profiler.target_frame_time*1000:.1f}ms")

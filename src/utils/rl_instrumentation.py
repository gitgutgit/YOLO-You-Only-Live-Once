"""
Reinforcement Learning Loop Instrumentation

Author: Minsuk Kim (mk4434)
Purpose: Comprehensive monitoring and analysis of RL training process

Key features:
- Real-time training metrics collection
- Rollout statistics and analysis
- Failure mode detection and categorization
- Performance trend analysis
- Integration with TensorBoard and Weights & Biases
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


@dataclass
class RLMetrics:
    """Container for RL training metrics."""
    
    # Episode-level metrics
    episode_reward: float = 0.0
    episode_length: int = 0
    survival_time: float = 0.0
    
    # Action statistics
    action_counts: Dict[str, int] = field(default_factory=dict)
    action_distribution: Dict[str, float] = field(default_factory=dict)
    
    # State statistics
    state_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Policy metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl_divergence: float = 0.0
    
    # Performance metrics
    fps: float = 0.0
    inference_time_ms: float = 0.0
    
    # Failure analysis
    failure_mode: Optional[str] = None
    failure_position: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'survival_time': self.survival_time,
            'action_counts': self.action_counts,
            'action_distribution': self.action_distribution,
            'state_stats': self.state_stats,
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'entropy': self.entropy,
            'kl_divergence': self.kl_divergence,
            'fps': self.fps,
            'inference_time_ms': self.inference_time_ms,
            'failure_mode': self.failure_mode,
            'failure_position': self.failure_position
        }


class FailureModeAnalyzer:
    """
    Analyzes and categorizes agent failure modes for debugging.
    
    Helps identify common failure patterns and guide training improvements.
    """
    
    def __init__(self):
        """Initialize failure mode analyzer."""
        self.failure_categories = {
            'collision_obstacle': 'Hit obstacle directly',
            'collision_wall': 'Hit wall/boundary',
            'timeout': 'Episode timeout',
            'poor_timing': 'Action timing issues',
            'stuck_behavior': 'Repetitive/stuck actions',
            'exploration_failure': 'Failed to explore properly'
        }
        
        self.failure_history = []
        self.failure_counts = defaultdict(int)
    
    def analyze_failure(self, 
                       final_state: Dict[str, float],
                       action_history: List[str],
                       state_history: List[Dict[str, float]],
                       episode_length: int) -> str:
        """
        Analyze episode failure and categorize the failure mode.
        
        Args:
            final_state: Final state when episode ended
            action_history: Sequence of actions taken
            state_history: Sequence of states observed
            episode_length: Total episode length
            
        Returns:
            Failure mode category
        """
        failure_mode = 'unknown'
        
        # Check for collision with obstacle
        if 'distance_to_obstacle' in final_state:
            if final_state['distance_to_obstacle'] < 0.1:
                failure_mode = 'collision_obstacle'
        
        # Check for boundary collision
        if 'player_x' in final_state:
            if final_state['player_x'] <= 0.05 or final_state['player_x'] >= 0.95:
                failure_mode = 'collision_wall'
        
        # Check for timeout (very long episode)
        if episode_length > 1000:  # Adjust threshold as needed
            failure_mode = 'timeout'
        
        # Check for stuck behavior (repetitive actions)
        if len(action_history) > 10:
            recent_actions = action_history[-10:]
            if len(set(recent_actions)) == 1:  # All same action
                failure_mode = 'stuck_behavior'
        
        # Check for poor timing (rapid action changes)
        if len(action_history) > 5:
            action_changes = sum(1 for i in range(1, len(action_history)) 
                               if action_history[i] != action_history[i-1])
            if action_changes / len(action_history) > 0.8:  # Very frequent changes
                failure_mode = 'poor_timing'
        
        # Record failure
        self.failure_counts[failure_mode] += 1
        self.failure_history.append({
            'mode': failure_mode,
            'episode_length': episode_length,
            'final_state': final_state.copy(),
            'timestamp': datetime.now().isoformat()
        })
        
        return failure_mode
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of failure modes and patterns."""
        total_failures = sum(self.failure_counts.values())
        
        if total_failures == 0:
            return {'total_failures': 0}
        
        failure_rates = {
            mode: count / total_failures 
            for mode, count in self.failure_counts.items()
        }
        
        return {
            'total_failures': total_failures,
            'failure_counts': dict(self.failure_counts),
            'failure_rates': failure_rates,
            'most_common_failure': max(self.failure_counts.items(), key=lambda x: x[1])[0],
            'recent_failures': self.failure_history[-10:]  # Last 10 failures
        }


class RLInstrumentationLogger:
    """
    Comprehensive logging system for RL training process.
    
    Integrates with multiple logging backends and provides real-time monitoring.
    """
    
    def __init__(self, 
                 experiment_name: str,
                 log_dir: Path,
                 use_wandb: bool = True,
                 use_tensorboard: bool = True,
                 wandb_project: str = "distilled-vision-agent"):
        """
        Initialize RL instrumentation logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
            wandb_project: W&B project name
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging backends
        self.tensorboard_writer = None
        self.wandb_run = None
        
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_log_dir = self.log_dir / "tensorboard"
            tb_log_dir.mkdir(exist_ok=True)
            self.tensorboard_writer = SummaryWriter(tb_log_dir)
        
        if use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project=wandb_project,
                name=experiment_name,
                dir=str(self.log_dir)
            )
        
        # Metrics storage
        self.episode_metrics = []
        self.training_metrics = []
        self.step_count = 0
        self.episode_count = 0
        
        # Performance tracking
        self.performance_buffer = deque(maxlen=100)  # Last 100 episodes
        
        # Failure analysis
        self.failure_analyzer = FailureModeAnalyzer()
        
        print(f"RL Instrumentation Logger initialized: {experiment_name}")
        print(f"Log directory: {self.log_dir}")
        print(f"TensorBoard: {'✓' if self.tensorboard_writer else '✗'}")
        print(f"Weights & Biases: {'✓' if self.wandb_run else '✗'}")
    
    def log_episode_start(self, episode_id: int) -> None:
        """Log the start of a new episode."""
        self.episode_count = episode_id
        
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('episode/start', episode_id, self.step_count)
        
        if self.wandb_run:
            wandb.log({'episode/start': episode_id}, step=self.step_count)
    
    def log_step(self, 
                 state: Dict[str, float],
                 action: str,
                 reward: float,
                 done: bool,
                 info: Dict[str, Any] = None) -> None:
        """
        Log a single step in the environment.
        
        Args:
            state: Current state vector
            action: Action taken
            reward: Reward received
            done: Whether episode is done
            info: Additional information
        """
        self.step_count += 1
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('step/reward', reward, self.step_count)
            
            # Log state variables
            for key, value in state.items():
                self.tensorboard_writer.add_scalar(f'state/{key}', value, self.step_count)
        
        # Log to W&B
        if self.wandb_run:
            log_data = {
                'step/reward': reward,
                'step/action': action,
                'step/done': done
            }
            
            # Add state variables
            for key, value in state.items():
                log_data[f'state/{key}'] = value
            
            wandb.log(log_data, step=self.step_count)
    
    def log_episode_end(self, 
                       metrics: RLMetrics,
                       action_history: List[str],
                       state_history: List[Dict[str, float]]) -> None:
        """
        Log the end of an episode with comprehensive metrics.
        
        Args:
            metrics: Episode metrics
            action_history: Complete action sequence
            state_history: Complete state sequence
        """
        # Analyze failure mode
        if len(state_history) > 0:
            final_state = state_history[-1]
            failure_mode = self.failure_analyzer.analyze_failure(
                final_state, action_history, state_history, metrics.episode_length
            )
            metrics.failure_mode = failure_mode
        
        # Store metrics
        self.episode_metrics.append(metrics)
        self.performance_buffer.append(metrics.survival_time)
        
        # Calculate rolling statistics
        recent_performance = list(self.performance_buffer)
        rolling_mean = np.mean(recent_performance) if recent_performance else 0
        rolling_std = np.std(recent_performance) if len(recent_performance) > 1 else 0
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('episode/reward', metrics.episode_reward, self.episode_count)
            self.tensorboard_writer.add_scalar('episode/length', metrics.episode_length, self.episode_count)
            self.tensorboard_writer.add_scalar('episode/survival_time', metrics.survival_time, self.episode_count)
            self.tensorboard_writer.add_scalar('episode/rolling_mean_survival', rolling_mean, self.episode_count)
            self.tensorboard_writer.add_scalar('episode/fps', metrics.fps, self.episode_count)
            
            # Action distribution
            for action, count in metrics.action_counts.items():
                self.tensorboard_writer.add_scalar(f'actions/{action}', count, self.episode_count)
        
        # Log to W&B
        if self.wandb_run:
            log_data = {
                'episode/reward': metrics.episode_reward,
                'episode/length': metrics.episode_length,
                'episode/survival_time': metrics.survival_time,
                'episode/rolling_mean_survival': rolling_mean,
                'episode/rolling_std_survival': rolling_std,
                'episode/fps': metrics.fps,
                'episode/inference_time_ms': metrics.inference_time_ms,
                'episode/failure_mode': metrics.failure_mode or 'success'
            }
            
            # Add action counts
            for action, count in metrics.action_counts.items():
                log_data[f'actions/{action}'] = count
            
            wandb.log(log_data, step=self.episode_count)
    
    def log_training_step(self, 
                         policy_loss: float,
                         value_loss: float,
                         entropy: float,
                         kl_divergence: float = None,
                         learning_rate: float = None) -> None:
        """
        Log training step metrics (for PPO/DQN updates).
        
        Args:
            policy_loss: Policy network loss
            value_loss: Value network loss
            entropy: Policy entropy
            kl_divergence: KL divergence (for PPO)
            learning_rate: Current learning rate
        """
        training_step = len(self.training_metrics)
        
        metrics = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'step': training_step
        }
        
        if kl_divergence is not None:
            metrics['kl_divergence'] = kl_divergence
        
        if learning_rate is not None:
            metrics['learning_rate'] = learning_rate
        
        self.training_metrics.append(metrics)
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('train/policy_loss', policy_loss, training_step)
            self.tensorboard_writer.add_scalar('train/value_loss', value_loss, training_step)
            self.tensorboard_writer.add_scalar('train/entropy', entropy, training_step)
            
            if kl_divergence is not None:
                self.tensorboard_writer.add_scalar('train/kl_divergence', kl_divergence, training_step)
            
            if learning_rate is not None:
                self.tensorboard_writer.add_scalar('train/learning_rate', learning_rate, training_step)
        
        # Log to W&B
        if self.wandb_run:
            log_data = {
                'train/policy_loss': policy_loss,
                'train/value_loss': value_loss,
                'train/entropy': entropy
            }
            
            if kl_divergence is not None:
                log_data['train/kl_divergence'] = kl_divergence
            
            if learning_rate is not None:
                log_data['train/learning_rate'] = learning_rate
            
            wandb.log(log_data, step=training_step)
    
    def create_progress_report(self, save_path: Path = None) -> Dict[str, Any]:
        """
        Create comprehensive progress report with visualizations.
        
        Args:
            save_path: Path to save report plots
            
        Returns:
            Progress report dictionary
        """
        if not self.episode_metrics:
            return {'error': 'No episode data available'}
        
        # Extract data for analysis
        survival_times = [m.survival_time for m in self.episode_metrics]
        episode_rewards = [m.episode_reward for m in self.episode_metrics]
        episode_lengths = [m.episode_length for m in self.episode_metrics]
        
        # Calculate statistics
        report = {
            'total_episodes': len(self.episode_metrics),
            'total_steps': self.step_count,
            'mean_survival_time': np.mean(survival_times),
            'std_survival_time': np.std(survival_times),
            'max_survival_time': np.max(survival_times),
            'mean_episode_reward': np.mean(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'failure_analysis': self.failure_analyzer.get_failure_summary()
        }
        
        # Calculate improvement metrics
        if len(survival_times) >= 20:
            early_performance = np.mean(survival_times[:10])
            recent_performance = np.mean(survival_times[-10:])
            improvement = (recent_performance - early_performance) / early_performance * 100
            report['performance_improvement_percent'] = improvement
        
        # Create visualizations if save path provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            self._create_progress_plots(survival_times, episode_rewards, save_path)
        
        return report
    
    def _create_progress_plots(self, 
                              survival_times: List[float],
                              episode_rewards: List[float],
                              save_dir: Path) -> None:
        """Create and save progress visualization plots."""
        plt.style.use('seaborn-v0_8')
        
        # 1. Learning curve
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Survival time progression
        axes[0, 0].plot(survival_times, alpha=0.6, label='Episode survival')
        if len(survival_times) >= 10:
            rolling_mean = pd.Series(survival_times).rolling(window=10).mean()
            axes[0, 0].plot(rolling_mean, color='red', linewidth=2, label='10-episode average')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Survival Time (s)')
        axes[0, 0].set_title('Learning Progress: Survival Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reward progression
        axes[0, 1].plot(episode_rewards, alpha=0.6, label='Episode reward')
        if len(episode_rewards) >= 10:
            rolling_mean = pd.Series(episode_rewards).rolling(window=10).mean()
            axes[0, 1].plot(rolling_mean, color='red', linewidth=2, label='10-episode average')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Reward')
        axes[0, 1].set_title('Learning Progress: Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Survival time distribution
        axes[1, 0].hist(survival_times, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(survival_times), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(survival_times):.1f}s')
        axes[1, 0].set_xlabel('Survival Time (s)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Survival Time Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Failure mode analysis
        failure_summary = self.failure_analyzer.get_failure_summary()
        if failure_summary.get('failure_counts'):
            failure_modes = list(failure_summary['failure_counts'].keys())
            failure_counts = list(failure_summary['failure_counts'].values())
            
            axes[1, 1].pie(failure_counts, labels=failure_modes, autopct='%1.1f%%')
            axes[1, 1].set_title('Failure Mode Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'No failure data', ha='center', va='center')
            axes[1, 1].set_title('Failure Mode Distribution')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Progress plots saved to {save_dir}")
    
    def save_metrics(self, output_path: Path) -> None:
        """Save all collected metrics to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'experiment_name': self.experiment_name,
            'total_episodes': len(self.episode_metrics),
            'total_steps': self.step_count,
            'episode_metrics': [m.to_dict() for m in self.episode_metrics],
            'training_metrics': self.training_metrics,
            'failure_analysis': self.failure_analyzer.get_failure_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Metrics saved to {output_path}")
    
    def close(self) -> None:
        """Close logging backends and save final data."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            wandb.finish()
        
        # Save final metrics
        self.save_metrics(self.log_dir / 'final_metrics.json')
        
        print("RL Instrumentation Logger closed")


if __name__ == "__main__":
    # Example usage
    logger = RLInstrumentationLogger(
        experiment_name="test_experiment",
        log_dir=Path("logs/test"),
        use_wandb=False,  # Set to True if you have W&B configured
        use_tensorboard=True
    )
    
    print("RL Instrumentation system ready!")
    print("Features:")
    print("- Episode and step-level logging")
    print("- Failure mode analysis")
    print("- Performance trend tracking")
    print("- Multi-backend logging (TensorBoard, W&B)")
    print("- Comprehensive progress reports")

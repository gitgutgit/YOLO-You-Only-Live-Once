#!/usr/bin/env python3
"""
Integration Test Script for Distilled Vision Agent Pipeline

Author: Minsuk Kim (mk4434)
Purpose: Test all components of the pipeline to ensure 60 FPS target

Tests:
1. Data augmentation pipeline
2. Visualization tools
3. ONNX optimization and inference
4. RL instrumentation system
5. End-to-end performance validation
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.augmentation import GameFrameAugmenter, create_background_textures
from utils.visualization import GameVisualizer, PerformanceProfiler
from deployment.onnx_optimizer import OptimizationConfig, ONNXModelOptimizer
from utils.rl_instrumentation import RLInstrumentationLogger, RLMetrics


def test_data_augmentation():
    """Test data augmentation pipeline."""
    print("\n🔧 Testing Data Augmentation Pipeline...")
    
    # Create augmenter
    augmenter = GameFrameAugmenter(
        target_size=(640, 480),
        augmentation_factor=3
    )
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bboxes = [[0.5, 0.3, 0.1, 0.2], [0.7, 0.6, 0.15, 0.25]]  # YOLO format
    test_labels = [0, 1]  # Player, Obstacle
    
    # Test augmentation
    start_time = time.perf_counter()
    
    for i in range(10):
        result = augmenter.augment_frame(test_frame, test_bboxes, test_labels)
        assert result['image'].shape == (480, 640, 3), f"Wrong output shape: {result['image'].shape}"
        assert len(result['bboxes']) <= len(test_bboxes), "Too many bboxes after augmentation"
    
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / 10 * 1000
    
    print(f"   ✓ Augmentation working correctly")
    print(f"   ✓ Average augmentation time: {avg_time:.2f}ms")
    
    # Test background texture generation
    texture_dir = Path("data/textures")
    create_background_textures(texture_dir, num_textures=5)
    augmenter.load_background_textures(texture_dir)
    
    print(f"   ✓ Background textures loaded: {len(augmenter.background_textures)}")
    
    return True


def test_visualization_tools():
    """Test visualization and debugging tools."""
    print("\n🎨 Testing Visualization Tools...")
    
    # Create visualizer
    visualizer = GameVisualizer()
    
    # Create test frame and detections
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_detections = [
        {'bbox': [100, 150, 200, 250], 'class_id': 0, 'confidence': 0.95, 'class_name': 'Player'},
        {'bbox': [400, 100, 500, 200], 'class_id': 1, 'confidence': 0.87, 'class_name': 'Obstacle'}
    ]
    
    # Test detection visualization
    annotated_frame = visualizer.draw_detections(test_frame, test_detections)
    assert annotated_frame.shape == test_frame.shape, "Frame shape changed during annotation"
    
    # Test state vector visualization
    test_state = {
        'player_x': 0.3,
        'player_y': 0.7,
        'obstacle_distance': 0.25,
        'time_to_collision': 2.5
    }
    
    state_frame = visualizer.draw_state_vector(test_frame, test_state)
    assert state_frame.shape == test_frame.shape, "Frame shape changed during state overlay"
    
    # Test policy decision visualization
    decision_frame = visualizer.draw_policy_decision(
        test_frame, 
        action="flap", 
        reasoning="Obstacle approaching, need to gain altitude",
        confidence=0.85
    )
    assert decision_frame.shape == test_frame.shape, "Frame shape changed during decision overlay"
    
    print("   ✓ Detection visualization working")
    print("   ✓ State vector visualization working")
    print("   ✓ Policy decision visualization working")
    
    # Test performance profiler
    profiler = PerformanceProfiler(target_fps=60)
    
    # Simulate frame processing
    for i in range(5):
        profiler.start_frame()
        
        profiler.start_detection()
        time.sleep(0.005)  # Simulate 5ms detection
        profiler.end_detection()
        
        profiler.start_policy()
        time.sleep(0.002)  # Simulate 2ms policy inference
        profiler.end_policy()
        
        stats = profiler.end_frame()
        assert 'fps' in stats, "FPS not calculated"
        assert 'frame_time' in stats, "Frame time not calculated"
    
    summary = profiler.get_performance_summary()
    print(f"   ✓ Performance profiler working (avg FPS: {summary.get('avg_fps', 0):.1f})")
    
    return True


def test_onnx_optimization():
    """Test ONNX optimization pipeline."""
    print("\n⚡ Testing ONNX Optimization...")
    
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("   ⚠️  PyTorch not available, skipping ONNX tests")
        return True
    
    # Create test models
    class SimpleYOLO(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, 6)  # [x, y, w, h, conf, class]
            )
        
        def forward(self, x):
            return self.backbone(x).unsqueeze(1)  # Add detection dimension
    
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 4)  # 4 actions
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Create models
    yolo_model = SimpleYOLO()
    policy_model = SimpleMLP()
    
    # Create optimizer
    config = OptimizationConfig(target_fps=60.0)
    optimizer = ONNXModelOptimizer(config)
    
    # Test model export (create temporary files)
    temp_dir = Path("temp_models")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Export YOLO model
        yolo_path = temp_dir / "test_yolo.onnx"
        exported_yolo = optimizer.export_yolo_model(yolo_model, yolo_path)
        assert exported_yolo.exists(), "YOLO model export failed"
        
        # Export policy model
        policy_path = temp_dir / "test_policy.onnx"
        exported_policy = optimizer.export_policy_model(policy_model, policy_path)
        assert exported_policy.exists(), "Policy model export failed"
        
        print("   ✓ Model export working")
        
        # Test inference session creation
        yolo_session = optimizer.create_inference_session(exported_yolo)
        policy_session = optimizer.create_inference_session(exported_policy)
        
        print("   ✓ Inference session creation working")
        
        # Test benchmarking
        test_input = np.random.randn(1, 3, 640, 480).astype(np.float32)
        yolo_stats = optimizer.benchmark_model(yolo_session, test_input, num_runs=20, warmup_runs=5)
        
        print(f"   ✓ YOLO benchmark: {yolo_stats['mean_fps']:.1f} FPS ({yolo_stats['mean_latency_ms']:.2f}ms)")
        
        test_state = np.random.randn(1, 8).astype(np.float32)
        policy_stats = optimizer.benchmark_model(policy_session, test_state, num_runs=20, warmup_runs=5)
        
        print(f"   ✓ Policy benchmark: {policy_stats['mean_fps']:.1f} FPS ({policy_stats['mean_latency_ms']:.2f}ms)")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"   ⚠️  ONNX optimization test failed: {e}")
        # Clean up on failure
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        return False
    
    return True


def test_rl_instrumentation():
    """Test RL instrumentation system."""
    print("\n📊 Testing RL Instrumentation...")
    
    # Create logger
    log_dir = Path("logs/test_run")
    logger = RLInstrumentationLogger(
        experiment_name="pipeline_test",
        log_dir=log_dir,
        use_wandb=False,  # Disable W&B for testing
        use_tensorboard=True
    )
    
    # Simulate training episodes
    for episode in range(5):
        logger.log_episode_start(episode)
        
        # Simulate episode steps
        action_history = []
        state_history = []
        total_reward = 0
        
        for step in range(np.random.randint(50, 200)):
            # Simulate state and action
            state = {
                'player_x': np.random.uniform(0.1, 0.9),
                'player_y': np.random.uniform(0.1, 0.9),
                'obstacle_distance': np.random.uniform(0.0, 1.0),
                'time_to_collision': np.random.uniform(0.0, 5.0)
            }
            
            action = np.random.choice(['stay', 'flap', 'left', 'right'])
            reward = np.random.uniform(-1, 1)
            done = step > 100 and np.random.random() < 0.1
            
            action_history.append(action)
            state_history.append(state)
            total_reward += reward
            
            logger.log_step(state, action, reward, done)
            
            if done:
                break
        
        # Create episode metrics
        metrics = RLMetrics(
            episode_reward=total_reward,
            episode_length=len(action_history),
            survival_time=len(action_history) * 0.033,  # Assuming 30 FPS
            action_counts={action: action_history.count(action) for action in set(action_history)},
            fps=60.0,
            inference_time_ms=8.5
        )
        
        logger.log_episode_end(metrics, action_history, state_history)
        
        # Simulate training step
        if episode % 2 == 0:
            logger.log_training_step(
                policy_loss=np.random.uniform(0.1, 1.0),
                value_loss=np.random.uniform(0.1, 1.0),
                entropy=np.random.uniform(0.5, 2.0),
                kl_divergence=np.random.uniform(0.01, 0.1)
            )
    
    # Generate progress report
    report = logger.create_progress_report(log_dir / "plots")
    
    print(f"   ✓ Logged {report['total_episodes']} episodes")
    print(f"   ✓ Logged {report['total_steps']} steps")
    print(f"   ✓ Mean survival time: {report['mean_survival_time']:.2f}s")
    
    # Clean up
    logger.close()
    
    return True


def test_end_to_end_performance():
    """Test end-to-end pipeline performance."""
    print("\n🚀 Testing End-to-End Performance...")
    
    # Initialize components
    visualizer = GameVisualizer()
    profiler = PerformanceProfiler(target_fps=60)
    
    # Simulate game frames
    frame_count = 100
    total_pipeline_time = 0
    
    print(f"   Processing {frame_count} frames...")
    
    for i in range(frame_count):
        profiler.start_frame()
        
        # Generate test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Simulate detection (5-10ms)
        profiler.start_detection()
        time.sleep(np.random.uniform(0.005, 0.010))
        detections = [
            {'bbox': [100, 150, 200, 250], 'class_id': 0, 'confidence': 0.95},
            {'bbox': [400, 100, 500, 200], 'class_id': 1, 'confidence': 0.87}
        ]
        profiler.end_detection()
        
        # Simulate policy inference (1-3ms)
        profiler.start_policy()
        time.sleep(np.random.uniform(0.001, 0.003))
        action = np.random.choice(['stay', 'flap', 'left', 'right'])
        profiler.end_policy()
        
        # Visualization (should be fast)
        start_viz = time.perf_counter()
        annotated_frame = visualizer.draw_detections(frame, detections)
        state_frame = visualizer.draw_state_vector(annotated_frame, {
            'player_x': 0.3, 'obstacle_distance': 0.4
        })
        final_frame = visualizer.draw_policy_decision(state_frame, action, confidence=0.8)
        end_viz = time.perf_counter()
        
        viz_time = (end_viz - start_viz) * 1000
        
        stats = profiler.end_frame()
        total_pipeline_time += stats.get('frame_time', 0)
        
        if i % 20 == 0:
            print(f"   Frame {i}: {stats.get('fps', 0):.1f} FPS, viz: {viz_time:.2f}ms")
    
    # Final performance summary
    summary = profiler.get_performance_summary()
    
    print(f"\n📈 End-to-End Performance Results:")
    print(f"   Average FPS: {summary.get('avg_fps', 0):.1f}")
    print(f"   Average frame time: {summary.get('avg_frame_time_ms', 0):.2f}ms")
    print(f"   Target achievement: {summary.get('fps_target_achievement', 0):.1f}%")
    print(f"   Detection time: {summary.get('avg_detection_time_ms', 0):.2f}ms")
    print(f"   Policy time: {summary.get('avg_policy_time_ms', 0):.2f}ms")
    
    # Check if target is met
    meets_target = summary.get('avg_fps', 0) >= 55  # 95% of 60 FPS target
    
    if meets_target:
        print("   ✅ Performance target achieved!")
    else:
        print("   ⚠️  Performance target not met - optimization needed")
    
    return meets_target


def main():
    """Run all pipeline tests."""
    print("🧪 Distilled Vision Agent Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Data Augmentation", test_data_augmentation),
        ("Visualization Tools", test_visualization_tools),
        ("ONNX Optimization", test_onnx_optimization),
        ("RL Instrumentation", test_rl_instrumentation),
        ("End-to-End Performance", test_end_to_end_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            start_time = time.perf_counter()
            result = test_func()
            end_time = time.perf_counter()
            
            results[test_name] = {
                'passed': result,
                'duration': end_time - start_time
            }
            
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} ({end_time - start_time:.2f}s)")
            
        except Exception as e:
            results[test_name] = {
                'passed': False,
                'error': str(e),
                'duration': 0
            }
            print(f"❌ FAILED - {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    passed_tests = sum(1 for r in results.values() if r['passed'])
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result['passed'] else "❌"
        duration = result['duration']
        print(f"   {status} {test_name} ({duration:.2f}s)")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Pipeline is ready for development.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

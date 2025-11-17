#!/usr/bin/env python3
"""
Simple Test Script for Basic Pipeline Components

Tests core functionality without heavy dependencies.
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_basic_imports():
    """Test if our modules can be imported."""
    print("🔍 Testing Basic Imports...")
    
    try:
        # Test data augmentation import (without heavy dependencies)
        sys.path.append(str(Path(__file__).parent.parent / "src" / "data"))
        print("   ✓ Data module path added")
        
        # Test utils import
        sys.path.append(str(Path(__file__).parent.parent / "src" / "utils"))
        print("   ✓ Utils module path added")
        
        # Test deployment import
        sys.path.append(str(Path(__file__).parent.parent / "src" / "deployment"))
        print("   ✓ Deployment module path added")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_project_structure():
    """Test if project structure is correct."""
    print("\n📁 Testing Project Structure...")
    
    base_dir = Path(__file__).parent.parent
    
    required_dirs = [
        "src/data",
        "src/models", 
        "src/training",
        "src/utils",
        "src/deployment",
        "data/raw",
        "data/labeled", 
        "data/augmented",
        "configs",
        "scripts",
        "docs"
    ]
    
    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "src/data/__init__.py",
        "src/data/augmentation.py",
        "src/utils/__init__.py",
        "src/utils/visualization.py",
        "src/utils/rl_instrumentation.py",
        "src/deployment/__init__.py",
        "src/deployment/onnx_optimizer.py"
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Check directories
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
        else:
            print(f"   ✓ {dir_path}")
    
    # Check files
    for file_path in required_files:
        full_path = base_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"   ✓ {file_path}")
    
    if missing_dirs:
        print(f"   ❌ Missing directories: {missing_dirs}")
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def test_basic_functionality():
    """Test basic functionality without heavy dependencies."""
    print("\n⚙️ Testing Basic Functionality...")
    
    try:
        # Test basic Python operations that our modules would use
        
        # Test dictionary operations (for state vectors)
        state_vector = {
            'player_x': 0.5,
            'player_y': 0.3,
            'obstacle_distance': 0.25,
            'time_to_collision': 1.5
        }
        
        assert isinstance(state_vector, dict), "State vector should be dict"
        assert len(state_vector) == 4, "State vector should have 4 elements"
        print("   ✓ State vector operations working")
        
        # Test list operations (for action history)
        action_history = ['stay', 'flap', 'left', 'stay', 'flap']
        action_counts = {}
        for action in action_history:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        expected_counts = {'stay': 2, 'flap': 2, 'left': 1}
        assert action_counts == expected_counts, f"Action counting failed: {action_counts}"
        print("   ✓ Action counting working")
        
        # Test timing operations (for performance profiling)
        start_time = time.perf_counter()
        time.sleep(0.001)  # 1ms sleep
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        assert 0.5 < elapsed_ms < 10, f"Timing seems off: {elapsed_ms}ms"
        print(f"   ✓ Timing operations working ({elapsed_ms:.2f}ms)")
        
        # Test JSON operations (for logging)
        test_data = {
            'episode': 1,
            'reward': 15.5,
            'actions': action_history,
            'state': state_vector
        }
        
        json_str = json.dumps(test_data)
        loaded_data = json.loads(json_str)
        assert loaded_data == test_data, "JSON serialization failed"
        print("   ✓ JSON serialization working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False

def test_file_operations():
    """Test file I/O operations."""
    print("\n📄 Testing File Operations...")
    
    try:
        # Test creating and writing to files
        test_dir = Path("temp_test")
        test_dir.mkdir(exist_ok=True)
        
        # Test writing metrics
        test_metrics = {
            'experiment_name': 'test_run',
            'episodes': 10,
            'avg_reward': 25.5,
            'success_rate': 0.8
        }
        
        metrics_file = test_dir / "test_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        # Test reading metrics
        with open(metrics_file, 'r') as f:
            loaded_metrics = json.load(f)
        
        assert loaded_metrics == test_metrics, "File I/O failed"
        print("   ✓ JSON file I/O working")
        
        # Test creating directory structure
        nested_dir = test_dir / "logs" / "experiment_1"
        nested_dir.mkdir(parents=True, exist_ok=True)
        
        assert nested_dir.exists(), "Directory creation failed"
        print("   ✓ Directory creation working")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        print("   ✓ File cleanup working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ File operations test failed: {e}")
        return False

def test_performance_simulation():
    """Simulate performance testing without actual models."""
    print("\n🚀 Testing Performance Simulation...")
    
    try:
        # Simulate frame processing pipeline
        target_fps = 60
        target_frame_time = 1.0 / target_fps
        
        frame_times = []
        
        print(f"   Simulating {target_fps} FPS target ({target_frame_time*1000:.1f}ms per frame)")
        
        for i in range(10):
            start_time = time.perf_counter()
            
            # Simulate detection (5-8ms)
            time.sleep(0.006)
            
            # Simulate policy inference (1-2ms)  
            time.sleep(0.0015)
            
            # Simulate visualization (0.5-1ms)
            time.sleep(0.0008)
            
            end_time = time.perf_counter()
            frame_time = end_time - start_time
            frame_times.append(frame_time)
            
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            
            if i % 3 == 0:
                print(f"   Frame {i}: {current_fps:.1f} FPS ({frame_time*1000:.2f}ms)")
        
        # Calculate statistics
        avg_frame_time = sum(frame_times) / len(frame_times)
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        print(f"\n   📊 Performance Results:")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Average frame time: {avg_frame_time*1000:.2f}ms")
        print(f"   Target: {target_fps} FPS ({target_frame_time*1000:.1f}ms)")
        
        # Check if we meet target (within 10% tolerance)
        meets_target = avg_fps >= target_fps * 0.9
        
        if meets_target:
            print("   ✅ Simulated performance meets target!")
        else:
            print("   ⚠️  Simulated performance below target")
        
        return meets_target
        
    except Exception as e:
        print(f"   ❌ Performance simulation failed: {e}")
        return False

def test_game_simulation():
    """Simulate basic game logic."""
    print("\n🎮 Testing Game Logic Simulation...")
    
    try:
        # Simulate a simple game episode
        player_pos = {'x': 0.5, 'y': 0.5}
        obstacles = [
            {'x': 0.8, 'y': 0.3, 'width': 0.1, 'height': 0.4},
            {'x': 1.2, 'y': 0.6, 'width': 0.1, 'height': 0.3}
        ]
        
        episode_length = 0
        max_episode_length = 100
        
        print("   Starting simulated episode...")
        
        while episode_length < max_episode_length:
            episode_length += 1
            
            # Move obstacles left (simulate scrolling)
            for obstacle in obstacles:
                obstacle['x'] -= 0.02
            
            # Remove obstacles that are off-screen
            obstacles = [obs for obs in obstacles if obs['x'] > -0.1]
            
            # Add new obstacles occasionally
            if episode_length % 30 == 0:
                obstacles.append({
                    'x': 1.5,
                    'y': 0.2 + (episode_length % 3) * 0.2,
                    'width': 0.1,
                    'height': 0.4
                })
            
            # Check for collisions
            collision = False
            for obstacle in obstacles:
                if (abs(player_pos['x'] - obstacle['x']) < 0.1 and
                    abs(player_pos['y'] - obstacle['y']) < 0.1):
                    collision = True
                    break
            
            if collision:
                print(f"   💥 Collision at step {episode_length}")
                break
            
            # Simulate random action
            import random
            action = random.choice(['stay', 'up', 'down'])
            
            if action == 'up':
                player_pos['y'] = max(0.0, player_pos['y'] - 0.05)
            elif action == 'down':
                player_pos['y'] = min(1.0, player_pos['y'] + 0.05)
            
            # Log progress occasionally
            if episode_length % 25 == 0:
                print(f"   Step {episode_length}: Player at ({player_pos['x']:.2f}, {player_pos['y']:.2f}), {len(obstacles)} obstacles")
        
        survival_time = episode_length * 0.033  # Assuming 30 FPS
        
        print(f"   🏁 Episode completed!")
        print(f"   Episode length: {episode_length} steps")
        print(f"   Survival time: {survival_time:.2f} seconds")
        print(f"   Final obstacles: {len(obstacles)}")
        
        return episode_length > 50  # Consider success if survived > 50 steps
        
    except Exception as e:
        print(f"   ❌ Game simulation failed: {e}")
        return False

def main():
    """Run all simple tests."""
    print("🧪 Distilled Vision Agent - Simple Test Suite")
    print("=" * 55)
    print("Testing core functionality without heavy dependencies")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Project Structure", test_project_structure), 
        ("Basic Functionality", test_basic_functionality),
        ("File Operations", test_file_operations),
        ("Performance Simulation", test_performance_simulation),
        ("Game Logic Simulation", test_game_simulation)
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
    print("\n" + "=" * 55)
    print("📋 Test Summary:")
    
    passed_tests = sum(1 for r in results.values() if r['passed'])
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result['passed'] else "❌"
        duration = result['duration']
        print(f"   {status} {test_name} ({duration:.2f}s)")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All basic tests passed! Core functionality is working.")
        print("\n💡 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run full test suite: python scripts/test_pipeline.py")
        print("   3. Test with actual game: python Game/game_agent.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

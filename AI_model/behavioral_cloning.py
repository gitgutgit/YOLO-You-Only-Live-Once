# AI_model/behavioral_cloning.py
"""
Stage 1: Policy Distillation via Behavioral Cloning

1. Collect (state, action) data while playing the game with a rule-based expert
2. Train the policy network using supervised learning
3. Use the learned weights as PPO initial values

This approach ensures:
- PPO does not start from a “policy that only plays left”
- It begins from a “balanced expert policy”
- → Enabling learning of a policy that appropriately uses the right action as well

Translated with DeepL.com (free version)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game_core import GameCore
from state_encoder import encode_state, STATE_DIM, ACTION_LIST
from ppo.networks import PolicyNetwork

# ============================================================
# Rule-Based Expert Agent (균형 잡힌 행동)
# ============================================================

class ExpertAgent:
    """
    Rule-based expert that uses BOTH left and right appropriately.
    This is what we want PPO to learn from.
    
    Priority Order:
    1. ACTIVE LAVA (exist_lava) - MUST avoid, 3초간 지속, health 깎임
    2. Close Meteor (dist < 0.25) - 즉시 회피
    3. LAVA WARNING (caution_lava) - 미리 피하기 시작
    4. Stay in center zone
    5. Collect star if safe
    6. Prepare for incoming meteor
    """
    def __init__(self):
        self.name = "Expert (Rule-based)"
        
    def select_action(self, state_vec):
        """
        State format (26-dim):
        [0-1]: player (x, y)
        [2-6]: Meteor 1 (dx, dy, dist, vx, vy)
        [7-11]: Meteor 2 (dx, dy, dist, vx, vy)
        [12-16]: Meteor 3 (dx, dy, dist, vx, vy)
        [17-19]: Star (dx, dy, dist)
        [20-22]: Lava (caution=warning, active=exist, dx)
        [23-25]: ground, reserved, reserved
        """
        player_x = state_vec[0]  # 0~1 normalized
        player_y = state_vec[1]
        on_ground = state_vec[23]
        
        # Get closest meteor info
        meteor1_dx = state_vec[2]   # Positive = meteor is to the RIGHT of player
        meteor1_dy = state_vec[3]   # Positive = meteor is BELOW player
        meteor1_dist = state_vec[4]
        meteor1_vx = state_vec[5]   # Positive = meteor moving RIGHT
        meteor1_vy = state_vec[6]
        
        # Get star info
        star_dx = state_vec[17]
        star_dy = state_vec[18]
        star_dist = state_vec[19]
        
        # Get lava info
        lava_warning = state_vec[20]  # caution_lava (class 3) - warning phase
        lava_active = state_vec[21]   # exist_lava (class 4) - active/damage phase
        lava_dx = state_vec[22]       # lava_zone_x - player_x (positive = lava on RIGHT)
        
        # ===== Priority 1: ACTIVE LAVA - MUST AVOID =====
        # exist_lava는 3초간 지속되고 닿으면 health가 깎임
        # 메테오보다 높은 우선순위 (단, 메테오가 바로 앞이면 trade-off)
        if lava_active == 1.0:
            # 라바 존 안에 있거나 근처에 있으면 즉시 탈출
            # lava_dx: positive = lava on RIGHT, negative = lava on LEFT
            if abs(lava_dx) < 0.4:  # In or near lava zone (zone_width ~= 0.33)
                # 메테오가 매우 가까우면 라바쪽으로 살짝 들어가는 것도 고려
                if meteor1_dist < 0.15:
                    # 메테오와 라바 사이에서 선택해야 함
                    # 메테오 피하는 방향이 라바 반대쪽이면 메테오 피하기
                    meteor_escape_dir = -1 if meteor1_dx > 0 else 1  # -1=left, 1=right
                    lava_escape_dir = -1 if lava_dx > 0 else 1
                    
                    if meteor_escape_dir == lava_escape_dir:
                        # 둘 다 같은 방향 탈출 → 그 방향으로!
                        return 1 if meteor_escape_dir == -1 else 2
                    else:
                        # 메테오 피하면 라바로 가고, 라바 피하면 메테오로 감
                        # → 메테오 우선 (즉시 죽음 vs 서서히 데미지)
                        return 1 if meteor_escape_dir == -1 else 2
                else:
                    # 메테오가 안 가까우면 → 라바 피하기
                    if lava_dx > 0:  # Lava is to the RIGHT
                        return 1  # move LEFT
                    else:  # Lava is to the LEFT
                        return 2  # move RIGHT
        
        # ===== Priority 2: Dodge closest meteor =====
        if meteor1_dist < 0.25:  # Meteor is close!
            # 라바 경고 상태면 라바쪽으로 피하지 않도록 조정
            escape_left = meteor1_dx > 0.05   # meteor on right → escape left
            escape_right = meteor1_dx < -0.05  # meteor on left → escape right
            
            if lava_warning == 1.0 or lava_active == 1.0:
                # 라바 방향 확인
                lava_on_left = lava_dx < 0
                lava_on_right = lava_dx > 0
                
                if escape_left and lava_on_left and abs(lava_dx) < 0.5:
                    # 왼쪽으로 피하고 싶은데 왼쪽에 라바 → 점프 or 오른쪽
                    if on_ground and meteor1_dy > 0:  # meteor below/at player level
                        return 3  # jump over meteor
                    else:
                        return 2  # risk: go right (toward meteor but away from lava)
                elif escape_right and lava_on_right and abs(lava_dx) < 0.5:
                    # 오른쪽으로 피하고 싶은데 오른쪽에 라바
                    if on_ground and meteor1_dy > 0:
                        return 3  # jump
                    else:
                        return 1  # risk: go left
            
            # 일반적인 메테오 회피 (라바 고려 없이)
            if escape_left:
                return 1  # left
            elif escape_right:
                return 2  # right
            else:
                # Meteor is directly above → move to safer side
                if player_x > 0.5:
                    return 1  # left (more space on left)
                else:
                    return 2  # right (more space on right)
        
        # ===== Priority 3: LAVA WARNING - Start avoiding early =====
        # caution_lava는 3초 후에 active가 되니까 미리 피하기 시작
        if lava_warning == 1.0:
            if abs(lava_dx) < 0.5:  # Near warning zone
                # 천천히 반대 방향으로 이동
                if lava_dx > 0:  # Lava warning on RIGHT
                    return 1  # move LEFT
                else:  # Lava warning on LEFT
                    return 2  # move RIGHT
        
        # ===== Priority 4: Stay in center zone =====
        # Target: 0.35 ~ 0.65 (center 30%)
        if player_x < 0.35:
            return 2  # right (move toward center)
        elif player_x > 0.65:
            return 1  # left (move toward center)
        
        # ===== Priority 5: Collect star if safe =====
        if star_dist < 0.4 and meteor1_dist > 0.35:
            # 
            if lava_warning == 1.0 or lava_active == 1.0:
                star_toward_lava = (star_dx > 0 and lava_dx > 0) or (star_dx < 0 and lava_dx < 0)
                if star_toward_lava and abs(lava_dx) < 0.5:
                    pass  # Don't go for star if it's toward lava
                elif star_dx > 0.1:
                    return 2  # right toward star
                elif star_dx < -0.1:
                    return 1  # left toward star
            else:
                if star_dx > 0.1:
                    return 2  # right toward star
                elif star_dx < -0.1:
                    return 1  # left toward star
        
        # ===== Priority 6: Prepare for incoming meteor =====
        if meteor1_dist < 0.4:
            # Predict where meteor will be
            # If meteor coming from right (dx > 0) and moving left (vx < 0)
            if meteor1_dx > 0 and meteor1_vx < 0:
                return 1  # left (get away from its path)
            # If meteor coming from left (dx < 0) and moving right (vx > 0)
            elif meteor1_dx < 0 and meteor1_vx > 0:
                return 2  # right (get away from its path)
        
        # ===== Default: Stay =====
        return 0  # stay


# ============================================================
# Data Collection
# ============================================================

def collect_expert_data(num_episodes=100, max_steps=1000):
    """
    Run expert agent and collect (state, action) pairs.
    """
    print("="*60)
    print("🎮 Collecting Expert Demonstrations")
    print("="*60)
    
    game = GameCore()
    expert = ExpertAgent()
    
    all_states = []
    all_actions = []
    
    action_counts = {i: 0 for i in range(4)}
    total_rewards = []
    total_steps = []
    
    for ep in range(num_episodes):
        game.reset()
        ep_states = []
        ep_actions = []
        ep_reward = 0
        
        for step in range(max_steps):
            # Get state from game (not using YOLO, using ground truth)
            game_state = game._get_state()
            
            # Convert to detection format (simulating perfect YOLO)
            detections = []
            
            # Player
            player = game_state['player']
            detections.append({
                'cls': 0,
                'x': player['x'] / 960,
                'y': player['y'] / 720,
                'w': 50/960,
                'h': 50/720
            })
            
            # Obstacles
            for obs in game_state['obstacles']:
                cls = 1 if obs['type'] == 'meteor' else 2
                detections.append({
                    'cls': cls,
                    'x': obs['x'] / 960,
                    'y': obs['y'] / 720,
                    'w': obs['size'] / 960,
                    'h': obs['size'] / 720
                })
            
            # Lava
            lava = game_state['lava']
            if lava['state'] == 'warning':
                detections.append({
                    'cls': 3,
                    'x': (lava['zone_x'] + lava['zone_width']/2) / 960,
                    'y': (720 - lava['height']/2) / 720,
                    'w': lava['zone_width'] / 960,
                    'h': lava['height'] / 720
                })
            elif lava['state'] == 'active':
                detections.append({
                    'cls': 4,
                    'x': (lava['zone_x'] + lava['zone_width']/2) / 960,
                    'y': (720 - lava['height']/2) / 720,
                    'w': lava['zone_width'] / 960,
                    'h': lava['height'] / 720
                })
            
            # Encode state
            state_vec = encode_state(detections, game_state)
            
            # Expert selects action
            action = expert.select_action(state_vec)
            action_str = ACTION_LIST[action]
            
            # Store
            ep_states.append(state_vec)
            ep_actions.append(action)
            action_counts[action] += 1
            
            # Step game
            _, reward, done, _ = game.step(action_str)
            ep_reward += reward
            
            if done:
                break
        
        all_states.extend(ep_states)
        all_actions.extend(ep_actions)
        total_rewards.append(ep_reward)
        total_steps.append(len(ep_states))
        
        if (ep + 1) % 10 == 0:
            avg_rew = np.mean(total_rewards[-10:])
            avg_steps = np.mean(total_steps[-10:])
            print(f"Episode {ep+1}/{num_episodes} | Avg Reward: {avg_rew:.1f} | Avg Steps: {avg_steps:.0f}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Data Collection Summary")
    print("="*60)
    print(f"Total samples: {len(all_states)}")
    print(f"Average episode reward: {np.mean(total_rewards):.2f}")
    print(f"Average episode steps: {np.mean(total_steps):.1f}")
    
    total = sum(action_counts.values())
    print(f"\nAction Distribution:")
    for i, name in enumerate(ACTION_LIST):
        pct = action_counts[i] / total * 100
        print(f"  {name}: {action_counts[i]} ({pct:.1f}%)")
    
    return np.array(all_states), np.array(all_actions)


# ============================================================
# Behavioral Cloning Training
# ============================================================

def train_bc(states, actions, epochs=100, batch_size=256, lr=1e-3):
    """
    Train policy network to imitate expert actions.
    """
    print("\n" + "="*60)
    print("🎓 Training Behavioral Cloning Policy")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create policy network (same architecture as PPO)
    policy = PolicyNetwork(STATE_DIM, len(ACTION_LIST)).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)
    
    # Split train/val
    n = len(states)
    indices = np.random.permutation(n)
    train_idx = indices[:int(0.9*n)]
    val_idx = indices[int(0.9*n):]
    
    train_states = states_tensor[train_idx]
    train_actions = actions_tensor[train_idx]
    val_states = states_tensor[val_idx]
    val_actions = actions_tensor[val_idx]
    
    print(f"Train samples: {len(train_states)}")
    print(f"Val samples: {len(val_states)}")
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        policy.train()
        
        # Shuffle
        perm = torch.randperm(len(train_states))
        train_states = train_states[perm]
        train_actions = train_actions[perm]
        
        total_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(train_states), batch_size):
            batch_s = train_states[i:i+batch_size]
            batch_a = train_actions[i:i+batch_size]
            
            # Forward
            probs = policy(batch_s)
            loss = criterion(probs, batch_a)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_s)
            
            # Accuracy
            pred = probs.argmax(dim=1)
            correct += (pred == batch_a).sum().item()
            total += len(batch_a)
        
        train_loss = total_loss / total
        train_acc = correct / total
        
        # Validation
        policy.eval()
        with torch.no_grad():
            val_probs = policy(val_states)
            val_loss = criterion(val_probs, val_actions).item()
            val_pred = val_probs.argmax(dim=1)
            val_acc = (val_pred == val_actions).float().mean().item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(policy.state_dict(), "bc_policy_best.pt")
    
    print(f"\n✅ Best Validation Accuracy: {best_val_acc:.3f}")
    print(f"💾 Saved to bc_policy_best.pt")
    
    return policy


# ============================================================
# Evaluate BC Policy
# ============================================================

def evaluate_bc(policy, num_episodes=20):
    """
    Test BC policy in the game.
    """
    print("\n" + "="*60)
    print("🎮 Evaluating BC Policy")
    print("="*60)
    
    device = next(policy.parameters()).device
    policy.eval()
    
    game = GameCore()
    
    action_counts = {i: 0 for i in range(4)}
    total_rewards = []
    total_steps = []
    
    for ep in range(num_episodes):
        game.reset()
        ep_reward = 0
        
        for step in range(1000):
            # Get state (same as collection)
            game_state = game._get_state()
            detections = []
            
            player = game_state['player']
            detections.append({
                'cls': 0, 'x': player['x']/960, 'y': player['y']/720,
                'w': 50/960, 'h': 50/720
            })
            
            for obs in game_state['obstacles']:
                cls = 1 if obs['type'] == 'meteor' else 2
                detections.append({
                    'cls': cls, 'x': obs['x']/960, 'y': obs['y']/720,
                    'w': obs['size']/960, 'h': obs['size']/720
                })
            
            lava = game_state['lava']
            if lava['state'] in ['warning', 'active']:
                cls = 3 if lava['state'] == 'warning' else 4
                detections.append({
                    'cls': cls,
                    'x': (lava['zone_x'] + lava['zone_width']/2) / 960,
                    'y': (720 - lava['height']/2) / 720,
                    'w': lava['zone_width'] / 960,
                    'h': lava['height'] / 720
                })
            
            state_vec = encode_state(detections, game_state)
            
            # Policy forward
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
                probs = policy(state_tensor)
                action = probs.argmax(dim=1).item()
            
            action_counts[action] += 1
            action_str = ACTION_LIST[action]
            
            _, reward, done, _ = game.step(action_str)
            ep_reward += reward
            
            if done:
                break
        
        total_rewards.append(ep_reward)
        total_steps.append(step + 1)
    
    # Summary
    print(f"\nResults over {num_episodes} episodes:")
    print(f"  Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Average Steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    
    total = sum(action_counts.values())
    print(f"\nAction Distribution:")
    for i, name in enumerate(ACTION_LIST):
        pct = action_counts[i] / total * 100
        print(f"  {name}: {action_counts[i]} ({pct:.1f}%)")
    
    return np.mean(total_rewards), np.mean(total_steps)


# ============================================================
# Create PPO-compatible checkpoint from BC
# ============================================================

def create_ppo_checkpoint_from_bc(bc_policy_path="bc_policy_best.pt", 
                                   output_path="ppo_agent_from_bc.pt"):
    """
    Convert BC policy weights to PPO checkpoint format.
    This allows PPO to start from the BC-trained policy.
    """
    print("\n" + "="*60)
    print("🔄 Converting BC Policy to PPO Format")
    print("="*60)
    
    from ppo.networks import PolicyNetwork, ValueNetwork
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load BC policy
    policy = PolicyNetwork(STATE_DIM, len(ACTION_LIST)).to(device)
    policy.load_state_dict(torch.load(bc_policy_path, map_location=device))
    
    # Create fresh value network
    value_net = ValueNetwork(STATE_DIM).to(device)
    
    # Create PPO checkpoint
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'policy_old_state_dict': policy.state_dict(),  # Same as policy initially
        'value_net_state_dict': value_net.state_dict(),
        'optimizer_state_dict': optim.Adam(policy.parameters(), lr=1e-4).state_dict(),
        'value_optimizer_state_dict': optim.Adam(value_net.parameters(), lr=1e-4).state_dict(),
        'state_dim': STATE_DIM,
        'action_dim': len(ACTION_LIST),
        'lr': 1e-4,
        'gamma': 0.99,
        'eps_clip': 0.2,
        'K_epochs': 10
    }
    
    torch.save(checkpoint, output_path)
    print(f"✅ Saved PPO-compatible checkpoint to {output_path}")
    print(f"   Now you can run: python train_ppo.py --resume {output_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Collect expert data")
    parser.add_argument("--train", action="store_true", help="Train BC policy")
    parser.add_argument("--eval", action="store_true", help="Evaluate BC policy")
    parser.add_argument("--convert", action="store_true", help="Convert BC to PPO format")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes for data collection")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for BC training")
    
    args = parser.parse_args()
    
    if args.all or (not any([args.collect, args.train, args.eval, args.convert])):
        args.collect = args.train = args.eval = args.convert = True
    
    if args.collect:
        states, actions = collect_expert_data(num_episodes=args.episodes)
        np.save("bc_states.npy", states)
        np.save("bc_actions.npy", actions)
        print(f"💾 Saved to bc_states.npy, bc_actions.npy")
    
    if args.train:
        if not os.path.exists("bc_states.npy"):
            print("❌ No data found. Run --collect first.")
        else:
            states = np.load("bc_states.npy")
            actions = np.load("bc_actions.npy")
            policy = train_bc(states, actions, epochs=args.epochs)
    
    if args.eval:
        if not os.path.exists("bc_policy_best.pt"):
            print("❌ No BC policy found. Run --train first.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            policy = PolicyNetwork(STATE_DIM, len(ACTION_LIST)).to(device)
            policy.load_state_dict(torch.load("bc_policy_best.pt", map_location=device))
            evaluate_bc(policy)
    
    if args.convert:
        if not os.path.exists("bc_policy_best.pt"):
            print("❌ No BC policy found. Run --train first.")
        else:
            create_ppo_checkpoint_from_bc()
    
    print("\n" + "="*60)
    print("🎉 Done!")
    print("="*60)
    print("\nNext steps:")
    print("1. python behavioral_cloning.py --all")
    print("2. python train_ppo.py  # Will start from BC weights")
    print("   OR")
    print("   python train_ppo.py --resume ppo_agent_from_bc.pt")

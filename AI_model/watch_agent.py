# AI_model/watch_agent.py
import os
import time
import argparse
import numpy as np

from game_core import GameCore
from state_encoder import encode_state, ACTION_LIST, STATE_DIM

import cv2

import torch
from ultralytics import YOLO

# can be replaced via args
YOLO_MODEL_PATH = "yolo_fine.pt"          # fine-tuning model
PPO_MODEL_PATH = "ppo_agent.pt"           # trained ppo

#  Action index → String mapping 
# 0: stay, 1: left, 2: right, 3: jump
IDX2ACTION = ACTION_LIST


def load_yolo(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found at {model_path}")
    print(f"✅ Loading YOLO model from {model_path}")
    return YOLO(model_path)


def load_ppo(model_path: str):
    """Load trained PPO agent - 새 checkpoint 형식 지원"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO agent not found at {model_path}")
    print(f"✅ Loading PPO agent from {model_path}")
    
    # checkpoint 로드해서 형식 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 새 형식인지 확인 (lr 키가 없으면 새 형식)
    if 'lr' in checkpoint:
        # 기존 형식: PPOAgent.load() 사용
        from ppo.agent import PPOAgent
        agent = PPOAgent.load(model_path)
        return agent
    else:
        # 새 형식: 직접 로드
        print("   📂 New checkpoint format detected")
        from ppo.agent import PPOAgent
        
        # state/action 차원
        state_dim = checkpoint.get('state_dim', STATE_DIM)
        action_dim = checkpoint.get('action_dim', len(ACTION_LIST))
        
        # 기본 하이퍼파라미터로 agent 생성
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=0.0001,
            gamma=0.95,
            eps_clip=0.2,
            K_epochs=10
        )
        
        # weight 로드
        if 'policy_state_dict' in checkpoint:
            agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            agent.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        if 'value_net_state_dict' in checkpoint:
            agent.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        
        print(f"   ✅ Loaded: state_dim={state_dim}, action_dim={action_dim}")
        return agent


def run_visualization(yolo_path=None, ppo_path=None):
    # path 
    yolo_model_path = yolo_path or YOLO_MODEL_PATH
    ppo_model_path = ppo_path or PPO_MODEL_PATH
    
    # 1. Game/model load 
    game = GameCore()
    yolo_model = load_yolo(yolo_model_path)
    ppo_agent = load_ppo(ppo_model_path)
    
    action_counts = {name: 0 for name in ACTION_LIST}
    
    # 2. initialize
    game.reset()
    fps_delay = 1.0 / 30.0  # 30 FPS 
    
    episode_count = 0
    total_reward = 0.0
    episode_reward = 0.0

    print("🎮 Game Start! (q: terminate)")
    step_count = 0
    
    while True:
        # --- 1)current frame render ---
        img = game.render()              # (H, W, 3) numpy array, RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # --- 2) YOLO model ---
        results = yolo_model(img, verbose=False)

        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()   # normalized
                conf = float(box.conf[0])  # confidence
                detections.append({'cls': cls, 'x': x, 'y': y, 'w': w, 'h': h, 'conf': conf})
                print(cls)
                # DEBUG: Lava detection logging
                if cls == 3:
                    print(f"   🟡 YOLO detected: caution_lava (cls=3) at x={x:.2f}, conf={conf:.2f}")
                elif cls == 4:
                    print(f"   🔴 YOLO detected: exist_lava (cls=4) at x={x:.2f}, conf={conf:.2f}")

        # --- 3) state encoding ---
        game_state = game._get_state()
        state_vec = encode_state(detections, game_state)

        # --- 4) PPO choose behaviour (eval mode ) ---
        action_idx = ppo_agent.select_action_eval(state_vec)
        action_str = IDX2ACTION[action_idx]
        action_counts[action_str] += 1

        # --- 5) one step process ---
        _, reward, done, _ = game.step(action_str)
        episode_reward += reward

        step_count += 1

        #  Debugging:
        if step_count % 10 == 0:
            print(f"\n📊 Step {step_count}")
            print(f"   Player: x={state_vec[0]:.2f}, y={state_vec[1]:.2f}")
            # New Indices (26-dim): Meteor 1 is at [2-6] (dx, dy, dist, vx, vy)
            print(f"   Meteor 1: dx={state_vec[2]:.2f}, dy={state_vec[3]:.2f}, dist={state_vec[4]:.2f}, vx={state_vec[5]:.2f}, vy={state_vec[6]:.2f}")
            print(f"   Meteor 2: dist={state_vec[9]:.2f}")
            print(f"   Meteor 3: dist={state_vec[14]:.2f}")
            # Star info (indices 17-19)
            print(f"   Star: dx={state_vec[17]:.2f}, dy={state_vec[18]:.2f}, dist={state_vec[19]:.2f}")
            # Lava info (indices 20-22)
            lava_warning = state_vec[20]
            lava_active = state_vec[21]
            lava_dx = state_vec[22]
            lava_status = "ACTIVE🔥" if lava_active else ("WARNING⚠️" if lava_warning else "inactive")
            print(f"   Lava: status={lava_status}, dx={lava_dx:.2f}")
            print(f"   On Ground: {state_vec[23]:.0f}") # Index 23 is ground
            print(f"   Action: {action_str}")
            print(f"   Reward: {reward:.2f}")

        # 메테오가 가까우면 워닝만 출력 (행동은 PPO 결과 그대로 사용)
        # Check all 3 meteors (indices: 4, 9, 14)
        min_dist = min(state_vec[4], state_vec[9], state_vec[14])
        if min_dist < 0.15:
            print(f"   ⚠️ METEOR CLOSE! (dist={min_dist:.2f}) → Action: {action_str}")
        
        # 라바가 활성화되면 워닝 출력
        if state_vec[21] == 1.0:  # lava_active
            lava_dx = state_vec[22]
            if abs(lava_dx) < 0.4:
                print(f"   🔥 LAVA ACTIVE NEARBY! (dx={lava_dx:.2f}) → Action: {action_str}")
        elif state_vec[20] == 1.0:  # lava_warning
            lava_dx = state_vec[22]
            if abs(lava_dx) < 0.5:
                print(f"   ⚠️ LAVA WARNING! (dx={lava_dx:.2f}) → Action: {action_str}")

        # --- 6) 시각화용 박스/텍스트 오버레이 ---
        H, W, _ = img_bgr.shape
        for d in detections:
            cls = d["cls"]
            cx = d["x"] * W
            cy = d["y"] * H
            w = d["w"] * W
            h = d["h"] * H
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            if cls == 0:
                color = (255, 0, 0)       # Player
                label = "Player"
            elif cls == 1:
                color = (0, 0, 255)       # Meteor
                label = "Meteor"
            elif cls == 2:
                color = (0, 255, 255)     # Star
                label = "Star"
            # elif cls == 3:
            #     color = (0, 165, 255)     # Caution Lava
            #     label = "Caution Lava"
            elif cls == 3 or cls ==4:
                color = (0, 140, 255)     # Lava
                label = "Lava"
            else:
                color = (255, 255, 255)
                label = f"cls{cls}"

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img_bgr,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # Current reaction & Reward
        cv2.putText(
            img_bgr,
            f"Action: {action_str}   Reward: {reward:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Score & Episdoe information
        cv2.putText(
            img_bgr,
            f"Score: {game.score}   Episode: {episode_count}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # HPbar
        cv2.putText(
            img_bgr,
            f"Health: {game.player_health}/100",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --- 7) 화면 출력 ---
        cv2.imshow("PPO Agent Playing (Vision-based)", img_bgr)

        key = cv2.waitKey(int(fps_delay * 1000))
        if key == ord('q'):
            break

        if done:
            # 에피소드 종료
            episode_count += 1
            total_reward += episode_reward
            avg_reward = total_reward / episode_count if episode_count > 0 else 0.0
            
            print(f"💀 Episode {episode_count} finished!")
            print(f"   Score: {game.score}")
            print(f"   Episode Reward: {episode_reward:.2f}")
            print(f"   Average Reward: {avg_reward:.2f}")
            print(f"   Resetting in 1.5s...")
            print(f"   Actions: {action_counts}")
            time.sleep(1.5)
            game.reset()
            episode_reward = 0.0
            

    cv2.destroyAllWindows()
    
    # ✅ stat print
    print("\n" + "="*50)
    print("🎮 Game Statistics")
    print("="*50)
    print(f"Total Episodes: {episode_count}")
    avg_ep_reward = total_reward / episode_count if episode_count > 0 else 0.0
    print(f"Average Reward: {avg_ep_reward:.2f}")
    print(f"Action Counts: {action_counts}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch PPO Agent Play")
    parser.add_argument("--model", type=str, default="ppo_agent.pt",
                        help="PPO model path (e.g., ppo_agent_ep100.pt)")
    parser.add_argument("--yolo", type=str, default="yolo_fine.pt",
                        help="YOLO model path")
    args = parser.parse_args()
    
    run_visualization(yolo_path=args.yolo, ppo_path=args.model)

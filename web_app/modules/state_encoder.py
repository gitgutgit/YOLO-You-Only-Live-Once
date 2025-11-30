# state_encoder.py
# YOLO detection + game_state → 26-dim state vector 변환
# PPO/BC 학습 및 추론에 사용

import numpy as np

WIDTH = 960
HEIGHT = 720
PLAYER_SIZE = 50
OBSTACLE_SIZE = 50

# 액션 순서: 환경 / PPO / 시각화 전부 이 순서로 통일
# 0: stay, 1: left, 2: right, 3: jump
ACTION_LIST = ["stay", "left", "right", "jump"]

STATE_DIM = 26  # 26-dim state vector

def encode_state(detections: list, game_state: dict = None) -> np.ndarray:
    """
    Encode YOLO detections (or game_state) into a state vector.
    
    Vector Layout (26 dims):
    [0-1]: Player (x, y) - normalized 0~1
    [2-6]: Meteor 1 (dx, dy, dist, vx, vy) - closest
    [7-11]: Meteor 2 (dx, dy, dist, vx, vy)
    [12-16]: Meteor 3 (dx, dy, dist, vx, vy)
    [17-19]: Star (dx, dy, dist)
    [20-22]: Lava (caution, exist, dx)
    [23]: Ground (1 if on ground)
    [24-25]: Reserved/Padding
    
    Args:
        detections: YOLO detection results [{'cls': int, 'x': float, 'y': float, 'w': float, 'h': float}, ...]
        game_state: Optional game state dict for velocity info
        
    Returns:
        26-dim numpy array (float32)
    """
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    
    # ===== Find Player (Class 0) =====
    player = None
    for d in detections:
        if d['cls'] == 0:
            player = d
            break
            
    if player:
        vec[0] = player['x']  # Center X (normalized 0~1)
        vec[1] = player['y']  # Center Y (normalized 0~1)
        
        # On Ground check (approximate)
        if player['y'] + player['h'] / 2 >= 0.92:
            vec[23] = 1.0
    else:
        # Player not found - use game_state if available
        if game_state and 'player' in game_state:
            p = game_state['player']
            vec[0] = p.get('x', WIDTH // 2) / WIDTH
            vec[1] = p.get('y', HEIGHT // 2) / HEIGHT
            if p.get('y', 0) >= HEIGHT - PLAYER_SIZE - 10:
                vec[23] = 1.0
        else:
            vec[0] = 0.5
            vec[1] = 0.5

    player_x = vec[0]
    player_y = vec[1]
    
    # ===== Process Meteors (Class 1) =====
    meteors = []
    for d in detections:
        if d['cls'] == 1:
            dx = d['x'] - player_x
            dy = d['y'] - player_y
            dist = np.sqrt(dx**2 + dy**2)
            
            # Extract velocity from game_state if available
            vx, vy = 0.0, 0.0
            if game_state and 'obstacles' in game_state:
                for obs in game_state['obstacles']:
                    if obs.get('type') == 'meteor':
                        obs_x_norm = obs['x'] / WIDTH
                        obs_y_norm = obs['y'] / HEIGHT
                        if abs(obs_x_norm - d['x']) < 0.08 and abs(obs_y_norm - d['y']) < 0.08:
                            vx = np.clip(obs.get('vx', 0) / 10.0, -1, 1)
                            vy = np.clip(obs.get('vy', 5) / 10.0, -1, 1)
                            break
            
            meteors.append((dist, dx, dy, vx, vy))
            
    # Sort by distance (ascending) and take top 3
    meteors.sort(key=lambda x: x[0])
    
    # Fill Meteor Slots (Indices 2-16)
    for i in range(3):
        base_idx = 2 + i * 5  # (dx, dy, dist, vx, vy)
        if i < len(meteors):
            dist, dx, dy, vx, vy = meteors[i]
            vec[base_idx] = np.clip(dx, -1, 1)
            vec[base_idx+1] = np.clip(dy, -1, 1)
            vec[base_idx+2] = np.clip(dist, 0, 1.5)
            vec[base_idx+3] = vx
            vec[base_idx+4] = vy
        else:
            # No meteor found for this slot -> set dist to 1.0 (far)
            vec[base_idx+2] = 1.0

    # ===== Process Star (Class 2) =====
    nearest_star_dist = 1.0
    for d in detections:
        if d['cls'] == 2:
            dx = d['x'] - player_x
            dy = d['y'] - player_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < nearest_star_dist:
                nearest_star_dist = dist
                vec[17] = np.clip(dx, -1, 1)
                vec[18] = np.clip(dy, -1, 1)
                vec[19] = np.clip(dist, 0, 1)
    if nearest_star_dist == 1.0:
        vec[19] = 1.0  # No star

    # ===== Process Lava (Class 3, 4) =====
    for d in detections:
        if d['cls'] == 3:  # Caution Lava (warning)
            vec[20] = 1.0
            vec[22] = np.clip(d['x'] - player_x, -1, 1)
        elif d['cls'] == 4:  # Exist Lava (active)
            vec[21] = 1.0
            vec[22] = np.clip(d['x'] - player_x, -1, 1)

    return vec


def game_state_to_detections(game_state: dict) -> list:
    """
    Convert game_state (from app.py) to YOLO-like detection format.
    
    This allows using the same encode_state() function for:
    1. Real YOLO inference (watch_agent.py)
    2. Simulation mode (app.py without YOLO)
    
    Args:
        game_state: Game state dict from app.py's Game.get_state()
        
    Returns:
        List of detection dicts [{'cls': int, 'x': float, 'y': float, 'w': float, 'h': float}, ...]
    """
    detections = []
    
    # Player (Class 0)
    player = game_state.get('player', {})
    if player:
        player_x = player.get('x', WIDTH // 2)
        player_y = player.get('y', HEIGHT // 2)
        player_size = player.get('size', PLAYER_SIZE)
        
        detections.append({
            'cls': 0,
            'x': (player_x + player_size / 2) / WIDTH,  # center x normalized
            'y': (player_y + player_size / 2) / HEIGHT,  # center y normalized
            'w': player_size / WIDTH,
            'h': player_size / HEIGHT
        })
    
    # Obstacles (Meteors: Class 1, Stars: Class 2)
    obstacles = game_state.get('obstacles', [])
    for obs in obstacles:
        obj_type = obs.get('type', 'meteor')
        obs_x = obs.get('x', 0)
        obs_y = obs.get('y', 0)
        obs_size = obs.get('size', OBSTACLE_SIZE)
        
        cls = 1 if obj_type == 'meteor' else 2  # meteor=1, star=2
        
        detections.append({
            'cls': cls,
            'x': (obs_x + obs_size / 2) / WIDTH,
            'y': (obs_y + obs_size / 2) / HEIGHT,
            'w': obs_size / WIDTH,
            'h': obs_size / HEIGHT
        })
    
    # Lava (Caution: Class 3, Active: Class 4)
    lava = game_state.get('lava', {})
    lava_state = lava.get('state', 'inactive')
    
    if lava_state in ['warning', 'active']:
        lava_zone_x = lava.get('zone_x', 0)
        lava_zone_width = lava.get('zone_width', 320)
        lava_height = lava.get('height', 120)
        
        cls = 3 if lava_state == 'warning' else 4
        
        detections.append({
            'cls': cls,
            'x': (lava_zone_x + lava_zone_width / 2) / WIDTH,
            'y': (HEIGHT - lava_height / 2) / HEIGHT,
            'w': lava_zone_width / WIDTH,
            'h': lava_height / HEIGHT
        })
    
    return detections

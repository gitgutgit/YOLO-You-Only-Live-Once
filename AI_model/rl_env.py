# AI_model/rl_env.py
import numpy as np
import os
from game_core import GameCore
from state_encoder import encode_state, STATE_DIM, ACTION_LIST

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics not installed. Vision-based RL will not work.")

class GameEnv:
    """
    Gym-like environment wrapper for GameCore.

    - use_yolo=True: game.render() + YOLO → detections → encode_state
    - use_yolo=False: GameCore._get_state()에서 직접 detections 생성 → encode_state
      (BC와 동일한 구조, 완전한 월드 정보 기반)
    """
    def __init__(self, model_path='yolo_fine.pt', use_yolo=True):
        self.game = GameCore()
        
        # ✅ 액션/상태 공간
        self.action_space = len(ACTION_LIST)   # 4
        self.observation_space = STATE_DIM

        self.use_yolo = use_yolo

        self.yolo_model = None
        if self.use_yolo:
            if YOLO_AVAILABLE and os.path.exists(model_path):
                print(f"✅ [GameEnv] Loading YOLO model: {model_path}")
                self.yolo_model = YOLO(model_path)
            else:
                print("⚠️ [GameEnv] YOLO model unavailable. Falling back to oracle (no_yolo) mode.")
                self.use_yolo = False

    # ---------- 공통 helper: GT state에서 detections 생성 ----------
    def _encode_from_gt(self):
        """
        GameCore._get_state()를 사용해서
        BC와 동일한 방식으로 detections를 만든 뒤 encode_state에 넣는다.
        """
        game_state = self.game._get_state()
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

        # Obstacles (meteor, star)
        for obs in game_state['obstacles']:
            cls = 1 if obs['type'] == 'meteor' else 2  # 1: meteor, 2: star
            detections.append({
                'cls': cls,
                'x': obs['x'] / 960,
                'y': obs['y'] / 720,
                'w': obs['size'] / 960,
                'h': obs['size'] / 720
            })

        # Lava (warning / active)
        lava = game_state['lava']
        if lava['state'] == 'warning':
            detections.append({
                'cls': 3,  # caution_lava
                'x': (lava['zone_x'] + lava['zone_width']/2) / 960,
                'y': (720 - lava['height']/2) / 720,
                'w': lava['zone_width'] / 960,
                'h': lava['height'] / 720
            })
        elif lava['state'] == 'active':
            detections.append({
                'cls': 4,  # exist_lava
                'x': (lava['zone_x'] + lava['zone_width']/2) / 960,
                'y': (720 - lava['height']/2) / 720,
                'w': lava['zone_width'] / 960,
                'h': lava['height'] / 720
            })

        # 디버그용
        # print(f"[DEBUG GT] lava_state={lava['state']} detections={len(detections)}")

        return encode_state(detections, game_state)

    # ---------- YOLO / Oracle 공통 인터페이스 ----------
    def _get_state_vec(self):
        """
        현재 설정에 맞는 state vector 반환:
        - use_yolo=True  → render() + YOLO
        - use_yolo=False → _encode_from_gt()
        """
        if not self.use_yolo:
            return self._encode_from_gt()

        # YOLO 사용 모드
        img = self.game.render()
        game_state = self.game._get_state()
        
        detections = []
        if self.yolo_model:
            results = self.yolo_model(img, verbose=False)
            for box in results[0].boxes:
                cls = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()
                detections.append({'cls': cls, 'x': x, 'y': y, 'w': w, 'h': h})
        else:
            # 이 경우는 거의 안 오겠지만 safety
            return encode_state([], game_state)
            
        return encode_state(detections, game_state)

    def reset(self):
        """
        Reset the environment.
        Returns:
            observation (np.ndarray): Initial state vector
        """
        self.game.reset()
        return self._get_state_vec()
        
    def step(self, action_idx):
        """
        Execute action.
        Args:
            action_idx (int): 0: stay, 1: left, 2: right, 3: jump
            
        Returns:
            observation (np.ndarray): Next state vector
            reward (float): Reward
            done (bool): Whether episode is done
            info (dict): Extra info
        """
        assert 0 <= action_idx < len(ACTION_LIST), f"Invalid action index: {action_idx}"
        action_str = ACTION_LIST[action_idx]
        
        _, reward, done, info = self.game.step(action_str)
        next_state_vec = self._get_state_vec()
        
        return next_state_vec, reward, done, info

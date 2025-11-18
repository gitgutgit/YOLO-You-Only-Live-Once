"""
Training Data Collector - 웹 게임에서 훈련 데이터 수집

GCP 웹에서 유저들이 플레이한 데이터를 수집하고 저장
제이와 클로가 훈련에 사용할 수 있도록 처리

Author: Minsuk Kim (mk4434)
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Any


class TrainingDataCollector:
    """
    훈련 데이터 수집기
    
    역할:
    1. 웹 게임에서 생성된 데이터 수집
    2. 제이(CV)와 클로(AI)가 사용할 형식으로 저장
    3. 데이터 품질 관리 및 통계
    """
    
    def __init__(self, data_dir: str = "data/gameplay"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장 경로
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cv_training_dir = self.data_dir / "cv_training"  # 제이용
        self.rl_training_dir = self.data_dir / "rl_training"  # 클로용
        
        for dir_path in [self.raw_dir, self.processed_dir, 
                         self.cv_training_dir, self.rl_training_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 통계
        self.stats = {
            'total_sessions': 0,
            'total_frames': 0,
            'total_actions': 0,
            'human_sessions': 0,
            'ai_sessions': 0
        }
        
        self.load_stats()
    
    def save_gameplay_session(self, session_data: Dict[str, Any]) -> str:
        """
        게임플레이 세션 저장
        
        Args:
            session_data: 게임 세션 데이터
            
        Returns:
            저장된 파일 경로
        """
        # 세션 ID 생성
        session_id = session_data.get('sessionId', f"session_{int(time.time())}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = session_data.get('mode', 'unknown')
        
        # Raw 데이터 저장
        raw_file = self.raw_dir / f"{timestamp}_{session_id}_{mode}.json"
        with open(raw_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # 통계 업데이트
        self.update_stats(session_data)
        
        # 훈련 데이터로 변환
        self.process_for_training(session_data, session_id, timestamp)
        
        print(f"✅ 세션 저장 완료: {raw_file.name}")
        print(f"   - 프레임: {len(session_data.get('frames', []))}")
        print(f"   - 액션: {len(session_data.get('actions', []))}")
        print(f"   - 모드: {mode}")
        
        return str(raw_file)
    
    def process_for_training(self, session_data: Dict, session_id: str, timestamp: str):
        """
        훈련 데이터로 변환 및 저장
        
        제이(CV)와 클로(AI)가 바로 사용할 수 있는 형식으로 변환
        """
        mode = session_data.get('mode', 'unknown')
        
        # 1. CV 훈련 데이터 (제이용)
        self.save_cv_training_data(session_data, session_id, timestamp, mode)
        
        # 2. RL 훈련 데이터 (클로용)
        self.save_rl_training_data(session_data, session_id, timestamp, mode)
    
    def save_cv_training_data(self, session_data: Dict, session_id: str, 
                              timestamp: str, mode: str):
        """
        CV 훈련 데이터 저장 (제이가 YOLOv8 훈련에 사용)
        
        형식:
        - frames/: 프레임 이미지들 (나중에 Canvas에서 캡처)
        - annotations.json: YOLO 형식 어노테이션
        """
        cv_file = self.cv_training_dir / f"{timestamp}_{session_id}_cv.json"
        
        # 프레임별 객체 위치 저장
        cv_data = {
            'session_id': session_id,
            'timestamp': timestamp,
            'mode': mode,
            'frames': []
        }
        
        for frame in session_data.get('frames', []):
            game_state = frame.get('gameState', {})
            
            # YOLO 형식으로 변환 (x_center, y_center, width, height, normalized)
            annotations = []
            
            # 플레이어
            if 'player' in game_state:
                player = game_state['player']
                annotations.append({
                    'class': 0,  # Player
                    'class_name': 'player',
                    'bbox': self.normalize_bbox(
                        player.get('x', 0), 
                        player.get('y', 0), 
                        40, 40  # 플레이어 크기
                    )
                })
            
            # 장애물들
            for obs in game_state.get('obstacles', []):
                annotations.append({
                    'class': 1,  # Obstacle
                    'class_name': 'obstacle',
                    'bbox': self.normalize_bbox(
                        obs.get('x', 0), 
                        obs.get('y', 0), 
                        40, 40  # 장애물 크기
                    )
                })
            
            cv_data['frames'].append({
                'frame_id': frame.get('timestamp'),
                'annotations': annotations
            })
        
        with open(cv_file, 'w') as f:
            json.dump(cv_data, f, indent=2)
        
        print(f"   📸 CV 훈련 데이터 저장: {len(cv_data['frames'])} frames")
    
    def save_rl_training_data(self, session_data: Dict, session_id: str, 
                              timestamp: str, mode: str):
        """
        RL 훈련 데이터 저장 (클로가 PPO/DQN 훈련에 사용)
        
        형식:
        - state-action-reward-next_state (SARS) 튜플들
        - 에피소드 정보
        """
        rl_file = self.rl_training_dir / f"{timestamp}_{session_id}_rl.json"
        
        rl_data = {
            'session_id': session_id,
            'timestamp': timestamp,
            'mode': mode,
            'episode': {
                'total_reward': session_data.get('finalScore', 0),
                'steps': len(session_data.get('frames', [])),
                'survival_time': session_data.get('finalSurvivalTime', 0)
            },
            'transitions': []
        }
        
        frames = session_data.get('frames', [])
        actions = session_data.get('actions', [])
        
        # State-Action-Reward-Next_State 튜플 생성
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            
            # 해당 프레임의 액션 찾기
            action = self.find_action_for_frame(
                current_frame.get('timestamp'),
                actions
            )
            
            # 보상 계산
            reward = self.calculate_reward(
                current_frame.get('gameState', {}),
                next_frame.get('gameState', {})
            )
            
            transition = {
                'state': self.extract_state_vector(current_frame.get('gameState', {})),
                'action': action,
                'reward': reward,
                'next_state': self.extract_state_vector(next_frame.get('gameState', {})),
                'done': next_frame.get('gameState', {}).get('game_over', False)
            }
            
            rl_data['transitions'].append(transition)
        
        with open(rl_file, 'w') as f:
            json.dump(rl_data, f, indent=2)
        
        print(f"   🤖 RL 훈련 데이터 저장: {len(rl_data['transitions'])} transitions")
    
    def normalize_bbox(self, x: float, y: float, w: float, h: float) -> List[float]:
        """
        바운딩 박스를 YOLO 형식으로 정규화
        
        Returns:
            [x_center, y_center, width, height] (0-1 범위)
        """
        canvas_width = 640
        canvas_height = 480
        
        x_center = (x + w / 2) / canvas_width
        y_center = (y + h / 2) / canvas_height
        norm_width = w / canvas_width
        norm_height = h / canvas_height
        
        return [x_center, y_center, norm_width, norm_height]
    
    def extract_state_vector(self, game_state: Dict) -> List[float]:
        """
        게임 상태를 RL 입력 벡터로 변환
        
        Returns:
            8차원 상태 벡터 (클로가 PPO/DQN에 사용)
        """
        player = game_state.get('player', {})
        obstacles = game_state.get('obstacles', [])
        
        # 플레이어 정보
        player_x = player.get('x', 320) / 640  # 정규화
        player_y = player.get('y', 240) / 480
        player_vy = player.get('vy', 0) / 20  # 속도 정규화
        
        # 가장 가까운 장애물 정보
        if obstacles:
            nearest_obs = min(obstacles, 
                            key=lambda o: abs(o.get('y', 0) - player.get('y', 0)))
            obs_x = nearest_obs.get('x', 320) / 640
            obs_y = nearest_obs.get('y', 0) / 480
            distance = np.sqrt((obs_x - player_x)**2 + (obs_y - player_y)**2)
        else:
            obs_x = 0.5
            obs_y = 0.0
            distance = 1.0
        
        return [
            player_x,
            player_y,
            player_vy,
            1.0 if player.get('y', 0) >= 440 else 0.0,  # on_ground
            obs_x,
            obs_y,
            distance,
            max(0, (obs_y - player_y) / 0.1)  # time_to_collision 추정
        ]
    
    def find_action_for_frame(self, frame_time: int, actions: List[Dict]) -> str:
        """
        프레임 시간에 가장 가까운 액션 찾기
        """
        if not actions:
            return 'stay'
        
        closest_action = min(actions, 
                           key=lambda a: abs(a.get('timestamp', 0) - frame_time))
        return closest_action.get('action', 'stay')
    
    def calculate_reward(self, current_state: Dict, next_state: Dict) -> float:
        """
        보상 계산 (클로의 RL 훈련용)
        """
        reward = 0.0
        
        # 생존 보상
        if not next_state.get('game_over', False):
            reward += 1.0
        else:
            reward -= 100.0  # 게임 오버 페널티
        
        # 점수 증가 보상
        score_diff = next_state.get('score', 0) - current_state.get('score', 0)
        reward += score_diff * 10.0
        
        return reward
    
    def update_stats(self, session_data: Dict):
        """통계 업데이트"""
        self.stats['total_sessions'] += 1
        self.stats['total_frames'] += len(session_data.get('frames', []))
        self.stats['total_actions'] += len(session_data.get('actions', []))
        
        mode = session_data.get('mode', 'unknown')
        if mode == 'human':
            self.stats['human_sessions'] += 1
        elif mode == 'ai':
            self.stats['ai_sessions'] += 1
        
        self.save_stats()
    
    def load_stats(self):
        """통계 로드"""
        stats_file = self.data_dir / "stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
    
    def save_stats(self):
        """통계 저장"""
        stats_file = self.data_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def get_stats(self) -> Dict:
        """통계 반환"""
        return self.stats.copy()
    
    def export_for_yolo(self, output_dir: str):
        """
        제이의 YOLOv8 훈련을 위한 데이터셋 export
        
        YOLO 표준 형식:
        - images/: 이미지 파일들
        - labels/: 어노테이션 .txt 파일들
        - data.yaml: 데이터셋 설정
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        (output_path / "images").mkdir(exist_ok=True)
        (output_path / "labels").mkdir(exist_ok=True)
        
        # data.yaml 생성
        yaml_content = """
# YOLOv8 Dataset Configuration
# Generated from web gameplay data

path: {}
train: images
val: images  # TODO: Split train/val

names:
  0: player
  1: obstacle

nc: 2
""".format(str(output_path.absolute()))
        
        with open(output_path / "data.yaml", 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ YOLO 데이터셋 export 완료: {output_path}")
    
    def export_for_rl(self, output_dir: str):
        """
        클로의 PPO/DQN 훈련을 위한 데이터셋 export
        
        형식:
        - replay_buffer.json: 모든 transitions
        - config.json: 환경 설정
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 모든 RL 데이터 합치기
        all_transitions = []
        for rl_file in self.rl_training_dir.glob("*.json"):
            with open(rl_file, 'r') as f:
                data = json.load(f)
                all_transitions.extend(data.get('transitions', []))
        
        replay_buffer = {
            'transitions': all_transitions,
            'size': len(all_transitions),
            'state_dim': 8,
            'action_space': ['stay', 'jump', 'left', 'right']
        }
        
        with open(output_path / "replay_buffer.json", 'w') as f:
            json.dump(replay_buffer, f, indent=2)
        
        print(f"✅ RL 데이터셋 export 완료: {output_path}")
        print(f"   총 transitions: {len(all_transitions)}")


# 사용 예시
if __name__ == "__main__":
    collector = TrainingDataCollector()
    
    # 통계 출력
    stats = collector.get_stats()
    print("📊 수집된 데이터 통계:")
    print(f"   총 세션: {stats['total_sessions']}")
    print(f"   총 프레임: {stats['total_frames']}")
    print(f"   총 액션: {stats['total_actions']}")
    print(f"   Human 세션: {stats['human_sessions']}")
    print(f"   AI 세션: {stats['ai_sessions']}")


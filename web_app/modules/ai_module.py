"""
AI Module - Reinforcement Learning Policy

Chloe Lee (cl4490) 담당 모듈
PPO/DQN 기반 게임 AI 정책

TODO for Chloe:
1. simulate_ai_decision() → real_ppo_decision() 교체
2. 정책 네트워크 훈련 및 로드
3. 실시간 의사결정 최적화
4. 자가 학습 (Self-Play) 구현
"""
from src.models.policy_network import PolicyNetwork, ValueNetwork

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import random

# PyTorch는 선택적 (실제 RL 모델 구현 시 필요)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch (torch) 없음 - 시뮬레이션 모드만 사용 가능")
    # 더미 클래스 (타입 힌트용)
    class nn:
        class Module:
            pass
        class Sequential:
            pass
        class Linear:
            pass
        class ReLU:
            pass
        class Softmax:
            pass

# TODO: Chloe가 추가할 import
# from stable_baselines3 import PPO, DQN
# from ..src.utils.rl_instrumentation import RLInstrumentationLogger


class PolicyNetwork(nn.Module):
    """
    정책 네트워크 (MLP)
    
    Chloe가 구현할 신경망 구조
    """
    
    def __init__(self, state_dim: int = 8, hidden_dim: int = 128, action_dim: int = 4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch (torch)가 필요합니다. 실제 RL 모델 구현 시 사용됩니다.")
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)


class ValueNetwork(nn.Module):
    """
    가치 네트워크 (PPO용)
    
    Chloe가 PPO 구현 시 사용
    """
    
    def __init__(self, state_dim: int = 8, hidden_dim: int = 128):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch (torch)가 필요합니다. 실제 RL 모델 구현 시 사용됩니다.")
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)


class AIDecisionResult:
    """AI 의사결정 결과"""
    
    def __init__(self, action: str, confidence: float, reasoning: str = "", 
                 action_probs: Optional[Dict[str, float]] = None):
        self.action = action
        self.confidence = confidence
        self.reasoning = reasoning
        self.action_probs = action_probs or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (웹 전송용)"""
        return {
            'action': self.action,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'action_probs': self.action_probs,
            'timestamp': self.timestamp
        }


class AIModule:
    """
    AI 모듈 - 강화학습 기반 게임 AI
    
    Chloe가 구현할 주요 기능:
    1. PPO/DQN 정책 로드 및 추론
    2. 실시간 의사결정
    3. 자가 학습 데이터 수집
    4. 성능 모니터링
    """
    
    def __init__(self, model_path: Optional[str] = None, algorithm: str = "PPO"):
        """
        초기화
        
        Args:
            model_path: 훈련된 모델 경로
            algorithm: 사용할 알고리즘 ("PPO" 또는 "DQN")
        """
        self.model_path = model_path
        self.algorithm = algorithm
        # PyTorch가 없으면 device는 None (시뮬레이션 모드)
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        
        # 모델들
        self.policy_net = None
        self.value_net = None
        self.ppo_model = None
        self.dqn_model = None
        
        # 성능 추적
        self.decision_times = []
        self.action_history = []
        self.reward_history = []
        
        # RL 계측 (Chloe가 구현)
        self.rl_logger = None
        
        # 초기화
        self._initialize_model()
    
    def _initialize_model(self):




        self.policy_net = PolicyNetwork().to(self.device)
        self.value_net = ValueNetwork().to(self.device)

        """
        모델 초기화
        
        TODO for Chloe: 실제 PPO/DQN 모델 로드 구현
        """
        if self.model_path:
            # TODO: 실제 구현
            # if self.algorithm == "PPO":
            #     self.ppo_model = PPO.load(self.model_path)
            # elif self.algorithm == "DQN":
            #     self.dqn_model = DQN.load(self.model_path)
            
            print(f"🤖 [Chloe TODO] {self.algorithm} 모델 로드: {self.model_path}")
        else:
            # 기본 정책 네트워크 (시뮬레이션용) - PyTorch가 있을 때만
            if TORCH_AVAILABLE:
                self.policy_net = PolicyNetwork().to(self.device)
            print("⚠️ 모델 경로가 없습니다. 시뮬레이션 모드로 실행합니다.")
        
        # RL 계측 시스템 초기화
        # TODO: self.rl_logger = RLInstrumentationLogger("web_game_ai")
    
    def make_decision(self, game_state: Dict[str, Any]) -> AIDecisionResult:
        """
        게임 상태를 보고 행동 결정
        
        Args:
            game_state: 게임 엔진에서 받은 상태 정보
            
        Returns:
            AI 의사결정 결과
            
        TODO for Chloe: 실제 PPO/DQN 추론 구현
        """
        start_time = time.perf_counter()
        
        if self.ppo_model or self.dqn_model:
            # 실제 RL 모델 추론
            result = self._real_rl_decision(game_state)
        else:
            # 시뮬레이션 모드
            result = self._simulate_decision(game_state)
        
        # 성능 측정
        decision_time = time.perf_counter() - start_time
        self.decision_times.append(decision_time)
        self.action_history.append(result.action)
        
        return result
    
    def _simulate_decision(self, game_state: Dict[str, Any]) -> AIDecisionResult:
        """
        시뮬레이션된 AI 의사결정 (현재 구현)
        
        Chloe가 _real_rl_decision()으로 교체할 예정
        """
        # 간단한 휴리스틱 기반 의사결정
        player_y = game_state.get('player_y', 0.5)
        obstacle_y = game_state.get('obstacle_y', 0.0)
        obstacle_distance = game_state.get('obstacle_distance', 1.0)
        time_to_collision = game_state.get('time_to_collision', 10.0)
        
        # 의사결정 로직
        if time_to_collision < 1.0 and obstacle_distance < 0.3:
            if player_y > 0.7:  # 플레이어가 아래쪽에 있으면
                action = "jump"
                reasoning = "장애물이 가까워서 점프"
                confidence = 0.8
            else:
                action = "stay"
                reasoning = "이미 위쪽에 있어서 대기"
                confidence = 0.6
        else:
            # 랜덤 행동 (탐험)
            actions = ["stay", "jump", "left", "right"]
            weights = [0.4, 0.3, 0.15, 0.15]
            action = np.random.choice(actions, p=weights)
            reasoning = f"탐험적 행동: {action}"
            confidence = 0.5
        
        # 행동 확률 분포 (시뮬레이션)
        action_probs = {
            "stay": 0.4,
            "jump": 0.3,
            "left": 0.15,
            "right": 0.15
        }
        action_probs[action] += 0.2  # 선택된 행동의 확률 증가
        
        return AIDecisionResult(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            action_probs=action_probs
        )
    
    def _real_rl_decision(self, game_state: Dict[str, Any]) -> AIDecisionResult:
        """
        실제 강화학습 모델 의사결정
        
        TODO for Chloe: 이 함수를 구현하세요!
        
        구현 가이드:
        1. 게임 상태를 RL 모델 입력 형식으로 변환
        2. PPO 또는 DQN 추론 실행
        3. 행동 확률 분포 계산
        4. 최적 행동 선택
        5. 의사결정 근거 생성
        """
        try:
            # 상태 벡터 생성
            state_vector = self._create_state_vector(game_state)
            
            if self.algorithm == "PPO" and self.ppo_model:
                # TODO: PPO 추론
                # action, _states = self.ppo_model.predict(state_vector, deterministic=False)
                # action_probs = self._get_action_probabilities(state_vector)
                
                # 임시: 시뮬레이션 호출
                return self._simulate_decision(game_state)
                
            elif self.algorithm == "DQN" and self.dqn_model:
                # TODO: DQN 추론
                # action, _states = self.dqn_model.predict(state_vector, deterministic=False)
                # q_values = self._get_q_values(state_vector)
                
                # 임시: 시뮬레이션 호출
                return self._simulate_decision(game_state)
            
        except Exception as e:
            print(f"❌ RL 모델 추론 오류: {e}")
            # 오류 시 시뮬레이션으로 폴백
            return self._simulate_decision(game_state)
    
    def _create_state_vector(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        게임 상태를 RL 모델 입력 벡터로 변환
        
        TODO for Chloe: 상태 표현 최적화
        """
        # 8차원 상태 벡터 생성
        state_vector = np.array([
            game_state.get('player_x', 0.5),
            game_state.get('player_y', 0.5),
            game_state.get('player_vy', 0.0),
            game_state.get('on_ground', 0.0),
            game_state.get('obstacle_x', 0.0),
            game_state.get('obstacle_y', 0.0),
            game_state.get('obstacle_distance', 1.0),
            game_state.get('time_to_collision', 10.0)
        ], dtype=np.float32)
        
        return state_vector
    
    def update_reward(self, reward: float, done: bool = False):
        """
        보상 업데이트 (자가 학습용)
        
        TODO for Chloe: 온라인 학습 구현
        """
        self.reward_history.append(reward)
        
        if self.rl_logger:
            # TODO: RL 계측 시스템에 기록
            # self.rl_logger.log_step(reward, done)
            pass
        
        # 에피소드 종료 시 학습 (선택적)
        if done and len(self.reward_history) > 100:
            self._update_policy()
    
    def _update_policy(self):
        """
        정책 업데이트 (온라인 학습)
        
        TODO for Chloe: PPO/DQN 온라인 학습 구현
        """
        # TODO: 실제 정책 업데이트 구현
        # 1. 경험 버퍼에서 배치 샘플링
        # 2. 정책 그래디언트 계산
        # 3. 모델 파라미터 업데이트
        # 4. 성능 로깅
        
        print("🔄 [Chloe TODO] 정책 업데이트 실행")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        if not self.decision_times:
            return {}
        
        avg_decision_time = np.mean(self.decision_times)
        avg_reward = np.mean(self.reward_history) if self.reward_history else 0
        
        # 행동 분포 계산
        action_counts = {}
        for action in self.action_history:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'avg_decision_time_ms': avg_decision_time * 1000,
            'avg_reward': avg_reward,
            'total_decisions': len(self.action_history),
            'action_distribution': action_counts,
            'recent_actions': self.action_history[-10:],  # 최근 10개 행동
            'algorithm': self.algorithm
        }
    
    def reset_episode(self):
        """에피소드 초기화"""
        if self.rl_logger:
            # TODO: 에피소드 종료 로깅
            # self.rl_logger.log_episode_end(...)
            pass
        
        # 히스토리 초기화 (선택적)
        if len(self.action_history) > 1000:  # 메모리 관리
            self.action_history = self.action_history[-500:]
            self.reward_history = self.reward_history[-500:]
    
    def save_model(self, save_path: str):
        """
        모델 저장
        
        TODO for Chloe: 훈련된 모델 저장 구현
        """
        if self.ppo_model:
            self.ppo_model.save(save_path)
        elif self.dqn_model:
            self.dqn_model.save(save_path)
        else:
            # PyTorch 모델 저장
            if TORCH_AVAILABLE and self.policy_net:
                torch.save(self.policy_net.state_dict(), save_path)
        
        print(f"💾 모델 저장 완료: {save_path}")


# Chloe가 사용할 헬퍼 함수들
def create_reward_function(game_state: Dict[str, Any], action: str, next_state: Dict[str, Any]) -> float:
    """
    보상 함수 설계
    
    TODO for Chloe: 게임에 맞는 보상 함수 구현
    """
    reward = 0.0
    
    # 생존 보상
    if not next_state.get('game_over', False):
        reward += 1.0
    
    # 충돌 페널티
    if next_state.get('game_over', False):
        reward -= 100.0
    
    # 점수 증가 보상
    score_diff = next_state.get('score', 0) - game_state.get('score', 0)
    reward += score_diff * 10.0
    
    # 불필요한 행동 페널티 (선택적)
    if action in ["left", "right"] and game_state.get('obstacle_distance', 1.0) > 0.5:
        reward -= 0.1
    
    return reward


def analyze_failure_mode(game_state: Dict[str, Any], action: str) -> str:
    """
    실패 모드 분석
    
    Chloe가 디버깅용으로 사용할 수 있는 함수
    """
    if game_state.get('game_over', False):
        obstacle_distance = game_state.get('obstacle_distance', 1.0)
        time_to_collision = game_state.get('time_to_collision', 10.0)
        
        if obstacle_distance < 0.2 and action == "stay":
            return "회피 실패: 장애물이 가까운데 행동하지 않음"
        elif time_to_collision < 0.5 and action in ["left", "right"]:
            return "잘못된 회피: 점프 대신 좌우 이동"
        else:
            return "일반적인 충돌"
    
    return "정상"


# 사용 예시 (Chloe가 참고할 코드)
if __name__ == "__main__":
    # AI 모듈 초기화
    ai_module = AIModule(
        model_path="path/to/ppo_model.zip",  # Chloe가 훈련한 모델
        algorithm="PPO"
    )
    
    # 테스트 게임 상태
    test_state = {
        'player_x': 0.5,
        'player_y': 0.8,
        'player_vy': 0.0,
        'on_ground': 1.0,
        'obstacle_x': 0.6,
        'obstacle_y': 0.3,
        'obstacle_distance': 0.4,
        'time_to_collision': 2.0
    }
    
    # AI 의사결정
    decision = ai_module.make_decision(test_state)
    
    # 결과 출력
    print(f"선택된 행동: {decision.action}")
    print(f"신뢰도: {decision.confidence:.2f}")
    print(f"근거: {decision.reasoning}")
    
    # 성능 통계
    stats = ai_module.get_performance_stats()
    print(f"평균 의사결정 시간: {stats.get('avg_decision_time_ms', 0):.1f}ms")

"""
Web Session Module - WebSocket Session Management

Minsuk Kim (mk4434) 담당 모듈
Flask-SocketIO 기반 실시간 게임 세션 관리

담당 기능:
1. WebSocket 세션 관리
2. 게임 상태 동기화
3. 성능 모니터링
4. 팀원 모듈 통합
"""

import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import queue

from .game_engine import GameState, GameActions
from .cv_module import ComputerVisionModule
from .ai_module import AIModule


@dataclass
class SessionStats:
    """세션 통계"""
    session_id: str
    start_time: float
    total_frames: int = 0
    total_actions: int = 0
    mode: str = "human"  # "human" or "ai"
    avg_fps: float = 0.0
    peak_fps: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WebGameSession:
    """
    웹 게임 세션 클래스
    
    각 클라이언트마다 하나씩 생성되어 독립적인 게임 상태 관리
    """
    
    def __init__(self, session_id: str, socketio_instance=None):
        """
        세션 초기화
        
        Args:
            session_id: 고유 세션 ID
            socketio_instance: Flask-SocketIO 인스턴스
        """
        self.session_id = session_id
        self.socketio = socketio_instance
        
        # 게임 상태
        self.game_state = GameState()
        self.mode = "human"  # "human" or "ai"
        self.is_active = True
        
        # 팀원 모듈들
        self.cv_module = ComputerVisionModule()  # Jeewon 모듈
        self.ai_module = AIModule()              # Chloe 모듈
        
        # 성능 추적
        self.stats = SessionStats(session_id=session_id, start_time=time.time())
        self.frame_times = []
        self.last_frame_time = time.time()
        
        # 스레드 안전성
        self.lock = threading.Lock()
        self.action_queue = queue.Queue()
        
        print(f"🎮 새 게임 세션 생성: {session_id}")
    
    def set_mode(self, mode: str):
        """게임 모드 설정 (human/ai)"""
        if mode in ["human", "ai"]:
            with self.lock:
                self.mode = mode
                self.stats.mode = mode
            
            self.emit_to_client('mode_changed', {'mode': mode})
            print(f"🔄 세션 {self.session_id} 모드 변경: {mode}")
    
    def handle_user_action(self, action: str):
        """사용자 액션 처리 (human 모드)"""
        if not self.is_active or self.mode != "human":
            return
        
        if GameActions.is_valid_action(action):
            self.action_queue.put(action)
            self.stats.total_actions += 1
    
    def update_game_loop(self):
        """
        게임 루프 업데이트 (메인 로직)
        
        1. 액션 처리 (Human/AI)
        2. 게임 상태 업데이트
        3. CV 모듈 호출 (Jeewon)
        4. 클라이언트 동기화
        """
        if not self.is_active:
            return
        
        current_time = time.time()
        
        with self.lock:
            # 1. 액션 결정 및 처리
            if self.mode == "human":
                action = self._get_human_action()
            else:  # ai 모드
                action = self._get_ai_action()
            
            # 게임 상태에 액션 적용
            if action:
                self.game_state.handle_action(action)
            
            # 2. 게임 물리/로직 업데이트
            self.game_state.update()
            
            # 3. CV 모듈 호출 (Jeewon 부분)
            # TODO: 실제 프레임 데이터로 교체
            self._process_computer_vision()
            
            # 4. AI 모듈 보상 업데이트 (Chloe 부분)
            if self.mode == "ai":
                self._update_ai_reward()
        
        # 5. 클라이언트에 상태 전송
        self._emit_game_state()
        
        # 6. 성능 통계 업데이트
        self._update_performance_stats(current_time)
    
    def _get_human_action(self) -> Optional[str]:
        """Human 모드 액션 가져오기"""
        try:
            return self.action_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _get_ai_action(self) -> Optional[str]:
        """AI 모드 액션 가져오기 (Chloe 모듈 사용)"""
        try:
            # 게임 상태를 AI 입력 형식으로 변환
            ai_state = self.game_state.get_state_for_ai()
            
            # Chloe의 AI 모듈 호출
            decision = self.ai_module.make_decision(ai_state)
            
            # AI 의사결정 정보를 클라이언트에 전송
            self.emit_to_client('ai_decision', decision.to_dict())
            
            return decision.action
            
        except Exception as e:
            print(f"❌ AI 의사결정 오류: {e}")
            return "stay"  # 안전한 기본 액션
    
    def _process_computer_vision(self):
        """
        컴퓨터 비전 처리 (Jeewon 모듈 사용)
        
        TODO: 실제 게임 프레임을 CV 모듈에 전달
        """
        try:
            # TODO: 실제 게임 프레임 캡처
            # 현재는 더미 프레임 사용
            import numpy as np
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Jeewon의 CV 모듈 호출
            detections = self.cv_module.detect_objects(dummy_frame)
            
            # 탐지 결과를 클라이언트에 전송 (디버깅용)
            detection_data = [det.to_dict() for det in detections]
            self.emit_to_client('cv_detections', detection_data)
            
        except Exception as e:
            print(f"❌ CV 처리 오류: {e}")
    
    def _update_ai_reward(self):
        """AI 보상 업데이트 (Chloe 모듈 사용)"""
        try:
            # 보상 계산 (생존 시간 기반)
            if self.game_state.game_over:
                reward = -100.0  # 게임 오버 페널티
                self.ai_module.update_reward(reward, done=True)
                self.ai_module.reset_episode()
            else:
                reward = 1.0  # 생존 보상
                self.ai_module.update_reward(reward, done=False)
                
        except Exception as e:
            print(f"❌ AI 보상 업데이트 오류: {e}")
    
    def _emit_game_state(self):
        """클라이언트에 게임 상태 전송"""
        if not self.socketio:
            return
        
        try:
            # 게임 상태 데이터
            game_data = self.game_state.get_state_for_web()
            
            # 성능 정보 추가
            game_data['performance'] = {
                'fps': self.stats.avg_fps,
                'mode': self.mode,
                'total_frames': self.stats.total_frames
            }
            
            # 클라이언트에 전송
            self.emit_to_client('game_update', game_data)
            
        except Exception as e:
            print(f"❌ 게임 상태 전송 오류: {e}")
    
    def _update_performance_stats(self, current_time: float):
        """성능 통계 업데이트"""
        # FPS 계산
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        
        # 최근 30프레임 평균 FPS
        if len(self.frame_times) > 30:
            self.frame_times = self.frame_times[-30:]
        
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.stats.avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.stats.peak_fps = max(self.stats.peak_fps, self.stats.avg_fps)
        
        self.stats.total_frames += 1
        self.last_frame_time = current_time
    
    def emit_to_client(self, event: str, data: Any):
        """클라이언트에 이벤트 전송"""
        if self.socketio:
            self.socketio.emit(event, data, room=self.session_id)
    
    def reset_game(self):
        """게임 리셋"""
        with self.lock:
            self.game_state.reset()
            
            # AI 모듈 리셋
            if self.mode == "ai":
                self.ai_module.reset_episode()
            
            # 액션 큐 비우기
            while not self.action_queue.empty():
                try:
                    self.action_queue.get_nowait()
                except queue.Empty:
                    break
        
        self.emit_to_client('game_reset', {})
        print(f"🔄 세션 {self.session_id} 게임 리셋")
    
    def get_session_info(self) -> Dict[str, Any]:
        """세션 정보 반환"""
        with self.lock:
            # 팀원 모듈 성능 정보
            cv_stats = self.cv_module.get_performance_stats()
            ai_stats = self.ai_module.get_performance_stats()
            
            return {
                'session': self.stats.to_dict(),
                'game_state': self.game_state.get_state_for_web(),
                'cv_performance': cv_stats,
                'ai_performance': ai_stats,
                'is_active': self.is_active
            }
    
    def close_session(self):
        """세션 종료"""
        self.is_active = False
        
        # 리소스 정리
        if hasattr(self.cv_module, 'cleanup'):
            self.cv_module.cleanup()
        
        if hasattr(self.ai_module, 'cleanup'):
            self.ai_module.cleanup()
        
        print(f"🔚 세션 {self.session_id} 종료")


class SessionManager:
    """
    세션 매니저 - 모든 웹 게임 세션 관리
    
    Minsuk이 구현한 중앙 관리 시스템
    """
    
    def __init__(self, socketio_instance=None):
        self.socketio = socketio_instance
        self.sessions: Dict[str, WebGameSession] = {}
        self.lock = threading.Lock()
    
    def create_session(self, client_id: str) -> WebGameSession:
        """새 세션 생성"""
        session_id = f"session_{client_id}_{int(time.time())}"
        
        with self.lock:
            session = WebGameSession(session_id, self.socketio)
            self.sessions[session_id] = session
        
        return session
    
    def get_session(self, session_id: str) -> Optional[WebGameSession]:
        """세션 가져오기"""
        return self.sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """세션 제거"""
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.close_session()
                del self.sessions[session_id]
    
    def get_all_sessions_info(self) -> Dict[str, Any]:
        """모든 세션 정보 반환"""
        with self.lock:
            return {
                'total_sessions': len(self.sessions),
                'active_sessions': sum(1 for s in self.sessions.values() if s.is_active),
                'sessions': {sid: session.get_session_info() 
                           for sid, session in self.sessions.items()}
            }
    
    def cleanup_inactive_sessions(self):
        """비활성 세션 정리"""
        current_time = time.time()
        to_remove = []
        
        with self.lock:
            for session_id, session in self.sessions.items():
                # 30분 이상 비활성 세션 제거
                if (current_time - session.stats.start_time) > 1800:
                    to_remove.append(session_id)
        
        for session_id in to_remove:
            self.remove_session(session_id)
        
        if to_remove:
            print(f"🧹 비활성 세션 {len(to_remove)}개 정리 완료")


# 사용 예시 (app.py에서 사용할 코드)
"""
# Flask-SocketIO 앱에서 사용 방법:

from modules.web_session import SessionManager

# 세션 매니저 초기화
session_manager = SessionManager(socketio)

@socketio.on('connect')
def handle_connect():
    session = session_manager.create_session(request.sid)
    join_room(session.session_id)

@socketio.on('user_action')
def handle_user_action(data):
    session = session_manager.get_session(data['session_id'])
    if session:
        session.handle_user_action(data['action'])

# 게임 루프 (별도 스레드)
def game_loop():
    while True:
        for session in session_manager.sessions.values():
            session.update_game_loop()
        time.sleep(1/30)  # 30 FPS
"""

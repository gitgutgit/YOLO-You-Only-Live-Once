"""
Modular Flask App - Team Collaboration Version

모듈화된 웹 애플리케이션
각 팀원이 독립적으로 작업할 수 있도록 구조화

팀원별 담당:
- Minsuk (mk4434): 웹 서버, 세션 관리, 통합
- Jeewon (jk4864): CV 모듈 (modules/cv_module.py)
- Chloe (cl4490): AI 모듈 (modules/ai_module.py)
"""

import os
import time
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room

# 팀원 모듈들 import
from modules.web_session import SessionManager
from modules.game_engine import GameActions


# Flask 앱 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'distilled-vision-agent-secret-key'

# SocketIO 설정 (실시간 통신)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25
)

# 세션 매니저 초기화 (Minsuk 담당)
session_manager = SessionManager(socketio)

# 게임 루프 스레드 제어
game_loop_running = False
game_loop_thread = None


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/health')
def health_check():
    """헬스 체크 (GCP Cloud Run용)"""
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'total_sessions': len(session_manager.sessions),
        'active_sessions': sum(1 for s in session_manager.sessions.values() if s.is_active)
    }


@app.route('/admin')
def admin_dashboard():
    """관리자 대시보드 (개발/디버깅용)"""
    return {
        'sessions': session_manager.get_all_sessions_info(),
        'server_info': {
            'game_loop_running': game_loop_running,
            'total_threads': threading.active_count()
        }
    }


# =============================================================================
# WebSocket 이벤트 핸들러들 (실시간 통신)
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """클라이언트 연결"""
    print(f"🔗 클라이언트 연결: {request.sid}")
    
    # 새 게임 세션 생성
    session = session_manager.create_session(request.sid)
    join_room(session.session_id)
    
    # 클라이언트에 세션 정보 전송
    emit('session_created', {
        'session_id': session.session_id,
        'mode': session.mode
    })
    
    # 게임 루프 시작 (첫 번째 클라이언트 연결 시)
    start_game_loop()


@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제"""
    print(f"🔌 클라이언트 연결 해제: {request.sid}")
    
    # 해당 세션 찾아서 제거
    for session_id, session in list(session_manager.sessions.items()):
        if request.sid in session_id:
            session_manager.remove_session(session_id)
            break


@socketio.on('set_mode')
def handle_set_mode(data):
    """게임 모드 변경 (Human/AI)"""
    session_id = data.get('session_id')
    mode = data.get('mode', 'human')
    
    session = session_manager.get_session(session_id)
    if session:
        session.set_mode(mode)
        print(f"🎮 모드 변경: {session_id} → {mode}")


@socketio.on('user_action')
def handle_user_action(data):
    """사용자 액션 처리 (키보드 입력)"""
    session_id = data.get('session_id')
    action = data.get('action', 'stay')
    
    session = session_manager.get_session(session_id)
    if session and session.mode == 'human':
        session.handle_user_action(action)


@socketio.on('reset_game')
def handle_reset_game(data):
    """게임 리셋"""
    session_id = data.get('session_id')
    
    session = session_manager.get_session(session_id)
    if session:
        session.reset_game()


@socketio.on('get_session_info')
def handle_get_session_info(data):
    """세션 정보 요청"""
    session_id = data.get('session_id')
    
    session = session_manager.get_session(session_id)
    if session:
        info = session.get_session_info()
        emit('session_info', info)


# =============================================================================
# 게임 루프 (별도 스레드에서 실행)
# =============================================================================

def game_loop():
    """
    메인 게임 루프
    
    모든 활성 세션의 게임 상태를 30 FPS로 업데이트
    """
    global game_loop_running
    
    print("🎮 게임 루프 시작 (30 FPS)")
    
    target_fps = 30
    frame_time = 1.0 / target_fps
    
    while game_loop_running:
        start_time = time.time()
        
        # 모든 활성 세션 업데이트
        active_sessions = [s for s in session_manager.sessions.values() if s.is_active]
        
        for session in active_sessions:
            try:
                session.update_game_loop()
            except Exception as e:
                print(f"❌ 세션 {session.session_id} 업데이트 오류: {e}")
        
        # 비활성 세션 정리 (1분마다)
        if int(time.time()) % 60 == 0:
            session_manager.cleanup_inactive_sessions()
        
        # FPS 제어
        elapsed = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed)
        time.sleep(sleep_time)
    
    print("🛑 게임 루프 종료")


def start_game_loop():
    """게임 루프 시작"""
    global game_loop_running, game_loop_thread
    
    if not game_loop_running:
        game_loop_running = True
        game_loop_thread = threading.Thread(target=game_loop, daemon=True)
        game_loop_thread.start()


def stop_game_loop():
    """게임 루프 중지"""
    global game_loop_running
    game_loop_running = False


# =============================================================================
# 팀원별 개발 가이드
# =============================================================================

"""
🎯 팀원별 작업 가이드:

📋 Minsuk (mk4434) - 웹 서버 & 통합:
✅ 완료: Flask 앱, SocketIO, 세션 관리
🔄 진행중: 팀원 모듈 통합, GCP 배포
📝 할일: 성능 최적화, 모니터링

👁️ Jeewon (jk4864) - 컴퓨터 비전:
📁 작업 파일: modules/cv_module.py
🎯 목표: YOLOv8 실시간 객체 탐지 (60 FPS)
📝 할일:
  1. _real_yolo_detection() 함수 구현
  2. ONNX 최적화 적용
  3. 웹 환경에서 프레임 처리

🤖 Chloe (cl4490) - AI 정책:
📁 작업 파일: modules/ai_module.py
🎯 목표: PPO/DQN 실시간 의사결정
📝 할일:
  1. _real_rl_decision() 함수 구현
  2. 정책 네트워크 훈련 및 로드
  3. 온라인 학습 (Self-Play) 구현

🔗 통합 포인트:
- CV 모듈: WebGameSession._process_computer_vision()
- AI 모듈: WebGameSession._get_ai_action()
- 공통 게임 엔진: modules/game_engine.py (수정 금지)

🚀 실행 방법:
1. 로컬 테스트: python3 app_modular.py
2. GCP 배포: 기존 cloudbuild.yaml 사용
3. 팀 Git: 각자 브랜치에서 모듈별 작업

📊 성능 목표:
- 전체 시스템: 30 FPS (웹 게임)
- CV 모듈: ≤16.7ms/frame (60 FPS 가능)
- AI 모듈: ≤5ms/decision
"""


if __name__ == '__main__':
    print("🚀 Distilled Vision Agent - 모듈화 버전 시작")
    print("👥 팀원별 모듈:")
    print("   - Minsuk: 웹 서버 & 통합")
    print("   - Jeewon: CV 모듈 (modules/cv_module.py)")
    print("   - Chloe: AI 모듈 (modules/ai_module.py)")
    print()
    
    # 개발 모드에서 실행
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        use_reloader=False  # 게임 루프 스레드 충돌 방지
    )

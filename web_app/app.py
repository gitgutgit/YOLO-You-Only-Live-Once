#!/usr/bin/env python3
"""
Web-based Interactive Vision Game

Flask 웹 애플리케이션으로 브라우저에서 플레이 가능한 게임
GCP Cloud Run에 배포 가능

Author: Minsuk Kim (mk4434)
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import time
import random
import threading
import uuid
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'distilled-vision-agent-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Game configuration
GAME_CONFIG = {
    'width': 640,
    'height': 480,
    'fps': 30,
    'player_size': 40,
    'obstacle_size': 40,
    'player_speed': 8,
    'jump_strength': -15,
    'gravity': 0.8,
    'obstacle_speed': 7
}

# Global game sessions
game_sessions = {}

class WebGameSession:
    """웹 게임 세션 관리 클래스"""
    
    def __init__(self, session_id):
        self.session_id = session_id
        self.reset()
        self.mode = "human"  # "human" or "ai"
        self.last_update = time.time()
        self.ai_thread = None
        self.running = False
        
    def reset(self):
        """게임 상태 초기화"""
        self.player_x = GAME_CONFIG['width'] // 2
        self.player_y = GAME_CONFIG['height'] - 80
        self.player_vy = 0
        self.obstacles = []
        self.score = 0
        self.game_over = False
        self.start_time = time.time()
        self.frame_count = 0
        
    def get_survival_time(self):
        """생존 시간 계산"""
        return time.time() - self.start_time
    
    def update_physics(self):
        """물리 엔진 업데이트"""
        if self.game_over:
            return
            
        # 중력 적용
        self.player_vy += GAME_CONFIG['gravity']
        self.player_y += self.player_vy
        
        # 바닥 충돌
        if self.player_y >= GAME_CONFIG['height'] - GAME_CONFIG['player_size']:
            self.player_y = GAME_CONFIG['height'] - GAME_CONFIG['player_size']
            self.player_vy = 0
        
        # 플레이어 경계 제한
        self.player_x = max(0, min(GAME_CONFIG['width'] - GAME_CONFIG['player_size'], self.player_x))
    
    def update_obstacles(self):
        """장애물 업데이트"""
        if self.game_over:
            return
            
        # 장애물 이동
        for obstacle in self.obstacles:
            obstacle['y'] += GAME_CONFIG['obstacle_speed']
        
        # 화면 밖 장애물 제거 및 점수 업데이트
        initial_count = len(self.obstacles)
        self.obstacles = [obs for obs in self.obstacles if obs['y'] < GAME_CONFIG['height']]
        self.score += initial_count - len(self.obstacles)
        
        # 새 장애물 생성
        if random.random() < 0.02:  # 2% 확률
            new_obstacle = {
                'x': random.randint(0, GAME_CONFIG['width'] - GAME_CONFIG['obstacle_size']),
                'y': -GAME_CONFIG['obstacle_size'],
                'id': str(uuid.uuid4())
            }
            self.obstacles.append(new_obstacle)
    
    def check_collisions(self):
        """충돌 검사"""
        if self.game_over:
            return
            
        player_rect = {
            'x': self.player_x,
            'y': self.player_y,
            'width': GAME_CONFIG['player_size'],
            'height': GAME_CONFIG['player_size']
        }
        
        for obstacle in self.obstacles:
            obstacle_rect = {
                'x': obstacle['x'],
                'y': obstacle['y'],
                'width': GAME_CONFIG['obstacle_size'],
                'height': GAME_CONFIG['obstacle_size']
            }
            
            if self.rects_collide(player_rect, obstacle_rect):
                self.game_over = True
                break
    
    def rects_collide(self, rect1, rect2):
        """사각형 충돌 검사"""
        return (rect1['x'] < rect2['x'] + rect2['width'] and
                rect1['x'] + rect1['width'] > rect2['x'] and
                rect1['y'] < rect2['y'] + rect2['height'] and
                rect1['y'] + rect1['height'] > rect2['y'])
    
    def handle_action(self, action):
        """플레이어 액션 처리"""
        if self.game_over:
            return
            
        if action == "jump" and self.player_y >= GAME_CONFIG['height'] - GAME_CONFIG['player_size'] - 5:
            self.player_vy = GAME_CONFIG['jump_strength']
        elif action == "left":
            self.player_x -= GAME_CONFIG['player_speed']
        elif action == "right":
            self.player_x += GAME_CONFIG['player_speed']
    
    def ai_decision(self):
        """AI 결정 로직 (시뮬레이션)"""
        if not self.obstacles:
            return "stay"
        
        # 가장 가까운 장애물 찾기
        visible_obstacles = [obs for obs in self.obstacles if obs['y'] > 0]
        if not visible_obstacles:
            return "stay"
            
        nearest = min(visible_obstacles, key=lambda o: o['y'])
        
        # 정규화된 좌표
        player_x_norm = self.player_x / GAME_CONFIG['width']
        obstacle_x_norm = nearest['x'] / GAME_CONFIG['width']
        obstacle_y_norm = nearest['y'] / GAME_CONFIG['height']
        
        # 간단한 휴리스틱
        if obstacle_y_norm > 0.7:  # 장애물이 화면 하단에 있을 때
            dx = obstacle_x_norm - player_x_norm
            
            if abs(dx) < 0.15:  # 장애물이 가까이 있을 때
                if dx < 0:
                    return "right"
                else:
                    return "left"
            elif obstacle_y_norm > 0.85:  # 매우 가까울 때 점프
                return "jump"
        
        return "stay"
    
    def get_state(self):
        """현재 게임 상태 반환"""
        return {
            'player': {
                'x': self.player_x,
                'y': self.player_y,
                'vy': self.player_vy
            },
            'obstacles': self.obstacles,
            'score': self.score,
            'survival_time': self.get_survival_time(),
            'game_over': self.game_over,
            'mode': self.mode,
            'frame_count': self.frame_count
        }

# AI 게임 루프 (별도 스레드)
def ai_game_loop(session_id):
    """AI 모드 게임 루프"""
    session = game_sessions.get(session_id)
    if not session:
        return
        
    session.running = True
    
    while session.running and session.mode == "ai" and not session.game_over:
        try:
            # AI 결정
            action = session.ai_decision()
            
            # 액션 적용
            session.handle_action(action)
            
            # 게임 상태 업데이트
            session.update_physics()
            session.update_obstacles()
            session.check_collisions()
            session.frame_count += 1
            
            # 클라이언트에 상태 전송
            socketio.emit('game_update', {
                'state': session.get_state(),
                'ai_action': action
            }, room=session_id)
            
            # FPS 제한
            time.sleep(1.0 / GAME_CONFIG['fps'])
            
        except Exception as e:
            print(f"AI 게임 루프 오류: {e}")
            break
    
    session.running = False

# Flask 라우트
@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html', config=GAME_CONFIG)

@app.route('/api/config')
def get_config():
    """게임 설정 API"""
    return jsonify(GAME_CONFIG)

@app.route('/api/leaderboard')
def get_leaderboard():
    """리더보드 API (추후 구현)"""
    # 실제로는 데이터베이스에서 가져올 예정
    mock_leaderboard = [
        {'name': 'AI Agent', 'score': 150, 'time': 45.2, 'mode': 'ai'},
        {'name': 'Human Player', 'score': 120, 'time': 38.7, 'mode': 'human'},
        {'name': 'Test User', 'score': 95, 'time': 32.1, 'mode': 'human'}
    ]
    return jsonify(mock_leaderboard)

# SocketIO 이벤트
@socketio.on('connect')
def handle_connect():
    """클라이언트 연결"""
    session_id = request.sid
    game_sessions[session_id] = WebGameSession(session_id)
    
    emit('connected', {
        'session_id': session_id,
        'config': GAME_CONFIG
    })
    
    print(f"클라이언트 연결: {session_id}")

@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제"""
    session_id = request.sid
    
    if session_id in game_sessions:
        session = game_sessions[session_id]
        session.running = False
        del game_sessions[session_id]
    
    print(f"클라이언트 연결 해제: {session_id}")

@socketio.on('start_game')
def handle_start_game(data):
    """게임 시작"""
    session_id = request.sid
    session = game_sessions.get(session_id)
    
    if session:
        session.reset()
        session.mode = data.get('mode', 'human')
        
        emit('game_started', {
            'state': session.get_state()
        })
        
        # AI 모드인 경우 AI 스레드 시작
        if session.mode == "ai":
            if session.ai_thread and session.ai_thread.is_alive():
                session.running = False
                session.ai_thread.join()
            
            session.ai_thread = threading.Thread(target=ai_game_loop, args=(session_id,))
            session.ai_thread.daemon = True
            session.ai_thread.start()
        
        print(f"게임 시작: {session_id}, 모드: {session.mode}")

@socketio.on('player_action')
def handle_player_action(data):
    """플레이어 액션 처리"""
    session_id = request.sid
    session = game_sessions.get(session_id)
    
    if session and session.mode == "human":
        action = data.get('action')
        session.handle_action(action)
        
        # 게임 상태 업데이트 (Human 모드에서만)
        session.update_physics()
        session.update_obstacles()
        session.check_collisions()
        session.frame_count += 1
        
        emit('game_update', {
            'state': session.get_state()
        })

@socketio.on('switch_mode')
def handle_switch_mode(data):
    """모드 전환"""
    session_id = request.sid
    session = game_sessions.get(session_id)
    
    if session:
        new_mode = data.get('mode')
        session.mode = new_mode
        session.running = False  # AI 스레드 중지
        
        if new_mode == "ai":
            # AI 모드 시작
            session.ai_thread = threading.Thread(target=ai_game_loop, args=(session_id,))
            session.ai_thread.daemon = True
            session.ai_thread.start()
        
        emit('mode_switched', {
            'mode': new_mode,
            'state': session.get_state()
        })
        
        print(f"모드 전환: {session_id} -> {new_mode}")

@socketio.on('get_state')
def handle_get_state():
    """현재 게임 상태 요청"""
    session_id = request.sid
    session = game_sessions.get(session_id)
    
    if session:
        emit('game_update', {
            'state': session.get_state()
        })

# 헬스체크 (GCP Cloud Run용)
@app.route('/health')
def health_check():
    """헬스체크 엔드포인트"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(game_sessions)
    })

if __name__ == '__main__':
    print("🌐 Distilled Vision Agent - Web Game Server")
    print("=" * 50)
    print("🎮 브라우저에서 접속하여 게임을 플레이하세요!")
    print("📱 Human Mode: 직접 플레이")
    print("🤖 AI Mode: AI 플레이 관찰")
    print("☁️ GCP Cloud Run 배포 준비 완료")
    print()
    
    # 개발 모드에서는 debug=True, 프로덕션에서는 False
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

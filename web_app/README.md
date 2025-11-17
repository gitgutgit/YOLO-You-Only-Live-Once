# 🌐 Distilled Vision Agent - Web Application

**팀원별 모듈화된 실시간 웹 게임 AI**

- **🎯 제이**: Computer Vision 모듈 (`modules/cv_module.py`)
- **🎯 클로**: AI Policy 모듈 (`modules/ai_module.py`)
- **🎯 래리**: Web Integration (`app_modular.py`, `modules/web_session.py`)

GCP Cloud Run에 배포 가능한 Flask + SocketIO 웹 애플리케이션

## 🎮 기능

### **Human Mode** 🧑

- 브라우저에서 직접 게임 플레이
- 키보드 컨트롤 (SPACE: 점프, A/D: 이동)
- 실시간 점수 및 생존 시간 표시

### **AI Mode** 🤖

- AI 에이전트 자동 플레이 관찰
- 실시간 AI 결정 과정 표시
- 컴퓨터 비전 + 정책 네트워크 시뮬레이션

### **실시간 모니터링** 📊

- FPS 및 성능 통계
- 리더보드 시스템
- WebSocket 기반 실시간 통신

## 🚀 로컬 실행

### 1. 의존성 설치

```bash
cd web_app
pip install -r requirements.txt
```

### 2. 개발 서버 실행

```bash
python app.py
```

### 3. 브라우저 접속

```
http://localhost:8080
```

## ☁️ GCP Cloud Run 배포

### 사전 준비

1. GCP 프로젝트 생성
2. Google Cloud SDK 설치
3. Docker 설치

### 자동 배포

```bash
# 프로젝트 ID를 입력하여 배포
./deploy.sh your-gcp-project-id

# 또는 수동으로 각 단계 실행
gcloud config set project your-gcp-project-id
gcloud services enable cloudbuild.googleapis.com run.googleapis.com
docker build -t gcr.io/your-gcp-project-id/distilled-vision-agent .
docker push gcr.io/your-gcp-project-id/distilled-vision-agent
gcloud run deploy distilled-vision-agent --image gcr.io/your-gcp-project-id/distilled-vision-agent --platform managed --allow-unauthenticated
```

### Cloud Build 자동 배포 (권장)

```bash
# GitHub 연동 후 자동 배포
gcloud builds submit --config cloudbuild.yaml
```

## 🏗️ 아키텍처

```
Frontend (HTML5 Canvas + JavaScript)
    ↕ WebSocket (SocketIO)
Flask Backend (Python)
    ├── Game Session Management
    ├── AI Decision Logic (Simulated)
    ├── Real-time State Updates
    └── Performance Monitoring
```

## 📁 프로젝트 구조

```
web_app/
├── app.py                 # Flask 메인 애플리케이션
├── templates/
│   └── index.html        # 게임 웹 페이지
├── static/
│   ├── css/style.css     # 스타일시트
│   └── js/game.js        # 게임 클라이언트 로직
├── requirements.txt      # Python 의존성
├── Dockerfile           # 컨테이너 이미지 빌드
├── cloudbuild.yaml      # GCP Cloud Build 설정
├── deploy.sh           # 배포 스크립트
└── README.md           # 이 파일
```

## 🎯 게임 컨트롤

### Human Mode

- **SPACE**: 점프/플랩
- **A** / **←**: 왼쪽 이동
- **D** / **→**: 오른쪽 이동

### 공통 컨트롤

- **H**: Human 모드 전환
- **I**: AI 모드 전환
- **R**: 게임 재시작

## 🔧 기술 스택

### Backend

- **Flask**: 웹 프레임워크
- **Flask-SocketIO**: 실시간 WebSocket 통신
- **Gunicorn + Eventlet**: 프로덕션 WSGI 서버

### Frontend

- **HTML5 Canvas**: 게임 렌더링
- **Socket.IO Client**: 실시간 통신
- **Vanilla JavaScript**: 게임 로직
- **CSS3**: 반응형 UI 디자인

### Infrastructure

- **GCP Cloud Run**: 서버리스 컨테이너 배포
- **GCP Container Registry**: 도커 이미지 저장
- **GCP Cloud Build**: CI/CD 파이프라인

## 📊 성능 최적화

- **실시간 FPS 모니터링**: 60 FPS 목표
- **WebSocket 최적화**: 최소 레이턴시 통신
- **Canvas 렌더링 최적화**: RequestAnimationFrame 사용
- **서버 리소스 관리**: 세션별 독립적 게임 상태

## 🎯 팀원별 상세 작업 가이드

### 👁️ **제이 - Computer Vision 모듈**

**📁 담당 파일:** `modules/cv_module.py`

**🎯 목표:** YOLOv8 기반 실시간 객체 탐지 (60 FPS)

**📝 구현할 핵심 함수:**

#### 1. `_real_yolo_detection()` ⭐ 가장 중요!

```python
def _real_yolo_detection(self, frame: np.ndarray) -> List[CVDetectionResult]:
    """
    실제 YOLOv8 객체 탐지 구현

    현재: _simulate_detection() 호출 (가짜)
    TODO: 실제 YOLOv8 모델로 교체
    """
```

#### 2. `_initialize_model()` - 모델 로드

```python
def _initialize_model(self):
    """TODO: self.model = YOLO(self.model_path)"""
```

**🧪 테스트:** `cd modules && python3 cv_module.py`

**📊 성능 목표:** ≤16.7ms/frame, mAP ≥ 0.7

---

### 🤖 **클로 - AI Policy 모듈**

**📁 담당 파일:** `modules/ai_module.py`

**🎯 목표:** PPO/DQN 기반 실시간 의사결정

**📝 구현할 핵심 함수:**

#### 1. `_real_rl_decision()` ⭐ 가장 중요!

```python
def _real_rl_decision(self, game_state: Dict[str, Any]) -> AIDecisionResult:
    """
    실제 강화학습 모델 의사결정

    현재: _simulate_decision() 호출 (간단한 휴리스틱)
    TODO: 실제 PPO/DQN 모델로 교체
    """
```

#### 2. `_initialize_model()` - 모델 로드

```python
def _initialize_model(self):
    """TODO: self.ppo_model = PPO.load(self.model_path)"""
```

#### 3. `_update_policy()` - 온라인 학습

```python
def _update_policy(self):
    """TODO: Self-Play 구현"""
```

**🧪 테스트:** `cd modules && python3 ai_module.py`

**📊 성능 목표:** ≤5ms/decision, 평균 120초 생존

---

### 🔗 **래리 - Web Integration 모듈**

**📁 담당 파일:** `app_modular.py`, `modules/web_session.py`

**✅ 완료:** 모듈화 구조, 웹 서버, 세션 관리, GCP 배포

**🔄 진행중:** 팀원 모듈 통합 테스트, 성능 최적화

---

## 🚀 **통합 작업 흐름**

### **1단계: 각자 브랜치에서 작업**

```bash
git checkout -b jeewon-cv-implementation    # 제이
git checkout -b chloe-ai-implementation     # 클로
git checkout -b larry-integration          # 래리
```

### **2단계: 모듈별 단독 테스트**

```bash
# 제이: CV 모듈
python3 modules/cv_module.py

# 클로: AI 모듈
python3 modules/ai_module.py

# 래리: 통합 테스트
python3 app_modular.py
```

### **3단계: 통합 테스트**

```bash
# 모든 모듈 완성 후
python3 app_modular.py
# http://localhost:5000 접속
# Human/AI 모드 전환 테스트
```

## 🔮 향후 통합 계획

현재는 시뮬레이션된 AI이지만, 팀원들과 통합 시:

1. **제이의 YOLOv8**: 실제 객체 탐지로 교체
2. **클로의 PPO/DQN**: 실제 강화학습 훈련 루프 통합
3. **실시간 학습**: 브라우저에서 AI 훈련 과정 관찰
4. **데이터 수집**: Human 플레이 데이터로 Policy Distillation

## 🌐 배포 URL 예시

배포 완료 후 다음과 같은 URL에서 접속 가능:

```
https://distilled-vision-agent-xxxxx-uc.a.run.app
```

## 🎉 팀 정보

**Team Backward** - COMS W4995 Deep Learning for Computer Vision

- **Jeewon Kim (jk4864)**: YOLOv8 & System Architecture
- **Chloe Lee (cl4490)**: PPO/DQN & Reinforcement Learning
- **Minsuk Kim (mk4434)**: Web Development & Deployment

---

**🚀 브라우저에서 바로 플레이하고 AI와 경쟁해보세요!**

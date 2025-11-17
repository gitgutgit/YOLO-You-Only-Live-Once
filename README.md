# Distilled Vision Agent: YOLO, You Only Live Once

**Team: Prof.Peter.backward()**

- **🎯 제이** - Computer Vision & YOLOv8 Implementation
- **🎯 클로** - AI Policy & Reinforcement Learning
- **🎯 래리** - Web Integration & Deployment

## Project Overview

Real-time vision-based game AI that learns to play a 2D survival game through:

1. **Policy Distillation**: Learning from expert demonstrations
2. **Self-Play RL**: Improving through PPO/DQN reinforcement learning

### Key Features

- 🎯 **Real-time Performance**: Target 60 FPS (≤16.7ms/frame)
- 👁️ **Vision-Only**: No privileged game state access
- 🧠 **Interpretable**: Structured state vectors for debugging
- 🚀 **End-to-End**: RGB frames → YOLO detection → MLP policy → Actions

## 📁 Project Structure

```
final_project/                          # 🏠 메인 프로젝트 디렉토리
├── 📋 README.md                        # 메인 프로젝트 가이드 (이 파일)
├── 📋 TEAM_CHECKLIST.md               # 팀원별 간단 체크리스트
├── 📋 TEAM_INTEGRATION.md             # 통합 가이드 문서
├── 📄 requirements.txt                 # 전체 프로젝트 의존성
├── 📄 teamBackward_deep_learning_computer_vision (3).pdf  # 프로젝트 제안서
│
├── 🎮 Game/                           # 기존 게임 프로토타입 (참고용)
│   ├── game_agent.py                  # 로컬 Pygame 버전
│   ├── interactive_game.py            # 인터랙티브 게임
│   └── requirements.txt               # 게임 의존성
│
├── 📦 src/                           # 소스 코드 라이브러리 (래리 개발)
│   ├── processing/                   # 데이터 처리 모듈
│   │   ├── __init__.py
│   │   └── augmentation.py           # 데이터 증강 (완성됨)
│   ├── utils/                        # 유틸리티 모듈
│   │   ├── __init__.py
│   │   ├── visualization.py          # 시각화 도구 (완성됨)
│   │   └── rl_instrumentation.py     # RL 계측 (완성됨)
│   ├── deployment/                   # 배포 최적화 모듈
│   │   ├── __init__.py
│   │   └── onnx_optimizer.py         # ONNX 최적화 (완성됨)
│   ├── models/                       # 모델 아키텍처 (비어있음)
│   └── training/                     # 훈련 파이프라인 (비어있음)
│
├── 🧪 tests/                        # 테스트 스크립트들
│   ├── test_pipeline.py              # 통합 테스트
│   ├── test_simple.py                # 간단한 테스트
│   └── test_core_logic.py            # 핵심 로직 테스트
│
├── 📊 data/                         # 데이터 저장소 (실제 데이터 파일들)
│   ├── raw/                         # 원본 게임플레이 녹화
│   ├── labeled/                     # 라벨링된 프레임
│   └── processed/                   # 처리된 훈련 데이터
│
├── ⚙️ configs/                      # 훈련 설정 파일들
├── 📚 docs/                         # 문서 및 보고서
│
└── 🌐 web_app/                      # ⭐ 메인 웹 애플리케이션 (팀 작업 공간)
    ├── 📋 README.md                  # 웹 앱 상세 가이드
    ├── 📋 TEAM_GUIDE.md             # 팀원별 작업 가이드
    ├── 📄 requirements.txt           # 웹 앱 의존성
    │
    ├── 🚀 app.py                    # 기존 버전 (참고용)
    ├── 🌟 app_modular.py            # ⭐ 새로운 모듈화된 메인 앱
    │
    ├── 📦 modules/                  # ⭐ 팀원별 작업 모듈
    │   ├── __init__.py              # 패키지 초기화
    │   ├── 🎮 game_engine.py        # 공통 게임 로직 (수정 금지)
    │   ├── 👁️ cv_module.py          # ⭐ 제이 담당 (YOLOv8)
    │   ├── 🤖 ai_module.py          # ⭐ 클로 담당 (PPO/DQN)
    │   └── 🔗 web_session.py        # ⭐ 래리 담당 (세션 관리)
    │
    ├── 🎨 templates/
    │   └── index.html               # 웹 UI 템플릿
    │
    ├── 📱 static/
    │   ├── css/
    │   │   └── style.css            # 스타일시트
    │   └── js/
    │       └── game.js              # 클라이언트 게임 로직
    │
    └── ☁️ 배포 관련 파일들
        ├── Dockerfile               # 컨테이너 이미지
        ├── .gcloudignore           # GCP 빌드 제외 파일
        ├── cloudbuild.yaml          # GCP 빌드 설정
        ├── cloudbuild_deploy.sh     # 빌드 배포 스크립트
        ├── deploy.sh                # 배포 스크립트
        └── quick_deploy.sh          # 간단 배포 스크립트
```

### 🎯 **핵심 작업 디렉토리**

**팀원들이 주로 작업할 곳:**

- **`web_app/modules/cv_module.py`** ← 👁️ **제이 작업 공간**
- **`web_app/modules/ai_module.py`** ← 🤖 **클로 작업 공간**
- **`web_app/app_modular.py`** ← 🔗 **래리 통합 공간**

### 📂 **디렉토리 역할 구분**

```
📦 src/          # 재사용 가능한 라이브러리 코드
📊 data/         # 실제 데이터 파일들 (이미지, 비디오 등)
🌐 web_app/      # 웹 애플리케이션 (메인 작업 공간)
🧪 tests/        # 테스트 코드들
```

**왜 이렇게 나눴나요?**

- `src/`: 다른 프로젝트에서도 재사용할 수 있는 유틸리티들
- `data/`: 실제 데이터 파일들 (용량이 크고 Git에 올리지 않음)
- `web_app/`: 실제 제품 (팀원들이 협업하는 메인 공간)

## Quick Start

### Option 1: Full Installation (Recommended)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run full test suite
python scripts/test_pipeline.py

# Run current prototype
cd Game
python game_agent.py
```

### Option 2: Core Logic Testing (No Dependencies)

```bash
# Test basic functionality without external packages
python3 scripts/simple_test.py

# Test core algorithms
python3 scripts/test_core_logic.py
```

### Verified Working Components ✅

- **Data Augmentation**: Algorithmic logic tested and working
- **Visualization Tools**: Core rendering and profiling logic verified
- **Performance Profiling**: Timing and FPS calculation systems operational
- **RL Instrumentation**: Episode logging and analysis systems functional
- **ONNX Optimization**: Model export and inference pipeline logic validated

## 🎯 팀원별 상세 작업 분담

### 👁️ **제이 - Computer Vision 모듈**

**📁 담당 파일:** `web_app/modules/cv_module.py`

**🎯 목표:** YOLOv8 기반 실시간 객체 탐지 (60 FPS)

**📝 구현해야 할 함수들:**

#### 1. `_initialize_model()` - 모델 로드

```python
def _initialize_model(self):
    # TODO: YOLOv8 모델 로드
    # self.model = YOLO(self.model_path)
    # ONNX 최적화 적용
```

#### 2. `_real_yolo_detection()` - 핵심 함수 ⭐

```python
def _real_yolo_detection(self, frame: np.ndarray) -> List[CVDetectionResult]:
    # TODO: 실제 YOLOv8 추론 구현
    # 1. 프레임 전처리 (640x640)
    # 2. YOLOv8 추론 실행
    # 3. 후처리 (NMS, 신뢰도 필터링)
    # 4. CVDetectionResult 객체로 변환
```

#### 3. `_preprocess_frame()` - 전처리

```python
def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
    # TODO: YOLOv8 입력 형식으로 변환
    # 리사이즈, 정규화, HWC→CHW 변환
```

#### 4. `_postprocess_outputs()` - 후처리

```python
def _postprocess_outputs(self, outputs: np.ndarray) -> List[CVDetectionResult]:
    # TODO: NMS, 신뢰도 필터링, 좌표 변환
```

**📊 성능 목표:**

- 추론 속도: ≤16.7ms/frame (60 FPS 가능)
- 탐지 정확도: mAP ≥ 0.7
- 메모리 사용량: ≤512MB

**🧪 테스트 방법:**

```bash
cd web_app/modules
python3 cv_module.py  # 단독 테스트
```

---

### 🤖 **클로 - AI Policy 모듈**

**📁 담당 파일:** `web_app/modules/ai_module.py`

**🎯 목표:** PPO/DQN 기반 실시간 의사결정 및 자가 학습

**📝 구현해야 할 함수들:**

#### 1. `_initialize_model()` - 모델 로드

```python
def _initialize_model(self):
    # TODO: PPO/DQN 모델 로드
    # self.ppo_model = PPO.load(self.model_path)
    # RL 계측 시스템 초기화
```

#### 2. `_real_rl_decision()` - 핵심 함수 ⭐

```python
def _real_rl_decision(self, game_state: Dict[str, Any]) -> AIDecisionResult:
    # TODO: 실제 강화학습 모델 의사결정
    # 1. 게임 상태 → RL 입력 변환
    # 2. PPO/DQN 추론 실행
    # 3. 행동 확률 분포 계산
    # 4. 최적 행동 선택 및 근거 생성
```

#### 3. `_create_state_vector()` - 상태 변환

```python
def _create_state_vector(self, game_state: Dict[str, Any]) -> np.ndarray:
    # TODO: 게임 상태를 RL 모델 입력으로 변환
    # 8차원 상태 벡터 생성
```

#### 4. `_update_policy()` - 온라인 학습

```python
def _update_policy(self):
    # TODO: PPO/DQN 온라인 학습 구현
    # 1. 경험 버퍼에서 배치 샘플링
    # 2. 정책 그래디언트 계산
    # 3. 모델 파라미터 업데이트
```

#### 5. `save_model()` - 모델 저장

```python
def save_model(self, save_path: str):
    # TODO: 훈련된 모델 저장
```

**📊 성능 목표:**

- 의사결정 속도: ≤5ms/decision
- 생존 시간: 평균 120초 이상
- 학습 안정성: 1000 에피소드 수렴

**🧪 테스트 방법:**

```bash
cd web_app/modules
python3 ai_module.py  # 단독 테스트
```

---

### 🔗 **래리 - Web Integration 모듈**

**📁 담당 파일:**

- `web_app/app_modular.py` (메인 서버)
- `web_app/modules/web_session.py` (세션 관리)

**🎯 목표:** 팀원 모듈 통합 및 웹 배포

**✅ 완료된 작업:**

- Flask-SocketIO 웹 서버 구축
- 실시간 게임 세션 관리 시스템
- 팀원 모듈 통합 인터페이스
- GCP Cloud Run 배포 파이프라인
- 모듈화된 구조 설계

**🔄 진행중인 작업:**

#### 1. 팀원 모듈 통합 테스트

```python
# 제이의 CV 모듈 통합 확인
def _process_computer_vision(self):
    detections = self.cv_module.detect_objects(frame)

# 클로의 AI 모듈 통합 확인
def _get_ai_action(self):
    decision = self.ai_module.make_decision(ai_state)
```

#### 2. 성능 모니터링 시스템

```python
def _update_performance_stats(self, current_time: float):
    # FPS 계산 및 성능 추적
```

#### 3. GCP 배포 최적화

- Cloud Run 자동 스케일링 설정
- 성능 모니터링 대시보드
- CI/CD 파이프라인 개선

**📊 성능 목표:**

- 웹 게임: 30 FPS 안정적 동작
- 실시간 응답: ≤100ms 지연시간
- 동시 접속: 50명 이상 지원

---

## 🚀 **통합 작업 흐름**

### **1단계: 개별 모듈 구현**

```bash
# 각자 브랜치에서 작업
git checkout -b jeewon-cv-implementation    # 제이
git checkout -b chloe-ai-implementation     # 클로
git checkout -b larry-integration          # 래리
```

### **2단계: 단독 테스트**

```bash
# 제이: CV 모듈 테스트
python3 web_app/modules/cv_module.py

# 클로: AI 모듈 테스트
python3 web_app/modules/ai_module.py

# 래리: 통합 테스트
python3 web_app/app_modular.py
```

### **3단계: 통합 테스트**

```bash
# 모든 모듈이 완성되면
python3 web_app/app_modular.py
# 브라우저에서 http://localhost:5000 접속
# Human/AI 모드 전환하며 테스트
```

### **4단계: 최종 배포**

```bash
# GCP Cloud Run 배포
cd web_app
gcloud builds submit --config cloudbuild.yaml
```

## 📋 **작업 체크리스트**

### 👁️ **제이 (CV 모듈)**

- [ ] YOLOv8 모델 훈련 완료
- [ ] `_real_yolo_detection()` 함수 구현
- [ ] ONNX 최적화 적용
- [ ] 60 FPS 성능 달성 확인
- [ ] 단독 테스트 통과
- [ ] 통합 테스트 참여

### 🤖 **클로 (AI 모듈)**

- [ ] PPO/DQN 모델 훈련 완료
- [ ] `_real_rl_decision()` 함수 구현
- [ ] 자가 학습 시스템 구축
- [ ] 생존 시간 목표 달성
- [ ] 단독 테스트 통과
- [ ] 통합 테스트 참여

### 🔗 **래리 (통합)**

- [x] 모듈화된 구조 설계 완료
- [x] 웹 서버 및 세션 관리 완료
- [x] GCP 배포 파이프라인 완료
- [ ] 팀원 모듈 통합 테스트
- [ ] 성능 최적화 및 모니터링
- [ ] 최종 배포 및 문서화

## Success Criteria

- **Detection Quality**: ≥70% mAP on game objects
- **Imitation Accuracy**: ≥75% action agreement with expert
- **Performance Gain**: ≥20% survival time improvement via RL
- **Real-time Constraint**: ≥60 FPS end-to-end inference

## License

Academic project for COMS W4995 - Deep Learning for Computer Vision, Columbia University.

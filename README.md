# 🎮 Distilled Vision Agent: YOLO, You Only Live Once

**Team: Prof.Peter.backward()** | **COMS W4995 - Deep Learning for Computer Vision**

## 🌐 Live Demo

**웹 게임 플랫폼**: https://distilled-vision-agent-fhuhwhnu3a-uc.a.run.app

- **Human Mode**: 직접 플레이하며 전문가 시연 데이터 수집
- **AI Mode**: AI 에이전트의 실시간 플레이 관찰
- **Leaderboard**: 전 세계 플레이어 순위

## 📝 Project Overview

**목표**: Vision-based Deep Learning Agent가 2D 게임을 순수 시각 정보만으로 학습하고 플레이

**핵심 파이프라인**:
```
RGB 프레임 → YOLO 탐지 → MLP 정책 네트워크 → 액션 결정
```

### Key Features

- 🎯 **Real-time Performance**: 60 FPS 목표 (≤16.7ms/frame)
- 👁️ **Vision-Only Input**: 게임 내부 상태 접근 없이 순수 RGB 이미지만 사용
- 🧠 **Dual Learning**: Policy Distillation + Self-Play RL
- 🚀 **End-to-End Pipeline**: 데이터 수집 → 훈련 → 배포
- ☁️ **Cloud Deployment**: GCP Cloud Run 실시간 서비스

## 👥 Team Responsibilities

### ✅ **Minsuk Kim (mk4434)** - 게임 개발 & 배포 (완료)

**담당 영역**: 웹 플랫폼, 데이터 파이프라인, GCP 배포

**완료된 작업**:
- ✅ 웹 게임 플랫폼 (Flask + SocketIO)
- ✅ GCP Cloud Run 배포
- ✅ 데이터 수집 파이프라인 (State-Action-Reward, Bounding Boxes)
- ✅ YOLO 데이터셋 자동 Export (`yolo_exporter.py`)
- ✅ 데이터 증강 시스템 (`src/data/augmentation.py`)
- ✅ ONNX 최적화 도구 (`src/deployment/onnx_optimizer.py`)

**담당 파일**:
- `web_app/app.py` - 메인 Flask 서버
- `web_app/yolo_exporter.py` - YOLO 데이터셋 Export
- `src/data/augmentation.py` - 데이터 증강
- `src/deployment/onnx_optimizer.py` - 모델 최적화

---

### 🔴 **Jeewon Kim (jk4864)** - YOLO 객체 탐지 (진행 중)

**담당 영역**: 컴퓨터 비전, YOLOv8 모델 훈련 및 분석

**현재 상태**:
- ✅ YOLOv8-nano 모델 훈련 완료 (`YOLO_demo/YOLO-dataset-11221748/best.pt`)
- ✅ YOLO 데이터셋 생성 완료 (`web_app/game_dataset/` - 483 train, 81 val)
- 🚧 웹 앱 통합 및 실시간 추론 구현 필요
- 🚧 모델 성능 분석 및 최적화 필요

**해야 할 일**:

1. **웹 앱 통합** (우선순위: 🔴 Critical)
   - [ ] `web_app/modules/cv_module.py`에 실제 YOLO 추론 구현
   - [ ] `_real_yolo_detection()` 함수 완성
   - [ ] 실시간 성능 테스트 (60 FPS 목표, ≤16.7ms/frame)
   - [ ] ONNX 변환 및 최적화

2. **모델 성능 분석** (우선순위: 🟡 High)
   - [ ] 클래스별 성능 분석 (mAP, Precision, Recall)
   - [ ] 오류 분석 (False Positive/Negative)
   - [ ] IoU 분포 분석
   - [ ] 실패 케이스 분석

3. **모델 비교 실험** (우선순위: 🟡 High)
   - [ ] YOLO 버전 비교 (nano, small, medium)
   - [ ] 해상도 비교 실험 (320, 416, 640, 832)
   - [ ] 하이퍼파라미터 튜닝
   - [ ] 속도/정확도 트레이드오프 분석

4. **모델 해석성 분석** (우선순위: 🟢 Medium)
   - [ ] Grad-CAM 시각화
   - [ ] Attention map 생성
   - [ ] 모델이 어디를 보고 있는지 분석

**작업 폴더**:
- `YOLO_demo/YOLO-dataset-11221748/` - YOLO 훈련 및 테스트
- `web_app/game_dataset/` - YOLO 데이터셋 (이미지 + 라벨)
- `web_app/modules/cv_module.py` - 웹 통합 모듈

**참고 문서**:
- `.agent_context/jeewon_analysis_research_tasks.md` - 상세 분석 작업 가이드

---

### 🟣 **Chloe Lee (cl4490)** - 강화학습 (시작 필요)

**담당 영역**: PPO/DQN 기반 RL 에이전트 훈련

**현재 상태**:
- ✅ 데이터 수집 완료 (`web_app/collected_gameplay/` - 23+ 세션)
- ✅ RL 데이터 형식 준비 완료 (`states_actions.jsonl`)
- ❌ RL 모델 훈련 미시작

**해야 할 일**:

1. **데이터 로더 구현** (우선순위: 🔴 Critical)
   - [ ] `src/training/data_loader.py` - RL 데이터 로더 구현
   - [ ] `states_actions.jsonl` 읽기 및 파싱
   - [ ] Replay Buffer 구현
   - [ ] 데이터 전처리 파이프라인

2. **Policy Distillation (Imitation Learning)** (우선순위: 🔴 Critical)
   - [ ] 전문가 시연 데이터 로드 (Human Mode 데이터)
   - [ ] MLP 정책 네트워크 아키텍처 설계
   - [ ] Supervised Learning으로 초기 정책 훈련
   - [ ] 목표: ≥75% action agreement

3. **PPO/DQN 훈련** (우선순위: 🔴 Critical)
   - [ ] `src/training/ppo_trainer.py` - PPO/DQN 훈련 구현
   - [ ] `src/models/policy_network.py` - 정책 네트워크 아키텍처
   - [ ] State-based 정책 먼저 구현
   - [ ] Self-Play 환경 구축
   - [ ] 목표: ≥20% 생존 시간 향상

4. **웹 앱 통합** (우선순위: 🟡 High)
   - [ ] `web_app/modules/ai_module.py`에 실제 RL 추론 구현
   - [ ] `_real_rl_decision()` 함수 완성
   - [ ] 실시간 의사결정 테스트 (≤5ms/decision)

5. **Vision-based RL** (선택, 우선순위: 🟢 Medium)
   - [ ] YOLO 출력 → RL 입력 변환
   - [ ] End-to-End Vision-based 정책

**작업 폴더**:
- `web_app/collected_gameplay/session_*/states_actions.jsonl` - RL 훈련 데이터
- `src/training/ppo_trainer.py` - PPO/DQN 훈련 스크립트
- `src/models/policy_network.py` - 정책 네트워크
- `web_app/modules/ai_module.py` - 웹 통합 모듈

**참고 문서**:
- `Legacy/Larry/RL_TRAINING_GUIDE.md` - RL 훈련 상세 가이드
- `web_app/TEAM_GUIDE.md` - 모듈 통합 가이드

---

## 📁 Project Structure

```
final_project/
├── 📱 web_app/                      # 웹 게임 플랫폼
│   ├── app.py                       # Flask 서버 (메인)
│   ├── modules/                     # 팀원별 모듈
│   │   ├── cv_module.py            # 👁️ Jeewon - YOLO 통합
│   │   ├── ai_module.py            # 🤖 Chloe - PPO/DQN 통합
│   │   └── game_engine.py          # 공통 게임 로직
│   ├── game_dataset/                # YOLO 데이터셋 (483 train, 81 val)
│   │   ├── images/train/           # 훈련 이미지
│   │   ├── labels/train/           # 훈련 라벨
│   │   └── data.yaml               # YOLO 설정
│   ├── collected_gameplay/          # 수집된 게임 데이터 (23+ 세션)
│   │   └── session_*/
│   │       ├── states_actions.jsonl # RL 훈련 데이터
│   │       └── bboxes.jsonl        # YOLO 라벨 데이터
│   └── yolo_exporter.py             # YOLO 데이터셋 자동 Export
│
├── 🔬 src/                          # 소스 코드 모듈
│   ├── models/                      # 모델 아키텍처
│   │   ├── policy_network.py       # 🚧 Chloe 작업 필요
│   ├── training/                    # 훈련 파이프라인
│   │   ├── ppo_trainer.py          # 🚧 Chloe 작업 필요
│   │   └── data_loader.py          # 🚧 Chloe 작업 필요
│   ├── data/                        # 데이터 파이프라인
│   │   └── augmentation.py         # 데이터 증강 (완성)
│   └── deployment/                  # 배포 최적화
│       └── onnx_optimizer.py       # ONNX 최적화 (완성)
│
├── 🎯 YOLO_demo/                    # Jeewon 작업 폴더
│   └── YOLO-dataset-11221748/      # YOLO 훈련 및 테스트
│       ├── best.pt                  # 훈련된 YOLO 모델
│       └── yolo_test.py            # YOLO 테스트 스크립트
│
└── 📚 docs/                          # 문서
    ├── AI_MODE_EXPLANATION.md      # AI 모드 동작 원리
    └── web_app/
        ├── DATA_COLLECTION_GUIDE.md # 데이터 수집 가이드
        └── TEAM_GUIDE.md           # 팀원별 작업 가이드
```

## 🚀 Quick Start

### 로컬 실행

```bash
cd web_app
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
# Access at http://localhost:5001
```

### YOLO 훈련 (Jeewon)

```bash
cd YOLO_demo/YOLO-dataset-11221748
yolo detect train model=yolov8n.pt data=./data.yaml epochs=50 imgsz=640
```

### RL 훈련 (Chloe)

```bash
# 데이터 로더 구현 후
python src/training/ppo_trainer.py
```

## 🎯 Success Criteria

| 기준                      | 목표                  | 담당자 | 현재 상태      | 중요도      |
| ------------------------- | --------------------- | ------ | -------------- | ----------- |
| **Detection Quality**     | mAP ≥ 70%             | Jeewon | ✅ 모델 훈련 완료 | 🔴 Critical |
| **Imitation Accuracy**    | ≥75% action agreement | Chloe  | ❌ 미시작      | 🔴 Critical |
| **Performance Gain**      | ≥20% survival time ↑  | Chloe  | ❌ 미시작      | 🔴 Critical |
| **Real-time Performance** | ≥60 FPS inference     | All    | ⚠️ 30 FPS (웹) | 🟡 High     |
| **Data Collection**       | ≥5,000 frames         | Minsuk | ✅ 500+ frames | ✅ 완료     |

## 🔗 Important Links

- **Live Demo**: https://distilled-vision-agent-fhuhwhnu3a-uc.a.run.app
- **Team GitHub**: https://github.com/gitgutgit/YOLO-You-Only-Live-Once
- **Minsuk GitHub**: https://github.com/Snowtype/distilled-vision-agent

## 📚 Additional Documentation

- `AI_MODE_EXPLANATION.md` - AI 모드 동작 원리
- `web_app/DATA_COLLECTION_GUIDE.md` - 데이터 수집 가이드
- `web_app/TEAM_GUIDE.md` - 팀원별 모듈 작업 가이드
- `.agent_context/jeewon_analysis_research_tasks.md` - Jeewon 분석 작업 가이드
- `Legacy/Larry/RL_TRAINING_GUIDE.md` - Chloe RL 훈련 가이드

---

**Academic project for COMS W4995 - Deep Learning for Computer Vision, Columbia University**  
**Team: Prof.Peter.backward() | Fall 2025**

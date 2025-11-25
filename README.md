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

**완료된 작업**: 웹 게임 플랫폼, GCP 배포, 데이터 수집 파이프라인, YOLO 데이터셋 Export

---

### 🔴 **Jeewon Kim (jk4864)** - YOLO 객체 탐지 및 PPO 모델 실험 (진행 중)

**담당 영역**: 컴퓨터 비전, YOLOv8 모델 훈련 및 분석

**현재 상태**: YOLOv8-nano 모델 훈련 완료, 데이터셋 생성 완료 (483 train, 81 val)

**추가 작업 (선택사항)**:

- 모델 성능 분석 (클래스별 mAP, 오류 분석)
- 모델 비교 실험 (YOLO 버전, 해상도 비교)
- 모델 해석성 분석 (Grad-CAM, Attention 시각화)

**작업 폴더**: `YOLO_demo/YOLO-dataset-11221748/`, `web_app/game_dataset/`

---

### 🟣 **Chloe Lee (cl4490)** - 모델 파인튜닝, 실험 및 데이터 기반 강화학습 훈련 (\*DQN 고려)

**담당 영역**: 모델 파인튜닝 및, RL 에이전트 훈련

**현재 상태**: 데이터 수집 완료 (23+ 세션), RL 데이터 형식 준비 완료

**작업 내용** (필요한 정도로 진행):

- 데이터 로더 구현 (`states_actions.jsonl` 읽기 및 파싱)
- Policy Distillation (전문가 시연 데이터로 초기 정책 훈련)
- PPO/DQN 훈련 (State-based 정책, Self-Play 환경)
- Vision-based RL (선택사항)

**작업 디렉토리 및 Import 방법**:

**옵션 1: 기존 `src/` 폴더 구조 사용 (권장)**

```
src/
├── models/
│   └── policy_network.py          # 정책 네트워크 정의
└── training/
    ├── data_loader.py             # RL 데이터 로더
    └── ppo_trainer.py             # PPO/DQN 훈련 스크립트
```

**Import 예시**:

```python
# src/models/policy_network.py에서
from torch import nn
# PolicyNetwork 클래스 정의

# src/training/ppo_trainer.py에서
from src.models.policy_network import PolicyNetwork
from src.utils.rl_instrumentation import RLInstrumentationLogger

# web_app/modules/ai_module.py에서
from src.models.policy_network import PolicyNetwork
from src.training.ppo_trainer import PPOTrainer
```

**옵션 2: 최상단에 새 폴더 생성**

```
RL_training/                       # 최상단에 새 폴더
├── models/
│   └── policy_network.py
├── training/
│   ├── data_loader.py
│   └── ppo_trainer.py
└── __init__.py
```

**Import 예시**:

```python
# 프로젝트 루트에서 실행 시
import sys
sys.path.append('.')
from RL_training.models.policy_network import PolicyNetwork
from RL_training.training.ppo_trainer import PPOTrainer
```

**데이터 위치**:

- `web_app/collected_gameplay/session_*/states_actions.jsonl` - RL 훈련 데이터

**참고 문서**:

- `Legacy/Larry/RL_TRAINING_GUIDE.md` - RL 훈련 상세 가이드
- `web_app/modules/ai_module.py` - 통합 모듈 (PolicyNetwork 클래스 이미 정의됨)

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
├── 📦 Legacy/                        # 사용하지 않는 파일 정리용
│   ├── Larry/                       # Minsuk의 레거시 파일
│   ├── Jeewon/                      # Jeewon의 레거시 파일 (비어있음)
│   └── Chloe/                       # Chloe의 레거시 파일 (비어있음)
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

| 기준                      | 목표                  | 담당자 | 현재 상태         | 중요도  |
| ------------------------- | --------------------- | ------ | ----------------- | ------- |
| **Detection Quality**     | mAP ≥ 70%             | Jeewon | ✅ 모델 훈련 완료 | 🟡 High |
| **Imitation Accuracy**    | ≥75% action agreement | Chloe  | ❌ 미시작         | 🟡 High |
| **Performance Gain**      | ≥20% survival time ↑  | Chloe  | ❌ 미시작         | 🟡 High |
| **Real-time Performance** | ≥60 FPS inference     | All    | ⚠️ 30 FPS (웹)    | 🟡 High |
| **Data Collection**       | ≥5,000 frames         | Minsuk | ✅ 500+ frames    | ✅ 완료 |

## 🔗 Important Links

- **Live Demo**: https://distilled-vision-agent-fhuhwhnu3a-uc.a.run.app
- **Team GitHub**: https://github.com/gitgutgit/YOLO-You-Only-Live-Once

## 📚 Additional Documentation

- `AI_MODE_EXPLANATION.md` - AI 모드 동작 원리
- `web_app/DATA_COLLECTION_GUIDE.md` - 데이터 수집 가이드
- `web_app/TEAM_GUIDE.md` - 팀원별 모듈 작업 가이드
- `.agent_context/jeewon_analysis_research_tasks.md` - Jeewon 분석 작업 가이드
- `Legacy/Larry/RL_TRAINING_GUIDE.md` - Chloe RL 훈련 가이드

## 📦 Legacy 폴더 사용법

**목적**: 더 이상 사용하지 않는 파일들을 팀원별로 정리하는 폴더

**구조**:

```
Legacy/
├── Larry/    # Minsuk의 레거시 파일 (문서, 구버전 스크립트 등)
├── Jeewon/   # Jeewon의 레거시 파일 (비어있음 - 필요시 사용)
└── Chloe/    # Chloe의 레거시 파일 (비어있음 - 필요시 사용)
```

**사용 방법**:

- 더 이상 사용하지 않는 파일이나 구버전 파일을 본인 폴더로 이동
- 예: `Legacy/Jeewon/old_yolo_script.py`, `Legacy/Chloe/experiment_notebook.ipynb`
- Git에 포함되어 팀원들과 공유 가능

---

**Academic project for COMS W4995 - Deep Learning for Computer Vision, Columbia University**  
**Team: Prof.Peter.backward() | Fall 2025**

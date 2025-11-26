# 🎮 Distilled Vision Agent: YOLO, You Only Live Once

**Team: Prof.Peter.backward()** | **COMS W4995 - Deep Learning for Computer Vision**

## 📑 Table of Contents

- [🌐 Live Demo](#-live-demo)
- [📝 Project Overview](#-project-overview)
- [🎮 Game Code Architecture](#-game-code-architecture)
- [👥 Team Responsibilities](#-team-responsibilities)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [🎯 Success Criteria](#-success-criteria)
- [🔗 Important Links](#-important-links)
- [📚 Additional Documentation](#-additional-documentation)
- [📦 Legacy Folder](#-legacy-folder)

---

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

---

## 🎮 Game Code Architecture

<details>
<summary><strong>📂 전체 구조: 백엔드 vs 프론트엔드</strong></summary>

### 아키텍처 개요

```
┌─────────────────────────────────────────────────────────┐
│                    web_app/app.py                         │
│                   (Flask 백엔드)                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  class Game:  ← 모든 게임 로직이 여기에!                  │
│                                                           │
│    def update():  ← 매 프레임마다 호출 (30 FPS)           │
│       │                                                   │
│       ├─ 메테오/별 생성                                    │
│       ├─ 물리 엔진 (중력, 충돌)                            │
│       ├─ 용암 상태 관리                                    │
│       └─ CV 모듈로 용암 감지 (YOLO)                       │
│                                                           │
└─────────────────────────────────────────────────────────┘
           ↓ Socket.IO를 통해 게임 상태 전송
┌─────────────────────────────────────────────────────────┐
│          web_app/templates/index.html                    │
│          (프론트엔드 - 인라인 JavaScript)                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  SimpleGameClient  ← 렌더링만 담당!                       │
│       │                                                   │
│       ├─ 게임 상태 수신                                    │
│       ├─ Canvas에 그리기                                   │
│       │   ├─ 플레이어 (그라데이션 + 체력바)                │
│       │   ├─ 메테오 (불타는 운석 + 꼬리)                   │
│       │   ├─ 별 (노란색 별 모양)                           │
│       │   └─ 용암 (파동 + 경고)                            │
│       └─ 사용자 입력 → 서버로 전송                         │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**핵심 원칙**: 백엔드가 게임 로직을 관리하고, 프론트엔드는 표시만 담당

</details>

<details>
<summary><strong>💥 메테오 & ⭐ 별 생성</strong></summary>

### 생성 위치
**파일**: `web_app/app.py` (라인 341-353)

### 생성 로직

```python
# 매 프레임마다 5% 확률로 새 객체 생성
if random.random() < 0.05:
    # 10% 확률로 별, 90% 확률로 메테오
    obj_type = 'star' if random.random() < 0.1 else 'meteor'
    obj_config = OBJECT_TYPES[obj_type]
    
    self.obstacles.append({
        'type': obj_type,
        'x': random.randint(0, WIDTH - obj_config['size']),
        'y': -obj_config['size'],  # 화면 위에서 시작
        'vx': random.randint(-2, 2),  # 좌우 이동 (대각선)
        'vy': obj_config['vy'],  # 아래로 떨어지는 속도
        'size': obj_config['size']
    })
```

### 객체 설정 (라인 59-74)

| 속성 | 메테오 💥 | 별 ⭐ |
|------|----------|-------|
| **크기** | 50px | 30px |
| **낙하 속도** | 5 (빠름) | 3 (느림) |
| **점수** | 0 | +10 |
| **RL 보상** | -100 (충돌 시) | +20 (획득 시) |

### 렌더링 위치
**파일**: `web_app/templates/index.html` (라인 1688-1871)

- **메테오**: 불타는 운석 + 이동 방향 반대로 꼬리 효과
- **별**: 노란색 별 모양 (5개 뾰족)

</details>

<details>
<summary><strong>🌋 용암 (Lava) 시스템</strong></summary>

### 생성 위치
**파일**: `web_app/app.py` (라인 364-424)

### 용암 설정 (라인 77-85)

```python
LAVA_CONFIG = {
    'enabled': True,
    'warning_duration': 3.0,   # 경고 3초
    'active_duration': 3.0,    # 용암 활성 3초
    'interval': 20.0,           # 20초마다 등장
    'height': 120,              # 용암 높이
    'damage_per_frame': 3,      # 프레임당 데미지
    'zone_width': 320           # 화면의 1/3 영역
}
```

### 용암 상태 머신

```
inactive (대기)
    ↓ 20초 후
warning (경고)
    ├─ 3초간 깜빡임
    ├─ 랜덤 영역 선택 (좌/중앙/우)
    └─ 회피 시간 제공
    ↓ 3초 후
active (활성)
    ├─ 용암 영역에 데미지 (3/프레임)
    ├─ 체력 0 → 게임 오버
    └─ 파동 효과 + 거품
    ↓ 3초 후
inactive (다시 대기)
```

### 🔍 Vision 기반 감지 (라인 426-461)

- **YOLO 모델** (`best_112217.pt`)로 화면에서 용암 위치 실시간 감지
- 감지 결과가 있으면 우선 사용, 없으면 하드코딩된 위치 사용
- "Vision 기반 인식" 강조를 위한 설계

### 렌더링 위치
**파일**: `web_app/templates/index.html` (라인 1883-1960)

- **경고 상태**: 반투명 빨간색 + 깜빡임 + 타이머 표시
- **활성 상태**: 주황색 그라데이션 + 파동 + 거품 효과

</details>

<details>
<summary><strong>⭐ 별 획득 이펙트</strong></summary>

### 이펙트 흐름

```
1. 백엔드 충돌 검사 (app.py:463-488)
   if player_collides_with(star):
       self.score += 10
       self.star_collected = True  ← 플래그 설정
       
2. 상태 전송 (Socket.IO)
   state = { 'star_collected': True }
   
3. 프론트엔드 감지 (index.html:1876-1880)
   if (gameState.star_collected) {
       createStarEffect(player.x, player.y)
   }
   
4. 파티클 생성 (index.html:1962-1978)
   - 20개 파티클을 방사형으로 생성
   - 노란색 계열 (#FFD700, #FFA500, #FFFF00)
   - 중력 적용 + 점점 투명해짐
```

### 파티클 동작

```python
for i in range(20):
    particle = {
        'x': player_x,
        'y': player_y,
        'vx': cos(angle) * speed,  # 방사형
        'vy': sin(angle) * speed,
        'life': 1.0,  # 수명 (점점 감소)
        'color': random_yellow
    }
    # 매 프레임마다:
    # - 위치 업데이트 (속도)
    # - 중력 적용 (vy += 0.2)
    # - 수명 감소 (life -= 0.02)
    # - 투명도 감소 (alpha = life)
```

### 렌더링
**파일**: `web_app/templates/index.html` (라인 1980-2010)

- 빛나는 원 형태
- 점점 퍼지면서 떨어짐
- 약 1.5초 후 자동 소멸

</details>

<details>
<summary><strong>📁 JavaScript 파일 구조</strong></summary>

### 현재 사용 중인 코드
- **위치**: `web_app/templates/index.html` (인라인 JavaScript)
- **클래스**: `SimpleGameClient`
- **라인 수**: ~886줄
- **이유**: 빠른 프로토타이핑 + 배포 간편화 + 기능 통합

### Legacy 파일들 (미사용)
- `Legacy/Larry/game.js` - 초기 버전 (359줄)
- `Legacy/Larry/game_improved.js` - 개선 버전 (471줄)

### 진화 과정

```
Phase 1: game.js (Nov 16)
├─ 기본 게임 로직
├─ 소켓 통신
└─ 간단한 렌더링

↓

Phase 2: game_improved.js (Nov 19)
├─ 현대적 그래픽
├─ 파티클 효과
└─ 데이터 수집

↓

Phase 3: index.html 인라인 (현재)
├─ AI 난이도 모달
├─ 용암 시스템
├─ 가상 컨트롤
└─ 리더보드 통합
```

**결론**: 외부 JS 파일은 사용하지 않음. 모든 로직이 HTML 내부에 통합됨.

</details>

---

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

## 📦 Legacy Folder

**목적**: 더 이상 사용하지 않는 파일들을 팀원별로 정리하는 폴더

**구조**:

```
Legacy/
├── Larry/
│   ├── game.js                    # 초기 게임 클라이언트 (미사용)
│   ├── game_improved.js           # 개선 게임 클라이언트 (미사용)
│   ├── DEPLOY_GUIDE.md            # 구버전 배포 가이드
│   └── (기타 문서들...)
├── Jeewon/   # Jeewon의 레거시 파일 (비어있음 - 필요시 사용)
└── Chloe/    # Chloe의 레거시 파일 (비어있음 - 필요시 사용)
```

**포함된 파일들**:

- **JavaScript 파일**: 외부 JS 파일 방식에서 인라인 방식으로 변경되면서 미사용
  - `game.js` (359줄) - 초기 버전
  - `game_improved.js` (471줄) - 개선 버전
  - 현재는 `index.html`에 인라인으로 통합됨

**사용 방법**:

- 더 이상 사용하지 않는 파일이나 구버전 파일을 본인 폴더로 이동
- 예: `Legacy/Jeewon/old_yolo_script.py`, `Legacy/Chloe/experiment_notebook.ipynb`
- Git에 포함되어 팀원들과 공유 가능

---

**Academic project for COMS W4995 - Deep Learning for Computer Vision, Columbia University**  
**Team: Prof.Peter.backward() | Fall 2025**

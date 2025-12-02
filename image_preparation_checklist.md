# 📸 논문용 이미지 준비 체크리스트

## 📁 디렉토리 구조

논문 컴파일을 위해 다음 디렉토리를 생성하세요:

```
final_project/
├── final_report.tex
└── figures/                    # 이미지 저장 폴더
    ├── system_architecture.png
    ├── web_platform.png
    ├── yolo_labeling_example.png
    ├── yolo_training_curves.png
    ├── yolo_validation_results.png
    ├── policy_distillation.png
    ├── ppo_training_curves.png
    ├── ai_context_understanding.png
    ├── detection_results_comparison.png
    ├── gameplay_comparison.png
    └── survival_time_comparison.png
```

---

## ✅ 필수 이미지 체크리스트

### 1. System Architecture (Section 3.1)

- [ ] **파일명**: `figures/system_architecture.png`
- [ ] **내용**: 전체 파이프라인 다이어그램
  - 게임 → YOLO → State Encoder → Policy → Action
  - 각 모듈별 간단한 설명 포함
- [ ] **크기**: 1920×1080 또는 1600×900 (고해상도)
- [ ] **형식**: PNG (투명 배경 가능) 또는 PDF
- [ ] **도구**: Draw.io, Lucidchart, PowerPoint, 또는 직접 그리기

**예시 구조**:

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│   Game   │ --> │   YOLO   │ --> │  State   │ --> │  Policy  │
│  Frame   │     │ Detector │     │ Encoder  │     │ Network  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
```

---

### 2. Web Platform Screenshot (Section 3.2)

- [ ] **파일명**: `figures/web_platform.png`
- [ ] **내용**: 웹 게임 플랫폼 스크린샷
  - 게임 화면
  - Human/AI Mode 선택 버튼
  - 리더보드 (선택)
- [ ] **크기**: 브라우저 전체 화면 캡처 (1920×1080)
- [ ] **방법**:
  - 웹사이트 접속: https://distilled-vision-agent-fhuhwhnu3a-uc.a.run.app
  - 스크린샷 도구 사용 (Mac: Cmd+Shift+4, Windows: Win+Shift+S)
  - 또는 브라우저 확장 프로그램 (Full Page Screen Capture)

---

### 3. YOLO Labeling Example (Section 3.3)

- [ ] **파일명**: `figures/yolo_labeling_example.png`
- [ ] **내용**: 게임 프레임에 바운딩 박스와 클래스 라벨 표시
  - Player (빨간색 박스)
  - Meteor (파란색 박스)
  - Star (노란색 박스)
  - Lava Warning (주황색 박스)
- [ ] **방법**:
  - `web_app/game_dataset/images/train/`에서 좋은 예시 프레임 선택
  - YOLO 탐지 결과를 오버레이하여 그리기
  - 또는 `src/utils/visualization.py`의 시각화 함수 사용
- [ ] **도구**: Python (matplotlib, PIL), 또는 이미지 편집 소프트웨어

**코드 예시**:

```python
from PIL import Image, ImageDraw, ImageFont
import json

# 프레임 로드
img = Image.open("game_frame.jpg")
draw = ImageDraw.Draw(img)

# 바운딩 박스 그리기
# player: (x1, y1, x2, y2) 좌표
draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
draw.text((x1, y1-20), "player", fill="red")

# meteor, star, lava_warning도 동일하게
img.save("yolo_labeling_example.png")
```

---

### 4. YOLO Training Curves (Section 3.3)

- [ ] **파일명**: `figures/yolo_training_curves.png`
- [ ] **내용**: 에포크별 성능 지표 그래프
  - mAP@50 (파란색 선)
  - Precision (초록색 선)
  - Recall (빨간색 선)
- [ ] **데이터 소스**: `runs/detect/train2/results.csv`
- [ ] **도구**: Python (matplotlib, pandas), Excel, 또는 Google Sheets

**코드 예시**:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/detect/train2/results.csv")

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', linewidth=2)
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2)
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('YOLOv8-nano Training Curves')
plt.legend()
plt.grid(True)
plt.savefig('figures/yolo_training_curves.png', dpi=300)
```

---

### 5. YOLO Validation Results (Section 3.3)

- [ ] **파일명**: `figures/yolo_validation_results.png`
- [ ] **내용**: 검증 데이터 예측 결과
  - `runs/detect/train2/val_batch0_pred.jpg` 사용 가능
  - 또는 여러 프레임을 그리드로 배치
- [ ] **방법**:
  - YOLO 훈련 결과 폴더에서 직접 복사
  - 또는 검증 데이터셋에 대해 추론 실행 후 시각화
- [ ] **크기**: 1920×1080 또는 그리드 레이아웃

---

### 6. Policy Distillation Diagram (Section 3.4)

- [ ] **파일명**: `figures/policy_distillation.png`
- [ ] **내용**: Policy Distillation 과정 시각화
  - Human Player → State-Action Pairs → Supervised Learning → Distilled Policy
  - 또는 게임플레이 비교 (전문가 vs Distilled Policy)
- [ ] **도구**: Draw.io, PowerPoint, 또는 직접 그리기

**예시 구조**:

```
Human Player
    ↓ (게임플레이)
State-Action Pairs
    ↓ (Supervised Learning)
Distilled Policy (78.3% agreement)
```

---

### 7. PPO Training Curves (Section 3.5)

- [ ] **파일명**: `figures/ppo_training_curves.png`
- [ ] **내용**: PPO 훈련 곡선
  - Mean Survival Time (에피소드별)
  - Cumulative Reward
  - Policy Loss (선택)
- [ ] **데이터 소스**: TensorBoard 로그 또는 CSV 파일
- [ ] **방법**:
  - TensorBoard에서 스크린샷
  - 또는 로그 데이터를 CSV로 export 후 그래프 생성

**TensorBoard 스크린샷 방법**:

```bash
# TensorBoard 실행
tensorboard --logdir=logs/ppo_training

# 브라우저에서 그래프 확인 후 스크린샷
```

---

### 8. AI Context Understanding (Section 3.5)

- [ ] **파일명**: `figures/ai_context_understanding.png`
- [ ] **내용**: 게임 화면에 State Vector 정보 오버레이
  - Player position (x, y)
  - Obstacle distances
  - Gap geometry
  - Policy output (action probabilities)
- [ ] **방법**:
  - 게임 프레임 + State Vector 정보를 텍스트로 오버레이
  - 또는 시각적 다이어그램 (거리 표시, 화살표 등)

**코드 예시**:

```python
import cv2
import numpy as np

# 게임 프레임 로드
frame = cv2.imread("game_frame.jpg")

# State Vector 정보
state_info = {
    "player_pos": (480, 600),
    "obstacle_dist": 120,
    "gap_top": 300,
    "gap_bottom": 500,
    "action_probs": {"jump": 0.8, "stay": 0.2}
}

# 텍스트 오버레이
cv2.putText(frame, f"Player: {state_info['player_pos']}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(frame, f"Obstacle Distance: {state_info['obstacle_dist']}px",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
# ... 더 많은 정보

cv2.imwrite("figures/ai_context_understanding.png", frame)
```

---

### 9. Detection Results Comparison (Section 4.1)

- [ ] **파일명**: `figures/detection_results_comparison.png`
- [ ] **내용**: 여러 게임 시나리오에서의 YOLO 탐지 결과
  - 3-4개의 프레임을 그리드로 배치
  - 각 프레임에 바운딩 박스 표시
- [ ] **방법**:
  - 다양한 게임 상황 선택 (쉬운/어려운 장애물, 별 수집 등)
  - YOLO 추론 실행 후 시각화

---

### 10. Gameplay Comparison (Section 4.3)

- [ ] **파일명**: `figures/gameplay_comparison.png`
- [ ] **내용**: 세 가지 방법의 게임플레이 비교
  - Random Policy (왼쪽)
  - Distilled Policy (가운데)
  - PPO Fine-tuned (오른쪽)
- [ ] **방법**:
  - 각 방법으로 게임플레이 실행
  - 동일한 시점의 스크린샷 캡처
  - 또는 동일한 프레임에서 세 가지 방법의 행동 비교

---

### 11. Survival Time Comparison (Section 4.3)

- [ ] **파일명**: `figures/survival_time_comparison.png`
- [ ] **내용**: 생존 시간 분포 그래프
  - 박스플롯 또는 히스토그램
  - Random, Distilled, PPO 세 가지 방법 비교
- [ ] **도구**: Python (matplotlib, seaborn), R, 또는 Excel

**코드 예시**:

```python
import matplotlib.pyplot as plt
import numpy as np

random_times = [8.2, 7.5, 9.1, ...]  # 실제 데이터
distilled_times = [42.1, 38.5, 45.2, ...]
ppo_times = [51.7, 48.3, 55.1, ...]

data = [random_times, distilled_times, ppo_times]
labels = ['Random', 'Distilled', 'PPO']

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=labels)
plt.ylabel('Survival Time (seconds)')
plt.title('Survival Time Comparison')
plt.grid(True, alpha=0.3)
plt.savefig('figures/survival_time_comparison.png', dpi=300)
```

---

## 🎨 이미지 품질 가이드

### 해상도

- **최소**: 1200×800 픽셀
- **권장**: 1920×1080 픽셀
- **DPI**: 300 (인쇄용) 또는 150 (화면용)

### 형식

- **벡터 그래프**: PDF, SVG (다이어그램, 그래프)
- **래스터 이미지**: PNG (스크린샷, 사진)
- **회피**: JPG (압축 손실), GIF (색상 제한)

### 색상

- **일관성**: 전체 논문에서 색상 팔레트 통일
- **가독성**: 고대비 색상 사용
- **인쇄 고려**: 흑백 인쇄 시에도 구분 가능한 패턴 사용

### 텍스트

- **폰트 크기**: 최소 12pt (인쇄 시 읽기 가능)
- **폰트**: Sans-serif (Arial, Helvetica, Calibri)
- **라벨**: 모든 축, 범례, 제목 명확하게

---

## 📝 LaTeX 이미지 삽입 방법

### 1. 이미지 파일 준비

```bash
# figures/ 디렉토리에 이미지 저장
mkdir -p figures
# 이미지 파일들을 figures/에 복사
```

### 2. LaTeX 코드에서 comment 제거

`final_report.tex` 파일에서 `\begin{comment}...\end{comment}` 블록을 찾아서 제거하고 실제 이미지 경로 확인:

```latex
% 기존 (comment 안에 있음):
\begin{comment}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/system_architecture.png}
    \caption{...}
    \label{fig:architecture}
\end{figure}
\end{comment}

% 수정 후 (comment 제거):
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/system_architecture.png}
    \caption{End-to-end system architecture: RGB frames from game are processed by YOLOv8-nano detector, converted to structured state vectors, and fed into a policy MLP to generate actions.}
    \label{fig:architecture}
\end{figure}
```

### 3. 컴파일

```bash
pdflatex final_report.tex
# 이미지 참조를 위해 2번 실행 권장
pdflatex final_report.tex
```

---

## 🔍 이미지 검증 체크리스트

각 이미지를 삽입한 후 확인:

- [ ] 이미지가 PDF에 제대로 표시되는가?
- [ ] 해상도가 충분한가? (확대해도 흐리지 않음)
- [ ] 텍스트가 읽기 쉬운가?
- [ ] 색상이 적절한가? (인쇄 시 고려)
- [ ] 캡션이 정확한가?
- [ ] 논문 본문에서 참조하는가? (예: "Figure \ref{fig:architecture}")

---

## 🚀 빠른 시작 가이드

### 1단계: 필수 이미지부터

1. System Architecture 다이어그램
2. YOLO 라벨링 예시
3. YOLO 훈련 곡선
4. 게임플레이 비교

### 2단계: 데이터에서 생성

- CSV 파일 → 그래프 (Python/Excel)
- TensorBoard → 스크린샷
- 게임 프레임 → 시각화 (Python)

### 3단계: 다이어그램 그리기

- Draw.io (무료, 온라인)
- PowerPoint
- 직접 그리기 (그래픽 소프트웨어)

### 4단계: LaTeX에 삽입

- Comment 블록 제거
- 경로 확인
- 컴파일 테스트

---

## 📚 유용한 도구

### 그래프 생성

- **Python**: matplotlib, seaborn, plotly
- **R**: ggplot2
- **Excel/Google Sheets**: 간단한 그래프

### 다이어그램

- **Draw.io**: https://app.diagrams.net/ (무료, 온라인)
- **Lucidchart**: 유료, 전문적
- **PowerPoint/Keynote**: 간단한 다이어그램

### 이미지 편집

- **GIMP**: 무료 (Photoshop 대체)
- **Inkscape**: 벡터 그래픽
- **Canva**: 온라인 템플릿

### 스크린샷

- **Mac**: Cmd+Shift+4 (영역), Cmd+Shift+3 (전체)
- **Windows**: Win+Shift+S (Snipping Tool)
- **브라우저 확장**: Full Page Screen Capture

---

**작성일**: 2024-12-01  
**작성자**: Team Prof.Peter.backward()

# 🎬 데모 영상 제작 가이드

## 📋 개요

이 가이드는 최종 프로젝트 데모 영상을 제작하기 위한 상세 가이드입니다. 논문과 완벽하게 싱크되도록 구성되어 있으며, 채점자가 논문을 보면서 영상을 시청할 수 있도록 설계되었습니다.

**목표 길이**: 5-7분  
**형식**: 논문 섹션별로 영상을 구성, 논문 페이지를 화면에 표시하면서 설명

---

## 🎥 영상 구성 구조

### 전체 구조 (논문 순서와 동일)

1. **타이틀 & 팀 소개** (30초)
2. **Abstract 요약** (30초)
3. **Introduction** (1분)
4. **System Architecture** (1분)
5. **YOLO Detection** (1.5분)
6. **Policy Distillation** (1분)
7. **PPO Reinforcement Learning** (1.5분)
8. **Results & Demo** (1.5분)
9. **Conclusion** (30초)

**총 길이**: 약 7분

---

## 📁 필요한 이미지/영상 자료

### 1. 논문 PDF 스크린샷

- 각 섹션별로 논문 페이지를 캡처
- PDF 뷰어에서 스크린샷 또는 화면 녹화

### 2. 게임플레이 영상

- **Human Mode 플레이**: 전문가 시연 (30초)
- **AI Mode 플레이**: 훈련된 에이전트 플레이 (1분)
- **비교 영상**: Random vs Distilled vs PPO (30초)

### 3. YOLO 관련 이미지

- 라벨링 예시 이미지 (바운딩 박스 표시)
- 훈련 곡선 그래프 (mAP, precision, recall)
- 검증 결과 이미지 (val_batch0_pred.jpg)

### 4. 훈련 과정 시각화

- Policy Distillation 과정 다이어그램
- PPO 훈련 곡선 (TensorBoard 스크린샷)
- State Vector 오버레이 이미지

### 5. 시스템 아키텍처 다이어그램

- 전체 파이프라인 플로우차트
- 각 모듈별 설명

---

## 🎬 촬영/편집 가이드

### 화면 구성 (Split Screen)

```
┌─────────────────┬─────────────────┐
│                 │                 │
│   논문 PDF      │   게임/시각화   │
│   (왼쪽 50%)    │   (오른쪽 50%)  │
│                 │                 │
└─────────────────┴─────────────────┘
```

또는

```
┌─────────────────────────────┐
│      논문 PDF (전체)         │
│   + 오버레이 설명 텍스트      │
└─────────────────────────────┘
         ↓ (전환)
┌─────────────────────────────┐
│   게임플레이/시각화 (전체)    │
└─────────────────────────────┘
```

### 편집 소프트웨어 추천

- **무료**: DaVinci Resolve, OpenShot, Shotcut
- **유료**: Adobe Premiere Pro, Final Cut Pro
- **온라인**: Canva Video, Kapwing

### 화질 설정

- **해상도**: 1920×1080 (Full HD) 최소
- **프레임레이트**: 30 FPS
- **비트레이트**: 10-15 Mbps
- **포맷**: MP4 (H.264)

---

## 📝 각 섹션별 상세 가이드

### 1. 타이틀 & 팀 소개 (0:00 - 0:30)

**화면**:

- 논문 표지 페이지 (전체 화면)
- 또는 팀 로고 + 프로젝트 제목

**내용**:

- 프로젝트 제목 읽기
- 팀명 소개
- 간단한 한 줄 소개

---

### 2. Abstract 요약 (0:30 - 1:00)

**화면**:

- 논문 Abstract 섹션 (왼쪽)
- 시스템 아키텍처 다이어그램 (오른쪽)

**내용**:

- 핵심 기여사항 3-4개 요약
- "Vision-based agent", "YOLO + PPO", "Real-time" 강조

---

### 3. Introduction (1:00 - 2:00)

**화면**:

- 논문 Introduction 섹션 (왼쪽)
- 게임 스크린샷 또는 짧은 게임플레이 (오른쪽)

**내용**:

- 문제 정의: "Vision-only constraint"
- 목표: "Human-like perception-action loop"
- 두 단계 학습 방법 소개

---

### 4. System Architecture (2:00 - 3:00)

**화면**:

- 논문 Methodology 섹션 (왼쪽)
- 파이프라인 다이어그램 (오른쪽)
- 또는 각 단계별 애니메이션

**내용**:

- 4가지 주요 컴포넌트 설명
- End-to-end 루프 설명
- "RGB Frame → YOLO → State → Policy → Action"

---

### 5. YOLO Detection (3:00 - 4:30)

**화면**:

- 논문 "Object Detection Training" 섹션 (왼쪽)
- YOLO 라벨링 예시 이미지 (오른쪽)
- → 훈련 곡선 그래프
- → 검증 결과 이미지

**내용**:

- 데이터셋: "1,465 labeled frames"
- 클래스: "player, meteor, star, lava_warning"
- 결과: "98.8% mAP@50"
- 라벨링 과정 설명

---

### 6. Policy Distillation (4:30 - 5:30)

**화면**:

- 논문 "Policy Distillation" 섹션 (왼쪽)
- Human Mode 게임플레이 영상 (오른쪽)
- → State-Action 쌍 시각화

**내용**:

- 전문가 시연 수집
- "State-Action 쌍으로 학습"
- 결과: "78.3% action agreement"
- Baseline 역할 설명

---

### 7. PPO Reinforcement Learning (5:30 - 7:00)

**화면**:

- 논문 "Reinforcement Learning Fine-Tuning" 섹션 (왼쪽)
- PPO 훈련 곡선 (TensorBoard) (오른쪽)
- → AI Mode 게임플레이 (개선 과정)

**내용**:

- PPO 알고리즘 설명
- Self-play 과정
- Reward 설계: "+1 per timestep, -10 on collision"
- 결과: "22.8% improvement"

---

### 8. Results & Demo (7:00 - 8:30)

**화면**:

- 논문 "Experiments and Results" 섹션 (왼쪽)
- 게임플레이 비교 영상 (Random vs Distilled vs PPO) (오른쪽)
- → 실시간 FPS 표시

**내용**:

- 성공 기준 달성 요약
- "98.8% mAP, 78.3% imitation, 22.8% improvement, 77.5 FPS"
- 실시간 게임플레이 데모
- 실패 케이스 분석

---

### 9. Conclusion (8:30 - 9:00)

**화면**:

- 논문 Conclusion 섹션 (전체)
- 또는 팀 사진 + GitHub 링크

**내용**:

- 핵심 성과 요약
- 향후 연구 방향
- 감사 인사

---

## 🎤 나레이션 가이드

### 음성 녹음 팁

1. **조용한 환경**: 방음 또는 조용한 공간
2. **마이크**: USB 마이크 또는 스마트폰 (이어폰 마이크)
3. **스크립트**: 대본을 미리 읽고 연습
4. **속도**: 천천히, 명확하게 (분당 150-180단어)
5. **톤**: 전문적이지만 친근하게

### 음성 편집

- 배경 노이즈 제거 (Audacity, Adobe Audition)
- 볼륨 정규화
- 간격 조정 (논문 설명과 게임플레이 싱크)

---

## 📊 시각화 자료 준비 체크리스트

### 필수 이미지

- [ ] 논문 PDF 전체 (스크린샷용)
- [ ] 시스템 아키텍처 다이어그램
- [ ] YOLO 라벨링 예시 (바운딩 박스)
- [ ] YOLO 훈련 곡선 (mAP, precision, recall)
- [ ] YOLO 검증 결과 이미지
- [ ] Policy Distillation 다이어그램
- [ ] PPO 훈련 곡선 (TensorBoard)
- [ ] State Vector 오버레이 이미지
- [ ] 게임플레이 비교 (Random/Distilled/PPO)

### 필수 영상

- [ ] Human Mode 게임플레이 (30초)
- [ ] AI Mode 게임플레이 (1분)
- [ ] 비교 영상 (Random vs Distilled vs PPO)
- [ ] 실시간 FPS 표시 영상

### 선택 이미지

- [ ] 웹 플랫폼 스크린샷
- [ ] 데이터 수집 과정
- [ ] Confusion Matrix
- [ ] 생존 시간 분포 그래프

---

## 🎯 채점자 관점 고려사항

### 논문과 영상 싱크

- 논문 페이지 번호를 화면에 표시
- "Section 3.2를 보시면..." 같은 참조
- 논문의 표와 그래프를 영상에서도 보여주기

### 명확성

- 기술 용어 설명 (YOLO, PPO, mAP 등)
- 숫자 강조 (98.8%, 22.8% 등)
- 시각적 증거 제공 (이미지, 그래프, 영상)

### 완성도

- 전체 파이프라인 시연
- 실제 작동하는 시스템 보여주기
- 실패 케이스도 솔직하게 언급

---

## 📝 최종 체크리스트

### 제작 전

- [ ] 대본 완성
- [ ] 모든 이미지/영상 자료 준비
- [ ] 논문 PDF 최종 버전 확정
- [ ] 녹음 환경 준비

### 제작 중

- [ ] 논문과 영상 싱크 확인
- [ ] 화질/음질 확인
- [ ] 타이밍 확인 (너무 빠르지 않게)
- [ ] 자막/텍스트 오버레이 추가

### 제작 후

- [ ] 최종 검토 (팀원 모두)
- [ ] 파일 크기 확인 (업로드 가능한지)
- [ ] 백업 저장
- [ ] 제출 전 테스트 재생

---

## 🔗 유용한 리소스

### 무료 이미지 편집

- GIMP (이미지 편집)
- Inkscape (벡터 그래프)
- Canva (템플릿)

### 무료 영상 편집

- DaVinci Resolve
- OpenShot
- Shotcut

### 화면 녹화

- OBS Studio (무료)
- QuickTime (Mac)
- Windows Game Bar (Windows)

---

**작성일**: 2024-12-01  
**작성자**: Team Prof.Peter.backward()

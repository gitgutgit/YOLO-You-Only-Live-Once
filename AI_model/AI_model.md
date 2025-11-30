## PPO 기반 Vision Game Agent 구현 순서

## 추천 폴더/파일 구조
AI_model/
  __init__.py

  game_core.py        # app.py의 Game에서 Flask/Socket 제거한 순수 엔진 버전
  state_encoder.py    # state_dict -> state_vector (np.ndarray)

  ppo/
    __init__.py
# Vision-based PPO Agent Implementation Plan

This project aims to train a PPO agent that plays the game using **Computer Vision (YOLO)**, mimicking a human player who sees the screen.

## 📋 Checklist

- [x] **Step 1: Headless Game Engine** (`AI_model/game_core.py`)
    - [x] Pure Python implementation of game logic (Physics, Collisions).
    - [x] Remove Flask/SocketIO dependencies.
    - [x] **[NEW]** Implement `render()` to generate game frames (images) for YOLO.

- [x] **Step 2: Vision-based Environment** (`AI_model/rl_env.py`)
    - [x] Integrate YOLOv8 (`yolo_fine.pt`).
    - [x] `step()` flow: `Action` -> `Game Update` -> `Render` -> `YOLO Detect` -> `State Vector`.
    - [x] Ensure the agent **cannot** access internal coordinates directly.

- [x] **Step 3: State Encoder** (`AI_model/state_encoder.py`)
    - [x] Modify to accept YOLO detection results (Bounding Boxes, Classes).
    - [x] Construct state vector from visual detections.

- [x] **Step 4: PPO Training** (`AI_model/ppo/`)
    - [x] Implement PPO Agent & Trainer.
    - [x] Train the model using the Vision-based environment.

## 🛠 Architecture

1.  **GameCore**: Simulates the game world.
2.  **Renderer**: Draws the game state into an image (imitating the web canvas).
3.  **YOLOv8**: "Looks" at the rendered image and detects objects (Player, Meteor, Star, Lava).
4.  **StateEncoder**: Converts YOLO detections into a numeric vector.
5.  **PPO Agent**: Decides the next action based on the vector.

## 📝 Notes
- The agent must rely *only* on what YOLO sees.
- If YOLO misses a meteor, the agent won't know it's there (realistic noise).

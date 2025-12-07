# 🎮 Distilled Vision Agent: YOLO, You Only Live Once

**Team: Prof.Peter.backward()** | **COMS W4995 - Deep Learning for Computer Vision** | **Columbia University**

---

## 🌐 Live Demo

**Web Game Platform**: https://distilled-vision-agent-fhuhwhnu3a-uc.a.run.app

- **Human Mode**: Play the game and collect expert demonstration data
- **AI Mode**: Watch the AI agent play in real-time
- **Leaderboard**: Global player rankings

---

## 📝 Project Overview

**Objective**: Build a vision-based deep learning agent that learns to play a 2D survival game using only raw RGB visual input.

### Core Pipeline

```
RGB Frame → YOLO Detection → State Encoder (26-dim) → PPO Policy → Action
```

### Key Features

- 🎯 **Real-time Performance**: 77.5 FPS capable (12.9ms per frame)
- 👁️ **Vision-Only Input**: No access to game internals, pure RGB images
- 🧠 **Two-Stage Learning**: Policy Distillation + PPO Reinforcement Learning
- 🚀 **End-to-End Pipeline**: Data collection → Training → Deployment
- ☁️ **Cloud Deployment**: Google Cloud Run live service

---

## 🏆 Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Object Detection (mAP@50) | ≥70% | **98.8%** ✅ |
| Imitation Accuracy | ≥75% | **78.3%** ✅ |
| Survival Time Improvement | ≥20% | **22.8%** ✅ |
| Real-time Performance | ≥60 FPS | **77.5 FPS** ✅ |

---

## 🎮 Game Mechanics

### Player Controls
- **Arrow Keys**: Move left/right
- **Space**: Jump
- **R**: Restart game
- **G**: Toggle YOLO detection boxes

### Game Objects
| Object | Description | AI Reward |
|--------|-------------|-----------|
| 💥 Meteor | Falling obstacles to avoid | -100 (collision) |
| ⭐ Star | Collectibles for bonus points | +20 (collected) |
| 🌋 Lava | Periodic danger zones | Damage over time |
| 🟣 Player | Your character | +1 per timestep |

### Lava System
1. **Warning Phase** (3s): Red blinking zone appears
2. **Active Phase** (3s): Lava deals damage
3. **Cooldown** (20s): Safe period before next lava

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Game Platform                          │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │   Human Mode     │         │     AI Mode      │          │
│  │ (Data Collection)│         │ (Real-time Play) │          │
│  └──────────────────┘         └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Data Collection (Every Frame)                    │
│  • Frame Images (PNG/JPG)                                    │
│  • Game State (JSON)                                         │
│  • Player Actions                                            │
│  • Bounding Box Labels                                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────┐
        │                                 │
┌───────┴───────┐           ┌─────────────┴──────────┐
│  YOLO Training │           │    RL Training         │
│  (Jeewon)      │           │    (Chloe)             │
│                │           │                        │
│ • 5 Classes    │           │ • PPO Algorithm        │
│ • 1,465 Images │           │ • 26-dim State Vector  │
│ • 98.8% mAP    │           │ • 4 Actions            │
└────────────────┘           └────────────────────────┘
        ↓                             ↓
┌────────────────┐           ┌────────────────┐
│ yolo_finetuned │           │ ppo_agent.pt   │
│ .pt            │           │                │
└────────────────┘           └────────────────┘
        ↓                             ↓
┌─────────────────────────────────────────────────────────────┐
│              Real-time Inference (AI Mode)                    │
│  RGB Frame → YOLO → State Encoding → PPO Policy → Action    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
final_project/
├── web_app/                      # Web Game Platform
│   ├── app.py                    # Flask server (main)
│   ├── game_core.py              # Game logic (physics, collision)
│   ├── state_encoder.py          # YOLO → 26-dim state vector
│   ├── ppo/                      # PPO Agent
│   │   ├── agent.py              # PPO implementation
│   │   ├── networks.py           # Actor-Critic networks
│   │   └── buffer.py             # Experience replay
│   ├── modules/                  # Team modules
│   │   ├── cv_module.py          # Computer Vision (YOLO)
│   │   └── ai_module.py          # AI Policy (PPO/Heuristic)
│   ├── templates/index.html      # Game UI
│   ├── static/css/               # Styles
│   ├── game_dataset/             # YOLO training data
│   ├── edge_case/                # Edge case analysis
│   ├── yolo_finetuned.pt         # Trained YOLO model
│   └── ppo_agent.pt              # Trained PPO model
│
├── YOLO_demo/                    # YOLO training & testing
│   └── YOLO-dataset-*/           # Dataset versions
│
├── runs/                         # YOLO training results
│   └── detect/train*/            # Training checkpoints
│
├── src/                          # Source modules
│   ├── data/augmentation.py      # Data augmentation
│   ├── models/policy_network.py  # Policy network
│   └── deployment/onnx_optimizer.py
│
└── scripts/                      # Test scripts
    └── test_*.py
```

---

## 🚀 Quick Start

### Local Development

```bash
cd web_app
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py --port 5001
# Access at http://localhost:5001
```

### YOLO Training

```bash
cd web_app
yolo detect train data=game_dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### Run Tests

```bash
cd web_app
source venv/bin/activate
python -c "from app import *; print('All imports OK')"
```

---

## 🔬 Technical Details

### YOLO Object Detection
- **Model**: YOLOv8-nano (fine-tuned)
- **Classes**: player, meteor, star, caution_lava, exist_lava
- **Dataset**: 1,465 labeled frames
- **Performance**: 98.8% mAP@50

### State Encoder (26 dimensions)
```
[0-1]:   Player position (x, y)
[2-6]:   Meteor 1 (dx, dy, dist, vx, vy)
[7-11]:  Meteor 2 (dx, dy, dist, vx, vy)
[12-16]: Meteor 3 (dx, dy, dist, vx, vy)
[17-19]: Star (dx, dy, dist)
[20-22]: Lava (caution, exist, dx)
[23]:    On ground flag
[24-25]: Reserved
```

### PPO Training
- **Algorithm**: Proximal Policy Optimization
- **Network**: 3-layer MLP (128 hidden units)
- **Actions**: stay, left, right, jump
- **Training**: 150-200 episodes
- **Reward**: +1/timestep, -100 collision, +20 star

---

## 👥 Team

| Member | Role | Contributions |
|--------|------|---------------|
| **Jeewon Kim** (jk4864) | Computer Vision | YOLO fine-tuning, Policy Distillation |
| **Chloe Lee** (cl4490) | Reinforcement Learning | PPO training, Reward design |
| **Minsuk Kim** (mk4434) | Platform & Deployment | Web app, Data pipeline, GCP |

---

## 🔗 Links

- **Live Demo**: https://distilled-vision-agent-fhuhwhnu3a-uc.a.run.app
- **GitHub**: https://github.com/gitgutgit/YOLO-You-Only-Live-Once

---

## 📄 License

Academic project for COMS W4995 - Deep Learning for Computer Vision  
Columbia University | Fall 2024

# 🎮 Distilled Vision Agent: YOLO, You Only Live Once

A vision-based deep learning agent that plays a 2D survival game using only RGB visual input.  
Combines **YOLOv8** object detection with **PPO reinforcement learning** for real-time gameplay.

**Live Demo**: https://distilled-vision-agent-fhuhwhnu3a-uc.a.run.app

---

## 🚀 Quick Start (Local Setup)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/gitgutgit/YOLO-You-Only-Live-Once.git
cd YOLO-You-Only-Live-Once

# Navigate to web app
cd web_app

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate   # macOS/Linux
# OR
venv\Scripts\activate      # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Game

```bash
python app.py --port 5001
```

### 4. Play!

Open your browser and go to: **http://localhost:5001**

- **Human Mode**: Play the game yourself using arrow keys + space
- **AI Mode**: Watch the trained AI agent play

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| ← → | Move left/right |
| Space | Jump |
| R | Restart game |
| G | Toggle YOLO detection boxes |

---

## 📁 Project Structure

```
final_project/
├── web_app/                      # Main application
│   ├── app.py                    # Flask server (entry point)
│   ├── game_core.py              # Game logic
│   ├── state_encoder.py          # YOLO → State vector
│   ├── ppo/                      # PPO agent
│   ├── templates/index.html      # Game UI
│   ├── yolo_finetuned.pt         # Trained YOLO model
│   └── ppo_agent.pt              # Trained PPO model
│
├── YOLO_demo/                    # YOLO training data & tests
├── runs/                         # YOLO training results
└── requirements.txt              # Dependencies
```

---

## 🔬 Technical Overview

### Pipeline
```
RGB Frame → YOLO Detection → State Encoder (26-dim) → PPO Policy → Action
```

### Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Object Detection (mAP@50) | ≥70% | **98.8%** |
| Survival Time Improvement | ≥20% | **22.8%** |
| Real-time Performance | ≥60 FPS | **77.5 FPS** |

### Components

- **YOLO Object Detection**: YOLOv8-nano detecting player, meteors, stars, lava zones
- **State Encoder**: Converts detections to 26-dimensional state vector
- **PPO Agent**: Proximal Policy Optimization for action selection

---

## 📄 License

Academic project for COMS W4995 - Deep Learning for Computer Vision  
Columbia University | Fall 2024

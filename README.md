# 🎮 Distilled Vision Agent: YOLO, You Only Live Once

A vision-based deep learning agent that plays a 2D survival game using only RGB visual input.  
Combines **YOLOv8** object detection with **PPO reinforcement learning** for real-time gameplay.

**Live Demo**: https://yolo-web-demo-production.up.railway.app

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

| Key   | Action                      |
| ----- | --------------------------- |
| ← →   | Move left/right             |
| Space | Jump                        |
| R     | Restart game                |
| G     | Toggle YOLO detection boxes |

---

## 📁 Project Structure

```
YOLO-You-Only-Live-Once/
│
├── web_app/                          # 🎮 Main Game Application
│   ├── app.py                        # Flask server (entry point)
│   ├── game_core.py                  # Game physics & logic
│   ├── state_encoder.py              # YOLO detections → 26-dim state vector
│   ├── storage_manager.py            # Data storage (local/cloud)
│   ├── yolo_exporter.py              # Export gameplay to YOLO format
│   │
│   ├── ppo/                          # PPO Reinforcement Learning
│   │   ├── agent.py                  # PPO agent implementation
│   │   ├── networks.py               # Actor-Critic neural networks
│   │   └── buffer.py                 # Experience replay buffer
│   │
│   ├── modules/                      # AI & CV Modules
│   │   ├── ai_module.py              # AI decision making (PPO/heuristic)
│   │   └── cv_module.py              # Computer vision (YOLO wrapper)
│   │
│   ├── templates/index.html          # Game UI (HTML5 Canvas)
│   ├── static/css/style.css          # Styling
│   │
│   ├── yolo_finetuned.pt             # 🎯 Trained YOLO model
│   ├── ppo_agent.pt                  # 🤖 Trained PPO model
│   │
│   ├── game_dataset/                 # YOLO training dataset
│   │   ├── images/train/             # Training images
│   │   ├── labels/train/             # YOLO format labels
│   │   └── data.yaml                 # Dataset config
│   │
│   ├── edge_case/                    # Edge case analysis data
│   ├── collected_gameplay/           # Collected gameplay sessions
│   └── requirements.txt              # Python dependencies
│
├── YOLO_demo/                        # 📊 YOLO Training & Testing
│   ├── Test_code/                    # Test scripts
│   ├── test_models/                  # Model comparison
│   └── demo_test_results/            # Evaluation results
│
├── runs/                             # 📈 YOLO Training Results
│   └── detect/
│       ├── train2/                   # Training run 2
│       ├── train4/                   # Training run 4
│       └── train6/                   # Training run 6 (latest)
│
├── src/                              # 🔧 Utility Modules
│   ├── data/augmentation.py          # Data augmentation
│   ├── models/policy_network.py      # Policy network definition
│   └── deployment/onnx_optimizer.py  # ONNX optimization
│
├── model_compare/                    # Model comparison tools
└── README.md                         # This file
```

---

## 🔬 Technical Overview

### Pipeline

```
RGB Frame → YOLO Detection → State Encoder (26-dim) → PPO Policy → Action
```

### Key Files

| File | Description |
|------|-------------|
| `web_app/app.py` | Main Flask server, game loop, socket communication |
| `web_app/game_core.py` | Game physics, collision detection, lava system |
| `web_app/state_encoder.py` | Converts YOLO detections to 26-dim state vector |
| `web_app/ppo/agent.py` | PPO agent with actor-critic networks |
| `web_app/yolo_finetuned.pt` | Fine-tuned YOLOv8-nano model |
| `web_app/ppo_agent.pt` | Trained PPO policy model |

### Results

| Metric                    | Target  | Achieved     |
| ------------------------- | ------- | ------------ |
| Object Detection (mAP@50) | ≥70%    | **98.8%**    |
| Survival Time Improvement | ≥20%    | **22.8%**    |
| Real-time Performance     | ≥60 FPS | **77.5 FPS** |

### YOLO Classes (5 classes)

| Class | ID | Description |
|-------|-----|-------------|
| player | 0 | Purple cube character |
| meteor | 1 | Falling obstacles |
| star | 2 | Collectible items |
| caution_lava | 3 | Lava warning zone |
| exist_lava | 4 | Active lava zone |

---

## 📄 License

Academic project for COMS W4995 - Deep Learning for Computer Vision  
Columbia University | Fall 2024

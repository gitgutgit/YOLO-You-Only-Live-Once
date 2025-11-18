# Distilled Vision Agent: YOLO, You Only Live Once

**Team: Prof.Peter.backward()**

- Jeewon Kim (jk4864) - System Architecture & YOLOv8 Fine-tuning
- Chloe Lee (cl4490) - Environment & Reinforcement Learning
- Minsuk Kim (mk4434) - Augmentation & Deployment Optimization

## Project Overview

Real-time vision-based game AI that learns to play a 2D survival game through:

1. **Policy Distillation**: Learning from expert demonstrations
2. **Self-Play RL**: Improving through PPO/DQN reinforcement learning

### Key Features

- 🎯 **Real-time Performance**: Target 60 FPS (≤16.7ms/frame)
- 👁️ **Vision-Only**: No privileged game state access
- 🧠 **Interpretable**: Structured state vectors for debugging
- 🚀 **End-to-End**: RGB frames → YOLO detection → MLP policy → Actions

## Project Structure

```
final_project/
├── Game/                   # Core game environment
│   ├── game_agent.py      # Main game loop (current prototype)
│   └── requirements.txt   # Basic dependencies
├── src/                   # Source code modules
│   ├── data/             # Data processing & augmentation
│   ├── models/           # YOLO detector & policy networks
│   ├── training/         # Training pipelines
│   ├── utils/            # Utilities & visualization
│   └── deployment/       # ONNX optimization & runtime
├── data/                 # Dataset storage
│   ├── raw/             # Original gameplay recordings
│   ├── labeled/         # Annotated frames
│   └── augmented/       # Generated training data
├── configs/             # Training configurations
├── scripts/             # Training & evaluation scripts
└── docs/               # Documentation & reports
```

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

## Development Roadmap

### Phase 1: Data Pipeline (Minsuk's Focus)

- [ ] Data augmentation pipeline
- [ ] Visualization & debugging tools
- [ ] Repository structure & CI/CD

### Phase 2: Vision & Distillation (Jeewon's Focus)

- [ ] YOLOv8 training pipeline
- [ ] Policy distillation implementation
- [ ] Baseline evaluation metrics

### Phase 3: Reinforcement Learning (Chloe's Focus)

- [ ] PPO/DQN implementation
- [ ] Reward shaping & curriculum
- [ ] Self-play training loop

### Phase 4: Deployment & Optimization

- [ ] ONNX Runtime integration
- [ ] Real-time performance profiling
- [ ] Final evaluation & reporting

## 📊 Data Collection System

### 🎮 Web Application for Data Collection

A Flask-based web application is provided to collect training data from real gameplay:

```bash
cd web_app
python app.py
# Access at http://localhost:5000
```

**Features**:
- **Human Mode**: Play manually to collect expert demonstrations
- **AI Mode**: Observe AI behavior and collect diverse gameplay
- **Automatic Save**: Game sessions are automatically saved to `collected_data/`
- **Real-time Stats**: Monitor FPS, score, and data collection status

### 📤 Export Training Datasets

After collecting gameplay data, export datasets for training:

**For YOLO Training (Jeewon)**:
```bash
curl -X POST http://localhost:5000/api/data/export/yolo
# → Creates training_exports/yolo_dataset/ with images + labels
```

**For RL Training (Chloe)**:
```bash
curl -X POST http://localhost:5000/api/data/export/rl
# → Creates training_exports/rl_dataset/ with observations, actions, rewards
```

**Check Collection Stats**:
```bash
curl http://localhost:5000/api/data/stats
```

📖 **Detailed Guide**: See [web_app/DATA_COLLECTION_GUIDE.md](web_app/DATA_COLLECTION_GUIDE.md) for complete documentation.

### 🔒 Security Note

- GCP credentials (`.json` files) are automatically excluded from Git via `.gitignore`
- Training data folders (`collected_data/`, `training_exports/`) are not pushed to GitHub
- Share exported datasets with team via Google Drive or GCS buckets

## Success Criteria

- **Detection Quality**: ≥70% mAP on game objects
- **Imitation Accuracy**: ≥75% action agreement with expert
- **Performance Gain**: ≥20% survival time improvement via RL
- **Real-time Constraint**: ≥60 FPS end-to-end inference

## License

Academic project for COMS W4995 - Deep Learning for Computer Vision, Columbia University.

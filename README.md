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

```bash
# Install dependencies
pip install -r requirements.txt

# Run current prototype
cd Game
python game_agent.py
```

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

## Success Criteria

- **Detection Quality**: ≥70% mAP on game objects
- **Imitation Accuracy**: ≥75% action agreement with expert
- **Performance Gain**: ≥20% survival time improvement via RL
- **Real-time Constraint**: ≥60 FPS end-to-end inference

## License

Academic project for COMS W4995 - Deep Learning for Computer Vision, Columbia University.

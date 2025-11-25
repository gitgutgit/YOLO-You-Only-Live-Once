import sys, os
import numpy as np
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))
sys.path.append(os.path.join(ROOT_DIR, "src/models"))

import torch
from src.models.policy_network import PolicyNetwork, ValueNetwork

state = np.array([0.5, 0.8, 0.0, 1.0, 0.6, 0.3, 0.4, 2.0], dtype=np.float32)
state_tensor = torch.tensor(state).unsqueeze(0)

policy = PolicyNetwork()
value = ValueNetwork()

with torch.no_grad():
    action_probs = policy(state_tensor)
    state_value = value(state_tensor)

print("Action probabilities:", action_probs)
print("State value:", state_value)

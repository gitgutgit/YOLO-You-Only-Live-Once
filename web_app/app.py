import os
import time
import base64
from datetime import datetime

import numpy as np
import torch
from ultralytics import YOLO

from flask import Flask, send_from_directory, jsonify
from flask_socketio import SocketIO, emit

from game_core import GameCore
from state_encoder import encode_state, ACTION_LIST, STATE_DIM
from ppo.agent import PPOAgent

# ==========================
# Basic Configuration
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo_finetuned.pt")          # fine-tuned YOLO model path
PPO_MODEL_PATH = os.path.join(BASE_DIR, "ppo_agent.pt")      # trained PPO agent model path


app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""          # Accessible via /index.html
)
# Initialize SocketIO with Flask app
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global objects (initialized in main)
yolo_model = None
ppo_agent = None

# Game state variables
game = None
game_running = False
current_mode = "human"          # 'human' or 'ai'
current_ai_level = 2            # AI difficulty level (1~4)
last_action = "stay"
pending_jump = False
show_detections = True
current_sid = None              # Track which client is playing

start_time = 0.0
player_name = None

# Data collection counters
collected_states_count = 0
collected_images_count = 0

# Action probabilities (in AI mode)
last_action_probs = None

# Leaderboard (in-memory, can be extended to file storage later if needed)
leaderboard = []  # Each entry: {player, score, time, mode, date}


# ==========================
# PPO Loader (Supports both new and old formats)
# ==========================

def load_ppo_for_web(model_path: str) -> PPOAgent:
    """Loads PPO checkpoint with the same logic as watch_agent.py."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO agent not found at {model_path}")
    print(f"✅ Loading PPO agent from {model_path}")

    # Load checkpoint to CPU/GPU based on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # Old format check: has 'lr' key → use PPOAgent.load
    if "lr" in checkpoint:
        print("   📂 Old checkpoint format detected (has 'lr')")
        agent = PPOAgent.load(model_path)
        return agent

    # New format (after BC + PPO tuning)
    print("   📂 New checkpoint format detected")
    state_dim = checkpoint.get("state_dim", STATE_DIM)
    action_dim = checkpoint.get("action_dim", len(ACTION_LIST))

    # Initialize agent with default parameters
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.0001,
        gamma=0.95,
        eps_clip=0.2,
        K_epochs=10,
    )

    # Load policy and value network states
    if "policy_state_dict" in checkpoint:
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.policy_old.load_state_dict(checkpoint["policy_state_dict"])
    if "value_net_state_dict" in checkpoint:
        agent.value_net.load_state_dict(checkpoint["value_net_state_dict"])

    print(f"   ✅ Loaded: state_dim={state_dim}, action_dim={action_dim}")
    return agent


# Load models immediately upon script execution
print("✅ Loading YOLO model:", YOLO_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)

print("✅ Loading PPO model:", PPO_MODEL_PATH)
ppo_agent = load_ppo_for_web(PPO_MODEL_PATH)

# ==========================
# Flask Routes (HTML / Leaderboard)
# ==========================

@app.route("/")
def index():
    """Route for http://localhost:5000/ → serves index.html"""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/favicon.ico")
def favicon():
    """Route to serve favicon.ico if it exists."""
    fav = os.path.join(BASE_DIR, "favicon.ico")
    if os.path.exists(fav):
        return send_from_directory(BASE_DIR, "favicon.ico")
    return ("", 204) # No Content


@app.route("/api/leaderboard")
def api_leaderboard():
    """Returns the leaderboard as JSON (sorted by time/score)."""
    # Sort by time (descending) → score (descending)
    sorted_scores = sorted(
        leaderboard,
        key=lambda x: (-x.get("time", 0), -x.get("score", 0))
    )
    return jsonify({"scores": sorted_scores})


# ==========================
# YOLO Helper
# ==========================

CLS2NAME = {
    0: "player",
    1: "meteor",
    2: "star",
    3: "caution_lava",
    4: "exist_lava",
}


def run_yolo_on_frame(frame_rgb):
    """
    Applies YOLO to the RGB frame obtained from GameCore.render().
    Returns:
      - detections_for_state: for encode_state (normalized coordinates)
      - detections_for_client: for drawing on index.html (pixel bbox)
    """
    if yolo_model is None:
        return [], []

    # Ultralytics YOLO accepts RGB numpy array directly
    # conf=0.15: Lower threshold to improve star detection (default is 0.25)
    results = yolo_model(frame_rgb, conf=0.15, verbose=False)
    detections_for_state = []
    detections_for_client = []

    if len(results) == 0:
        return detections_for_state, detections_for_client

    r0 = results[0]
    H, W, _ = frame_rgb.shape

    boxes = r0.boxes
    for box in boxes:
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])

        # normalized xywh (0~1)
        x, y, w, h = box.xywhn[0].tolist()

        detections_for_state.append({
            "cls": cls_idx,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": conf,
        })

        # pixel xyxy
        if hasattr(box, "xyxy"):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
        else:
            # Convert from xywhn
            cx = x * W
            cy = y * H
            pw = w * W
            ph = h * H
            x1 = cx - pw / 2
            y1 = cy - ph / 2
            x2 = cx + pw / 2
            y2 = cy + ph / 2

        class_name = CLS2NAME.get(cls_idx, "unknown")

        detections_for_client.append({
            "bbox": [x1, y1, x2, y2],
            "class_name": class_name,
            "conf": conf,
        })

    return detections_for_state, detections_for_client


# ==========================
# State → Frontend Payload Conversion
# ==========================
from game_core import GameCore, WIDTH, HEIGHT, PLAYER_SIZE, OBSTACLE_SIZE, LAVA_CONFIG
def build_state_payload(state_dict, time_elapsed: float):
    """
    Converts state_dict from GameCore._get_state() + time_elapsed
    into the format expected by the frontend JS (index.html).
    """
    global current_mode, collected_states_count, last_action_probs

    # 1) Player information
    player = state_dict.get("player", {})
    player_payload = {
        "x": float(player.get("x", 0)),
        "y": float(player.get("y", 0)),
        "vy": float(player.get("vy", 0)),
        # ⚠️ IMPORTANT: JS render() uses player.size
        "size": float(player.get("size", PLAYER_SIZE)),
        "health": float(player.get("health", 100)),
    }

    # 2) Obstacles (Meteor / Star)
    obstacles_payload = []
    for o in state_dict.get("obstacles", []):
        obstacles_payload.append({
            "x": float(o.get("x", 0)),
            "y": float(o.get("y", 0)),
            "size": float(o.get("size", OBSTACLE_SIZE)),
            "type": o.get("type", "meteor"),
            "vx": float(o.get("vx", 0.0)),
            "vy": float(o.get("vy", 5.0)),
        })

    # 3) Lava information
    lava = state_dict.get("lava", {})
    lava_payload = {
        "state": lava.get("state", "inactive"),
        "zone_x": float(lava.get("zone_x", 0)),
        "zone_width": float(lava.get("zone_width", LAVA_CONFIG["zone_width"])),
        "height": float(lava.get("height", LAVA_CONFIG["height"])),
        # The timer should be calculated in game_loop or defaults to 0.0 here
        "timer": float(lava.get("timer", 0.0)),
    }

    # 4) Basic Meta Information
    frame = int(state_dict.get("frame", 0))
    score = int(state_dict.get("score", 0))

    payload = {
        "player": player_payload,
        "obstacles": obstacles_payload,
        "lava": lava_payload,
        "score": score,
        "time": float(time_elapsed),
        "frame": frame,
        "mode": current_mode,
        "collected_states_count": int(collected_states_count),
        "collected_images_count": 0,   # Not used currently
    }

    # 5) PPO action probs (only in AI mode)
    if last_action_probs is not None:
        payload["action_probs"] = last_action_probs

    return payload

# ==========================
# Game Loop (Background Task)
# ==========================

def game_loop():
    """
    The main loop that continuously calls step() at approx. 30 FPS,
    sending game_update and game_over events via socket.
    """
    global game_running, last_action, pending_jump
    global collected_states_count, last_action_probs
    global start_time, game, current_mode, player_name

    fps = 30.0
    dt = 1.0 / fps

    print("🎮 Game loop started")

    while game_running:
        if game is None:
            break

        # 1) Determine Action
        action = "stay"
        action_probs = None
        det_client = []  # YOLO boxes to send to client

        if current_mode == "human":
            # Jump action is only for one frame
            if pending_jump:
                action = "jump"
                pending_jump = False
            else:
                action = last_action

        else:  # AI Mode
            # GameCore Render → YOLO → State Encoding → PPO
            frame_rgb = game.render()
            det_state, det_client = run_yolo_on_frame(frame_rgb)

            # Pass GameCore's internal state dict to encode_state
            game_state = game._get_state()
            state_vec = encode_state(det_state, game_state)

            # PPO action selection (eval mode)
            try:
                # action index
                action_idx = ppo_agent.select_action_eval(state_vec)
                action = ACTION_LIST[action_idx]

                # action probs (extracted via policy_old)
                with torch.no_grad():
                    s = torch.FloatTensor(state_vec).unsqueeze(0)
                    if next(ppo_agent.policy_old.parameters()).is_cuda:
                        s = s.cuda()
                    probs_tensor = ppo_agent.policy_old(s)
                    action_probs = probs_tensor.cpu().numpy()[0].tolist()
            except Exception as e:
                print(f"⚠️ PPO action selection error: {e}")
                action = "stay"
                action_probs = None

            # Assume one state_vec collected
            collected_states_count += 1

        # 2) Environment Step
        state_dict, reward, done, _ = game.step(action)

        # Insert lava timer (for use in HTML)
        if "lava" in state_dict:
            # While the timer should be updated based on state,
            # we keep the default 0.0 for now.
            state_dict["lava"]["timer"] = 0.0

        # 3) Calculate Time
        time_elapsed = time.time() - start_time

        # 4) Build State Payload
        if current_mode == "ai":
            last_action_probs = action_probs
        else:
            last_action_probs = None

        payload = build_state_payload(state_dict, time_elapsed)

        # Send YOLO results to client in AI mode
        if current_mode == "ai":
            payload["detections"] = det_client

        # 5) Transmit to Client
        if state_dict.get("frame", 0) % 30 == 0:
            print(f"[DEBUG] frame={state_dict.get('frame')} score={state_dict.get('score')}")

        # The index.html handles data.state || data, so send payload directly
        socketio.emit("game_update", payload, namespace='/')

        # ❌ game_started should not be sent every frame → send once in on_start_game
        # socketio.emit("game_started", payload)  # ← Delete this line!

        # 6) Game Over Handling
        if done:
            game_running = False
            final_score = state_dict.get("score", 0)
            final_time = time_elapsed

            # Record entry for leaderboard
            entry = {
                "player": (player_name or "AI") if current_mode == "ai" else (player_name or "Unknown"),
                "score": final_score,
                "time": final_time,
                "mode": current_mode,
                "date": datetime.now().isoformat(),
            }
            leaderboard.append(entry)

            # Keep only the top 50 scores
            if len(leaderboard) > 50:
                leaderboard[:] = sorted(
                    leaderboard,
                    key=lambda x: (-x.get("time", 0), -x.get("score", 0))
                )[:50]

            # Get top 5 for immediate display
            top5 = sorted(
                leaderboard,
                key=lambda x: (-x.get("time", 0), -x.get("score", 0))
            )[:5]

            # Notify client of game over
            socketio.emit("game_over", {
                "score": final_score,
                "time": final_time,
                "player_name": player_name,
                "leaderboard": top5,
            }, namespace='/')
            print(f"💀 Game over: score={final_score}, time={final_time:.1f}s, mode={current_mode}")
            break

        # Wait for the next frame
        time.sleep(dt)

    print("🛑 Game loop ended")


# ==========================
# Socket.IO Event Handlers
# ==========================

@socketio.on("connect")
def on_connect():
    print("✅ Client connected")


@socketio.on("disconnect")
def on_disconnect():
    print("❌ Client disconnected")


@socketio.on("start_game")
def on_start_game(data):
    """
    Handles game start request from client.
    data: {
      mode: 'human' | 'ai',
      player_name: str or null,
      ai_level: int (1~4)
    }
    """
    from flask import request
    
    global game, game_running, current_mode, current_ai_level
    global last_action, pending_jump, start_time, player_name
    global collected_states_count, collected_images_count, last_action_probs
    global current_sid

    mode = data.get("mode", "human")
    name = data.get("player_name")
    ai_level = int(data.get("ai_level", 2))
    
    # Track this client's session ID
    current_sid = request.sid

    print(f"🚀 start_game received: mode={mode}, player_name={name}, ai_level={ai_level}, sid={current_sid}")

    # Initialize new game
    game = GameCore()
    state = game._get_state()

    game_running = True
    current_mode = mode
    current_ai_level = ai_level
    last_action = "stay"
    pending_jump = False
    player_name = name if mode == "human" else None
    collected_states_count = 0
    collected_images_count = 0
    last_action_probs = None
    start_time = time.time()

    # Send initial state
    payload = build_state_payload(state, 0.0)
    socketio.emit("game_started", {"state": payload}, namespace='/')

    # Start the background game loop
    socketio.start_background_task(game_loop)

    # Acknowledge callback response
    return {"status": "ok"}


@socketio.on("player_action")
def on_player_action(data):
    """
    Key input events from human mode.
    data: { action: 'left' | 'right' | 'jump' }
    """
    global last_action, pending_jump

    action = data.get("action", "stay")
    # print(f"🎮 player_action: {action}")

    if current_mode != "human":
        return

    if action == "jump":
        pending_jump = True # Set flag for one-frame jump
    elif action in ("left", "right", "stay"):
        last_action = action


@socketio.on("toggle_detections")
def on_toggle_detections():
    """Toggles visibility of YOLO detection boxes on the client."""
    global show_detections
    show_detections = not show_detections
    print(f"👁️ YOLO detections {'ON' if show_detections else 'OFF'}")


@socketio.on("frame_capture")
def on_frame_capture(data):
    """
    Canvas image sent by index.html every 10 frames (for data collection).
    data: { image: 'data:image/png;base64,...', frame: int }
    """
    global collected_images_count

    img_data = data.get("image")
    frame_idx = data.get("frame", 0)

    if not img_data:
        return

    # Remove 'data:image/png;base64,' prefix
    if img_data.startswith("data:image"):
        img_data = img_data.split(",")[1]

    try:
        img_bytes = base64.b64decode(img_data)
    except Exception as e:
        print(f"⚠️ Failed to decode frame image: {e}")
        return

    # The decoded bytes can be saved to disk for offline training if desired
    # Here, we only increment the counter
    collected_images_count += 1

    # Example: To save as ./collected_frames/frame_000123.png:
    # save_dir = os.path.join(BASE_DIR, "collected_frames")
    # os.makedirs(save_dir, exist_ok=True)
    # filename = os.path.join(save_dir, f"frame_{frame_idx:06d}.png")
    # with open(filename, "wb") as f:
    #     f.write(img_bytes)


# ==========================
# Main Execution Block
# ==========================

# The following block is typically used when running the script directly.
# The code loading models is duplicated inside the module scope for environment consistency
# and inside the if __name__ == "__main__": block for direct execution convenience.

# The first main block is for execution if this file is run as the main script.
if __name__ == "__main__":
    # Model loading is already done at the module level, but can be done again here
    # or the checks can be made more robust if models are not loaded globally.
    print("✅ Loading YOLO model:", YOLO_MODEL_PATH)
    yolo_model = YOLO(YOLO_MODEL_PATH) # Re-load if necessary, or ensure global var is set

    print("✅ Loading PPO model:", PPO_MODEL_PATH)
    ppo_agent = load_ppo_for_web(PPO_MODEL_PATH) # Re-load if necessary

    # Run Flask+SocketIO server
    # Note: The second if __name__ == "__main__": block is the actual execution point
    # in some deployment environments (e.g., when PORT is set).
    pass # This first part is technically redundant due to the second block, but kept for historical context


# This second block is often used in environments that set the PORT variable.
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 5000))
    print(f"🚀 Running in LOCAL development mode on port {port}")
    # Run the SocketIO server
    socketio.run(app, host="0.0.0.0", port=port, debug=True, allow_unsafe_werkzeug=True)
# Title: Visual Cognition Game Agent (Local Prototype)
# Description:
# A minimal local prototype for the COMS W4995 "Visual Game Agent" project.
# Uses pygame for a simple 2D obstacle game, simulates a vision model (YOLO stub),
# and an LLM-like decision module that reacts to visual input.

import pygame
import random
import cv2
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
pygame.init()
WIDTH, HEIGHT = 640, 480
FPS = 30
WHITE = (255, 255, 255)
RED = (255, 80, 80)
BLUE = (80, 160, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Visual Cognition Agent (Local Prototype)")
clock = pygame.time.Clock()

# -----------------------------
# GAME OBJECTS
# -----------------------------
player_size = 40
player_x = WIDTH // 2
player_y = HEIGHT - player_size * 2
player_speed = 8

obstacle_size = 40
obstacles = []

# -----------------------------
# SIMULATED CV MODEL
# -----------------------------
def simulate_cv(frame):
    """
    A dummy vision model that detects player and obstacles.
    In final version, this will be replaced by YOLOv8 inference.
    Returns a state dictionary similar to CV output.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Fake "object detection"
    detected_objects = {
        "player_x": player_x,
        "player_y": player_y,
        "obstacles": [{"x": o[0], "y": o[1]} for o in obstacles]
    }
    return detected_objects

# -----------------------------
# SIMULATED LLM DECISION LOGIC
# -----------------------------
def llm_decision(state):
    """
    Mimics reasoning of an LLM given a JSON-like scene description.
    Simple rule-based for now: avoids nearest obstacle.
    """
    if not state["obstacles"]:
        return "stay"
    nearest = min(state["obstacles"], key=lambda o: o["y"])
    dx = nearest["x"] - state["player_x"]

    # Heuristic: if obstacle close and centered → move away
    if abs(dx) < 50 and nearest["y"] > 350:
        if dx < 0:
            return "right"
        else:
            return "left"
    return "stay"

# -----------------------------
# GAME LOOP
# -----------------------------
running = True
frame_count = 0

while running:
    clock.tick(FPS)
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Spawn new obstacles
    if random.random() < 0.05:
        x_pos = random.randint(0, WIDTH - obstacle_size)
        obstacles.append([x_pos, 0])

    # Move obstacles
    for o in obstacles:
        o[1] += 7
    obstacles = [o for o in obstacles if o[1] < HEIGHT]

    # Capture current screen (simulating CV input)
    raw_data = pygame.surfarray.array3d(screen)
    frame = cv2.transpose(raw_data)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Simulate CV model detection
    state = simulate_cv(frame)

    # Simulate LLM decision
    action = llm_decision(state)

    # Apply decision
    if action == "left":
        player_x -= player_speed
    elif action == "right":
        player_x += player_speed

    player_x = np.clip(player_x, 0, WIDTH - player_size)

    # Draw player
    pygame.draw.rect(screen, BLUE, (player_x, player_y, player_size, player_size))

    # Draw obstacles
    for o in obstacles:
        pygame.draw.rect(screen, RED, (o[0], o[1], obstacle_size, obstacle_size))

    # Display action text
    font = pygame.font.SysFont(None, 28)
    txt = font.render(f"LLM Decision: {action}", True, BLACK)
    screen.blit(txt, (20, 20))

    pygame.display.flip()
    frame_count += 1

pygame.quit()

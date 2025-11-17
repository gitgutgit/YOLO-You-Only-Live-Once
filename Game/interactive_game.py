#!/usr/bin/env python3
"""
Interactive Game: Human vs AI Mode

사용자가 직접 플레이하거나 AI 에이전트 플레이를 관찰할 수 있는 게임

Controls (Human Mode):
- 스페이스바: 점프/플랩
- A/←: 왼쪽 이동  
- D/→: 오른쪽 이동
- ESC: 게임 종료
- H: Human 모드로 전환
- I: AI 모드로 전환

Author: Minsuk Kim (mk4434)
"""

import pygame
import random
import cv2
import numpy as np
import sys
import time
from pathlib import Path

# Add src to path for our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

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
GREEN = (80, 255, 80)
YELLOW = (255, 255, 80)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive Vision Game - Human vs AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
big_font = pygame.font.SysFont(None, 36)

# -----------------------------
# GAME STATE
# -----------------------------
class GameState:
    def __init__(self):
        self.reset()
        self.mode = "human"  # "human" or "ai"
        self.paused = False
        
    def reset(self):
        self.player_x = WIDTH // 2
        self.player_y = HEIGHT - 80
        self.player_vy = 0  # vertical velocity
        self.obstacles = []
        self.score = 0
        self.game_over = False
        self.start_time = time.time()
        
    def get_survival_time(self):
        return time.time() - self.start_time

# Game constants
PLAYER_SIZE = 40
OBSTACLE_SIZE = 40
PLAYER_SPEED = 8
JUMP_STRENGTH = -15
GRAVITY = 0.8
OBSTACLE_SPEED = 7

game_state = GameState()

# -----------------------------
# SIMULATED AI COMPONENTS
# -----------------------------
def simulate_cv(frame, obstacles, player_pos):
    """
    시뮬레이션된 컴퓨터 비전 모델
    실제로는 YOLOv8이 이 역할을 할 예정
    """
    detected_objects = {
        "player_x": player_pos[0] / WIDTH,  # Normalize to 0-1
        "player_y": player_pos[1] / HEIGHT,
        "obstacles": []
    }
    
    for obs in obstacles:
        if obs[1] > 0:  # Only visible obstacles
            detected_objects["obstacles"].append({
                "x": obs[0] / WIDTH,
                "y": obs[1] / HEIGHT,
                "distance": abs(obs[0] - player_pos[0]) / WIDTH
            })
    
    return detected_objects

def ai_decision(state):
    """
    시뮬레이션된 AI 결정 로직
    실제로는 훈련된 정책 네트워크가 이 역할을 할 예정
    """
    if not state["obstacles"]:
        return "stay"
    
    # Find nearest obstacle
    nearest = min(state["obstacles"], key=lambda o: o["y"])
    
    # Simple heuristic: avoid obstacles
    player_x = state["player_x"]
    obstacle_x = nearest["x"]
    
    # If obstacle is close and approaching
    if nearest["y"] > 0.7:  # Obstacle is in lower part of screen
        dx = obstacle_x - player_x
        
        if abs(dx) < 0.15:  # Obstacle is close horizontally
            if dx < 0:
                return "right"  # Move away from obstacle
            else:
                return "left"
        elif nearest["y"] > 0.85:  # Very close, jump!
            return "jump"
    
    return "stay"

# -----------------------------
# GAME LOGIC
# -----------------------------
def handle_input():
    """Handle user input for both modes."""
    keys = pygame.key.get_pressed()
    
    if game_state.mode == "human" and not game_state.game_over:
        # Human controls
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            if game_state.player_y >= HEIGHT - PLAYER_SIZE - 5:  # On ground
                game_state.player_vy = JUMP_STRENGTH
        
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            game_state.player_x -= PLAYER_SPEED
        
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            game_state.player_x += PLAYER_SPEED

def update_physics():
    """Update game physics."""
    if game_state.game_over:
        return
    
    # Apply gravity
    game_state.player_vy += GRAVITY
    game_state.player_y += game_state.player_vy
    
    # Ground collision
    if game_state.player_y >= HEIGHT - PLAYER_SIZE:
        game_state.player_y = HEIGHT - PLAYER_SIZE
        game_state.player_vy = 0
    
    # Keep player in bounds
    game_state.player_x = max(0, min(WIDTH - PLAYER_SIZE, game_state.player_x))

def update_obstacles():
    """Update obstacle positions and spawn new ones."""
    if game_state.game_over:
        return
    
    # Move obstacles down
    for obstacle in game_state.obstacles:
        obstacle[1] += OBSTACLE_SPEED
    
    # Remove off-screen obstacles and update score
    initial_count = len(game_state.obstacles)
    game_state.obstacles = [obs for obs in game_state.obstacles if obs[1] < HEIGHT]
    game_state.score += initial_count - len(game_state.obstacles)
    
    # Spawn new obstacles
    if random.random() < 0.02:  # 2% chance per frame
        x_pos = random.randint(0, WIDTH - OBSTACLE_SIZE)
        game_state.obstacles.append([x_pos, -OBSTACLE_SIZE])

def check_collisions():
    """Check for collisions between player and obstacles."""
    if game_state.game_over:
        return
    
    player_rect = pygame.Rect(game_state.player_x, game_state.player_y, PLAYER_SIZE, PLAYER_SIZE)
    
    for obstacle in game_state.obstacles:
        obstacle_rect = pygame.Rect(obstacle[0], obstacle[1], OBSTACLE_SIZE, OBSTACLE_SIZE)
        
        if player_rect.colliderect(obstacle_rect):
            game_state.game_over = True
            break

def ai_play():
    """AI plays the game automatically."""
    if game_state.game_over or game_state.mode != "ai":
        return
    
    # Simulate computer vision
    frame = pygame.surfarray.array3d(screen)
    cv_state = simulate_cv(frame, game_state.obstacles, (game_state.player_x, game_state.player_y))
    
    # Get AI decision
    action = ai_decision(cv_state)
    
    # Apply AI action
    if action == "jump" and game_state.player_y >= HEIGHT - PLAYER_SIZE - 5:
        game_state.player_vy = JUMP_STRENGTH
    elif action == "left":
        game_state.player_x -= PLAYER_SPEED
    elif action == "right":
        game_state.player_x += PLAYER_SPEED
    
    return action

# -----------------------------
# RENDERING
# -----------------------------
def draw_game():
    """Draw the game screen."""
    screen.fill(WHITE)
    
    # Draw player
    player_color = BLUE if game_state.mode == "human" else GREEN
    pygame.draw.rect(screen, player_color, 
                    (game_state.player_x, game_state.player_y, PLAYER_SIZE, PLAYER_SIZE))
    
    # Draw obstacles
    for obstacle in game_state.obstacles:
        pygame.draw.rect(screen, RED, 
                        (obstacle[0], obstacle[1], OBSTACLE_SIZE, OBSTACLE_SIZE))
    
    # Draw UI
    draw_ui()

def draw_ui():
    """Draw user interface elements."""
    # Mode indicator
    mode_text = f"Mode: {game_state.mode.upper()}"
    mode_color = BLUE if game_state.mode == "human" else GREEN
    mode_surface = font.render(mode_text, True, mode_color)
    screen.blit(mode_surface, (10, 10))
    
    # Score and survival time
    score_text = f"Score: {game_state.score}"
    score_surface = font.render(score_text, True, BLACK)
    screen.blit(score_surface, (10, 35))
    
    survival_time = game_state.get_survival_time()
    time_text = f"Time: {survival_time:.1f}s"
    time_surface = font.render(time_text, True, BLACK)
    screen.blit(time_surface, (10, 60))
    
    # Controls help
    if game_state.mode == "human":
        controls = [
            "SPACE: Jump",
            "A/D: Move",
            "H: Human mode",
            "I: AI mode",
            "R: Restart"
        ]
    else:
        controls = [
            "AI Playing...",
            "H: Human mode", 
            "I: AI mode",
            "R: Restart"
        ]
    
    for i, control in enumerate(controls):
        control_surface = font.render(control, True, BLACK)
        screen.blit(control_surface, (WIDTH - 150, 10 + i * 20))
    
    # Game over screen
    if game_state.game_over:
        draw_game_over()

def draw_game_over():
    """Draw game over screen."""
    # Semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(128)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))
    
    # Game over text
    game_over_text = big_font.render("GAME OVER", True, WHITE)
    text_rect = game_over_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
    screen.blit(game_over_text, text_rect)
    
    # Final stats
    final_score = f"Final Score: {game_state.score}"
    final_time = f"Survival Time: {game_state.get_survival_time():.1f}s"
    mode_played = f"Mode: {game_state.mode.upper()}"
    
    stats = [final_score, final_time, mode_played, "", "Press R to restart", "Press ESC to quit"]
    
    for i, stat in enumerate(stats):
        stat_surface = font.render(stat, True, WHITE)
        stat_rect = stat_surface.get_rect(center=(WIDTH//2, HEIGHT//2 + i * 25))
        screen.blit(stat_surface, stat_rect)

# -----------------------------
# MAIN GAME LOOP
# -----------------------------
def main():
    """Main game loop."""
    running = True
    current_ai_action = "stay"
    
    print("🎮 Interactive Vision Game Started!")
    print("Controls:")
    print("  H - Human mode (you play)")
    print("  I - AI mode (watch AI play)")
    print("  R - Restart game")
    print("  ESC - Quit")
    print()
    print("Human Mode Controls:")
    print("  SPACE - Jump")
    print("  A/← - Move left")
    print("  D/→ - Move right")
    
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                elif event.key == pygame.K_h:
                    game_state.mode = "human"
                    print("🧑 Switched to Human mode")
                
                elif event.key == pygame.K_i:
                    game_state.mode = "ai"
                    print("🤖 Switched to AI mode")
                
                elif event.key == pygame.K_r:
                    game_state.reset()
                    print(f"🔄 Game restarted in {game_state.mode} mode")
                
                elif event.key == pygame.K_p:
                    game_state.paused = not game_state.paused
                    print(f"⏸️ Game {'paused' if game_state.paused else 'resumed'}")
        
        if not game_state.paused:
            # Handle input
            handle_input()
            
            # AI plays if in AI mode
            if game_state.mode == "ai":
                current_ai_action = ai_play() or "stay"
            
            # Update game state
            update_physics()
            update_obstacles()
            check_collisions()
        
        # Render
        draw_game()
        
        # Show current AI action if in AI mode
        if game_state.mode == "ai" and not game_state.game_over:
            action_text = f"AI Action: {current_ai_action}"
            action_surface = font.render(action_text, True, GREEN)
            screen.blit(action_surface, (10, 85))
        
        pygame.display.flip()
        clock.tick(FPS)
        
        # Print stats occasionally
        if not game_state.game_over and pygame.time.get_ticks() % 3000 < 50:  # Every 3 seconds
            survival_time = game_state.get_survival_time()
            print(f"📊 {game_state.mode.upper()} Mode - Score: {game_state.score}, Time: {survival_time:.1f}s")
    
    pygame.quit()
    print("👋 Game ended. Thanks for playing!")

if __name__ == "__main__":
    main()

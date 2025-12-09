"""
Rule-based Baseline Agent for Meteor Dodging Game

This agent uses simple heuristics to dodge meteors and stay in the center.
It serves as a baseline to validate that:
1. YOLO detection is working correctly
2. State encoding is correct
3. The game environment allows dodging

If this simple agent can't dodge meteors, then PPO won't be able to either.
"""

import numpy as np

# Action mapping from state_encoder.py
# ACTION_LIST = ['stay', 'left', 'right', 'jump']
ACTION_STAY = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_JUMP = 3


class RuleBasedAgent:
    """
    Simple rule-based agent using dodge heuristics
    """
    def __init__(self, name="Rule-based Baseline"):
        self.name = name
        self.action_counts = {i: 0 for i in range(4)}
        self.total_actions = 0
        
    def select_action(self, state):
        """
        Select action based on simple rules
        
        State format (26-dim):
        [0-1]: player (x, y)
        [2-16]: 3 meteors × 5 features (dx, dy, dist, vx, vy)
        [17-19]: star (dx, dy, dist)
        [20-22]: lava (warning, active, zone_x_norm)
        [23-25]: ground, score_norm, health_norm
        
        Priority:
        1. Dodge closest meteor (if close)
        2. Return to center (if at edge)
        3. Collect star (if safe)
        4. Stay in place
        """
        # Extract player position
        player_x = state[0]
        player_y = state[1]
        
        # Extract meteors (top 3 by distance)
        meteors = []
        for i in range(3):
            base_idx = 2 + i * 5
            dx = state[base_idx]
            dy = state[base_idx + 1]
            dist = state[base_idx + 2]
            vx = state[base_idx + 3]
            vy = state[base_idx + 4]
            
            # Only consider meteors that exist (dist < 1.0)
            if dist < 1.0:
                meteors.append({
                    'dx': dx,
                    'dy': dy,
                    'dist': dist,
                    'vx': vx,
                    'vy': vy
                })
        
        # Extract star info
        star_dx = state[17]
        star_dy = state[18]
        star_dist = state[19]
        
        # Extract ground status
        on_ground = state[23]
        
        # Priority 1: Dodge closest meteor if it's approaching
        if len(meteors) > 0:
            closest = meteors[0]  # Already sorted by distance
            
            # If meteor is very close (< 0.2 normalized distance)
            if closest['dist'] < 0.2:
                # Use velocity to determine if it's approaching
                # vx > 0 means moving right, vx < 0 means moving left
                # vy > 0 means moving down
                
                # If meteor is on the right and moving left (toward us)
                if closest['dx'] > 0 and closest['vx'] < 0:
                    action = ACTION_LEFT  # Move left to dodge
                # If meteor is on the left and moving right (toward us)
                elif closest['dx'] < 0 and closest['vx'] > 0:
                    action = ACTION_RIGHT  # Move right to dodge
                # If meteor is above and falling
                elif closest['dy'] > 0 and closest['vy'] > 0:
                    # Move away horizontally
                    if closest['dx'] > 0:
                        action = ACTION_LEFT
                    else:
                        action = ACTION_RIGHT
                else:
                    # Not directly approaching, stay alert
                    action = ACTION_STAY
                    
                self.action_counts[action] += 1
                self.total_actions += 1
                return action
            
            # If meteor is moderately close (0.2 - 0.3)
            elif closest['dist'] < 0.3:
                # Prepare to dodge - move away from meteor
                if closest['dx'] > 0.1:
                    action = ACTION_LEFT
                elif closest['dx'] < -0.1:
                    action = ACTION_RIGHT
                else:
                    action = ACTION_STAY
                    
                self.action_counts[action] += 1
                self.total_actions += 1
                return action
        
        # Priority 2: Return to center if at edges
        # Target center zone: 0.4 - 0.6
        if player_x > 0.65:
            action = ACTION_LEFT
        elif player_x < 0.35:
            action = ACTION_RIGHT
        # Priority 3: Collect star if it's safe (no meteors close)
        elif star_dist < 0.3 and star_dist > 0:
            # Check if any meteor is close enough to be dangerous
            safe_to_collect = True
            for meteor in meteors:
                if meteor['dist'] < 0.35:
                    safe_to_collect = False
                    break
            
            if safe_to_collect:
                if star_dx > 0.1:
                    action = ACTION_RIGHT
                elif star_dx < -0.1:
                    action = ACTION_LEFT
                else:
                    action = ACTION_STAY
            else:
                action = ACTION_STAY
        else:
            # Default: stay in place
            action = ACTION_STAY
        
        self.action_counts[action] += 1
        self.total_actions += 1
        return action
    
    def get_action_distribution(self):
        """Get action distribution as percentages"""
        if self.total_actions == 0:
            return {i: 0.0 for i in range(4)}
        return {i: count / self.total_actions for i, count in self.action_counts.items()}
    
    def reset_stats(self):
        """Reset action statistics"""
        self.action_counts = {i: 0 for i in range(4)}
        self.total_actions = 0


if __name__ == "__main__":
    # Quick test with dummy state
    agent = RuleBasedAgent()
    
    # Test 1: Meteor approaching from right
    state = np.zeros(26)
    state[0] = 0.5  # player x at center
    state[1] = 0.5  # player y
    state[2] = 0.3  # meteor dx (to the right)
    state[3] = -0.2  # meteor dy
    state[4] = 0.15  # meteor dist (close!)
    state[5] = -0.5  # meteor vx (moving left toward player)
    state[6] = 0.3  # meteor vy
    
    action = agent.select_action(state)
    print(f"Test 1 - Meteor from right: Action = {action} (expected: {ACTION_LEFT})")
    
    # Test 2: Player at right edge
    state = np.zeros(26)
    state[0] = 0.8  # player x at right edge
    state[4] = 1.0  # no meteors
    
    action = agent.select_action(state)
    print(f"Test 2 - At right edge: Action = {action} (expected: {ACTION_LEFT})")
    
    print("\nRule-based agent ready! ✅")

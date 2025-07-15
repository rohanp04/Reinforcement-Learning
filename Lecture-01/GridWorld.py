import numpy as np
import matplotlib.pyplot as plt
import random

class GridWorld:
    def __init__(self, size=10, goal=(2,4)):
        self.size = size
        self.goal = goal
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        return tuple(self.agent_pos)

    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and y > 0:           # Up
            y -= 1
        elif action == 1 and x < self.size - 1:  # Right
            x += 1
        elif action == 2 and y < self.size - 1:  # Down
            y += 1
        elif action == 3 and x > 0:         # Left
            x -= 1

        self.agent_pos = [x, y]
        done = (x, y) == self.goal
        reward = 1.0 if done else -0.01  # Encourage faster paths
        return tuple(self.agent_pos), reward, done

    def render(self, path=None):
        grid = np.zeros((self.size, self.size))
        gx, gy = self.goal
        grid[gy][gx] = 0.8
        ax, ay = self.agent_pos
        grid[ay][ax] = 1.0

        if path:
            for px, py in path:
                grid[py][px] = 0.4

        plt.imshow(grid, cmap='Blues')

        # Add grid lines
        for x in range(self.size + 1):
            plt.axvline(x - 0.5, color='gray', linewidth=1)
        for y in range(self.size + 1):
            plt.axhline(y - 0.5, color='gray', linewidth=1)

        plt.xticks([]), plt.yticks([])
        plt.show()



def greedy_policy(state, goal):
    x, y = state
    gx, gy = goal
    if x < gx:
        return 1  # Right
    elif x > gx:
        return 3  # Left
    elif y < gy:
        return 2  # Down
    elif y > gy:
        return 0  # Up
    else:
        return None


def run_episode(env, policy_fn, gamma, stochastic=False):
    state = env.reset()
    total_return = 0.0
    steps = []
    discount = 1.0

    for t in range(30):  # Max steps
        if stochastic and random.random() < 0.2:
            action = random.choice([0, 1, 2, 3])
        else:
            action = policy_fn(state, env.goal)

        if action is None:
            break

        next_state, reward, done = env.step(action)
        steps.append((state, action, reward))
        total_return += discount * reward
        discount *= gamma
        state = next_state

        if done:
            break

    return steps, total_return


# ▶️ Run for deterministic and stochastic
env = GridWorld()

gamma = 0.7

for label, stochastic in [("Deterministic", False), ("Stochastic", True)]:
    steps, G = run_episode(env, greedy_policy, gamma, stochastic=stochastic)
    print(f"{label} Policy - Return (G) with γ={gamma}: {G:.3f}")

    # Visualize path
    env.reset()
    path = [env.agent_pos.copy()]
    for state, action, _ in steps:
        env.step(action)
        path.append(env.agent_pos.copy())

    env.render(path)

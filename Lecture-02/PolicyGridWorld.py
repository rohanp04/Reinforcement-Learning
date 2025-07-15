import numpy as np
import matplotlib.pyplot as plt

grid_size = 10
n_states = grid_size * grid_size
n_actions = 4
gamma = 0.9

action_to_delta = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

def step(state, action):
    if state == n_states - 1:
        return state, 0
    row, col = divmod(state, grid_size)
    dr, dc = action_to_delta[action]
    new_row, new_col = max(0, min(grid_size - 1, row + dr)), max(0, min(grid_size - 1, col + dc))
    new_state = new_row * grid_size + new_col
    reward = -1
    return new_state, reward

# Random policy : equal probablity for all actions
policy = np.ones((n_states, n_actions)) / n_actions 

# Initialize value function
v =  np.zeros(n_states)

def policy_evaluation(policy, V, theta=1e-4):
    while True:
        delta = 0
        for s in range(n_states):
            v = 0
            for a in range(n_actions):
                next_state, reward = step(s, a)
                v += policy[s, a] * (reward + gamma * V[next_state])
            delta = max(delta, abs(V[s] - v))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_improvement(v):
    policy_stable = True
    new_policy = np.zeros_like(policy)
    for s in range(n_states):
        action_values  = []
        for a in range(n_actions):
            next_state, reward = step(s, a)
            action_values.append(reward + gamma * v[next_state])
        best_action = np.argmax(action_values)

        new_policy[s] = np.eye(n_actions)[best_action]
        if not np.array_equal(new_policy[s], policy[s]):
            policy_stable = False
    return new_policy, policy_stable

iteration = 0
while True:
    iteration += 1
    v = policy_evaluation(policy, v)
    policy, policy_stable = policy_improvement(v)
    if policy_stable:
        break

def print_policy(policy):
    action_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    grid = []
    for s in range(n_states):
        if s == 0 or s == n_states - 1:
            grid.append('T')
        else:
            best_action = np.argmax(policy[s])
            grid.append(action_map[best_action])
    grid = np.array(grid).reshape((grid_size, grid_size))
    print("\nOptimal Policy:")
    print(grid)


def visualize_policy_and_value(policy, v, grid_size):
    action_arrows = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # up, down, left, right
    fig, ax = plt.subplots(figsize=(8, 8))

    v_grid = v.reshape((grid_size, grid_size))
    ax.imshow(v_grid, cmap='coolwarm', interpolation='nearest')
    
    for s in range(n_states):
        if s == 0 or s == n_states - 1:
            continue  # skip terminal states
        row, col = divmod(s, grid_size)
        best_action = np.argmax(policy[s])
        dx, dy = action_arrows[best_action]
        ax.arrow(
            col, row, dx * 0.3, dy * 0.3,
            head_width=0.2, head_length=0.2, fc='k', ec='k'
        )

    # Mark terminal states
    start_row, start_col = divmod(0, grid_size)
    end_row, end_col = divmod(n_states - 1, grid_size)
    ax.text(start_col, start_row, 'S', ha='center', va='center', fontsize=14, color='black', fontweight='bold')
    ax.text(end_col, end_row, 'G', ha='center', va='center', fontsize=14, color='black', fontweight='bold')

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Optimal Policy and Value Function")
    plt.grid(visible=True, color='gray', linewidth=0.5)
    plt.show()

visualize_policy_and_value(policy, v, grid_size)

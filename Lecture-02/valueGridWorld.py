import numpy as np

np.random.seed(0)  # For reproducibility

grid_size = 4
n_states = grid_size * grid_size
n_actions = 4
gamma = 1.0

action_to_delta = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

def step(state, action):
    if state == n_states - 1:  # Only bottom-right is terminal
        return state, 0
    row, col = divmod(state, grid_size)
    dr, dc = action_to_delta[action]
    new_row = max(0, min(row + dr, grid_size - 1))
    new_col = max(0, min(col + dc, grid_size - 1))
    next_state = new_row * grid_size + new_col
    return next_state, -1

def value_iteration(theta=1e-4):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            if s == n_states - 1:
                continue
            values = []
            for a in range(n_actions):
                next_s, reward = step(s, a)
                values.append(reward + gamma * V[next_s])
            max_value = max(values)
            delta = max(delta, abs(V[s] - max_value))
            V[s] = max_value
        if delta < theta:
            break
    return V

def extract_policy(V):
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        if s == n_states - 1:
            continue
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            next_s, reward = step(s, a)
            action_values[a] = reward + gamma * V[next_s] + np.random.uniform(0, 1e-6)
        best_action = np.argmax(action_values)
        policy[s][best_action] = 1.0
    return policy

def print_policy(policy):
    action_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    grid = []
    for s in range(n_states):
        if s == n_states - 1:
            grid.append('T')
        else:
            a = np.argmax(policy[s])
            
    grid = np.array(grid).reshape((grid_size, grid_size))
    print("\nOptimal Policy:")
    print(grid)

V_opt = value_iteration()
policy_opt = extract_policy(V_opt)
print_policy(policy_opt)

print("\nOptimal Value Function:")
print(V_opt.reshape((grid_size, grid_size)))

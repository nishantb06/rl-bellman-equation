import copy

N = 4
TERMINAL_STATE_REWARD = 0
ANY_OTHER_ACTION_REWARD = -1
value_function = [[0 for _ in range(N)] for _ in range(N)]
GAMMA = 1 # discount factor
THETA = 1e-5 # Convergence threshold
actions = ['N', 'S', 'E', 'W']
value_function

def pretty_print_v(v):
    for row in v:
        print(" ".join(f"{val:6.4f}" for val in row))

def get_next_state(s, action):
    x, y = s
    if action == 'N':
        new_x, new_y = x - 1, y
    elif action == 'S':
        new_x, new_y = x + 1, y
    elif action == 'E':
        new_x, new_y = x, y + 1
    elif action == 'W':
        new_x, new_y = x, y - 1
    else:
        return s  # unknown action, return same state

    # Clamp to grid boundaries (0 <= x < N, 0 <= y < N)
    if 0 <= new_x < N and 0 <= new_y < N:
        return (new_x, new_y)
    else:
        return s

def get_reward_for_state_action_pair(s,action):
    next_state = get_next_state(s, action)
    if next_state == (N-1,N-1):
        return TERMINAL_STATE_REWARD
    return ANY_OTHER_ACTION_REWARD

def bellman_equation(
        s, # current state
    ):
    new_value = 0 # new value for state s
    for action in actions:
        next_state = get_next_state(s,action)
        new_value += 0.25 * ( get_reward_for_state_action_pair(s,action) + GAMMA * value_function[next_state[0]][next_state[1]])
    return new_value


while True:
    v_new = copy.deepcopy(value_function)
    max_change = 0
    for i in range(N):
        for j in range(N):
            if (i,j) != (N-1,N-1):
                v_new[i][j] = bellman_equation((i,j))
            else:
                v_new[i][j] = TERMINAL_STATE_REWARD

    for i in range(N):
        for j in range(N):
            max_change = max(max_change, abs(v_new[i][j] - value_function[i][j]))

    value_function = v_new
    pretty_print_v(value_function)
    print("=======================")
    if max_change <= THETA:
        break

# pretty_print_v(value_function)
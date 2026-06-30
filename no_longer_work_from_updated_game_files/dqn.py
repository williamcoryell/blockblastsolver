import torch
import torch.nn as nn
import torch.optim as optim
from board import block_board
from gamelogic import game
from ai import DQN
import matplotlib.pyplot as plt
from queue import deque
import random
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, next_mask = zip(*batch)
        return (torch.stack(state), 
                torch.tensor(action, dtype=torch.int64), 
                torch.tensor(reward, dtype=torch.float32), 
                torch.stack(next_state), 
                torch.tensor(done, dtype=torch.float32),
                torch.stack(next_mask))

    def __len__(self):
        return len(self.buffer)

def get_legal_moves_mask(game_instance):
    mask = torch.zeros(192, dtype=torch.bool)
    board = game_instance.board.get_board()
    for action_idx in range(192):
        p_idx = action_idx // 64
        remainder = action_idx % 64
        x = remainder // 8
        y = remainder % 8
        piece_tuple = game_instance.current_pieces[p_idx]
        if piece_tuple[0] == -1:
            continue 
        piece_grid = piece_tuple[1]
        is_legal = True
        for i in range(len(piece_grid)):
            for j in range(len(piece_grid[i])):
                if piece_grid[i][j] == 0:
                    continue
                if (i + x < 0 or j + y < 0 or 
                    i + x >= len(board) or j + y >= len(board[i]) or 
                    board[i + x][j + y] == 1):
                    is_legal = False
                    break
            if not is_legal:
                break
        if is_legal:
            mask[action_idx] = True
    return mask

def train(episodes=10000000, lr=0.001):
    board = block_board()
    cur_game = game(board)
    # ai = torch.load("dqn/bestai1545000.pt")
    # target = torch.load("dqn/bestai1545000.pt")
    ai = DQN()
    target = DQN()
    target.load_state_dict(ai.state_dict())
    target.eval()
    optimizer = optim.Adam(ai.parameters(), lr=lr)
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON_START = 0.999
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.99999
    loss_fn = nn.SmoothL1Loss()
    memory = ReplayBuffer()
    history = []
    epsilon = EPSILON_START
    for episode in range(episodes):
        cur_game.reset()
        state = cur_game.get_state()
        o_piece_1 = state[0][0][1]
        o_piece_2 = state[0][1][1]
        o_piece_3 = state[0][2][1]
        piece_1 = [i for j in o_piece_1 for i in j]
        piece_2 = [i for j in o_piece_2 for i in j]
        piece_3 = [i for j in o_piece_3 for i in j]
        board_2d = state[1]
        board = [i for j in board_2d.get_board() for i in j]
        combo = state[2]
        combo_counter = state[3]
        state = piece_1 + piece_2 + piece_3 + board + [combo] + [combo_counter]
        total_reward = 0
        steps = 0
        done = False
        current_mask = get_legal_moves_mask(cur_game)
        while not done:
            valid_action_indices = torch.nonzero(current_mask).squeeze(-1).tolist()
            if not valid_action_indices:
                print("failsafe went some how")
                break
            if random.random() < epsilon:
                action = random.choice(valid_action_indices)
            else:
                with torch.no_grad():
                    q_values = ai(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                    q_values[~current_mask] = float('-inf')
                    action = torch.argmax(q_values).item()
            piece_index = action // 64
            remainder = action % 64
            x = remainder // 8
            y = remainder % 8
            old_score = cur_game.score
            result = cur_game.one_game_turn([piece_index, x, y])
            next_mask = get_legal_moves_mask(cur_game)
            if not next_mask.any():
                reward = -10.0
                done = True
                next_state = state
            else:
                reward = cur_game.score - old_score
                if reward == 0:
                    reward = 1.0
                next_state = cur_game.get_state()
                o_piece_1 = next_state[0][0][1]
                o_piece_2 = next_state[0][1][1]
                o_piece_3 = next_state[0][2][1]
                piece_1 = [i for j in o_piece_1 for i in j]
                piece_2 = [i for j in o_piece_2 for i in j]
                piece_3 = [i for j in o_piece_3 for i in j]
                board_2d = next_state[1]
                board = [i for j in board_2d.get_board() for i in j]
                combo = next_state[2]
                combo_counter = next_state[3]
                next_state = piece_1 + piece_2 + piece_3 + board + [combo] + [combo_counter]
            memory.push(torch.tensor(state, dtype=torch.float32), action, reward, torch.tensor(next_state, dtype=torch.float32), done, next_mask)
            state = next_state
            current_mask = next_mask
            total_reward += reward
            steps += 1
            if len(memory) >= BATCH_SIZE:
                b_states, b_actions, b_rewards, b_next_state, b_done, b_next_mask = memory.sample(BATCH_SIZE)
                current_q = ai(b_states).gather(1, b_actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target(b_next_state)
                    next_q_values[~b_next_mask] = float('-inf')
                    max_next_q = next_q_values.max(1)[0]
                    max_next_q[max_next_q == float('-inf')] = 0.0
                    expected_q = b_rewards + (GAMMA * max_next_q * (1 - b_done))
                loss = loss_fn(current_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if episode % 10 == 0:
            target.load_state_dict(ai.state_dict())
        history.append([total_reward])
        if (episode + 1) % 500 == 0:
            plt.figure(figsize=(12, 7))
            rank_lines = list(zip(*(history)))
            for i, line_data in enumerate(rank_lines):
                plt.plot(line_data, linewidth=0.5, color='red')
            plt.title("progress")
            plt.xlabel("epoch")
            plt.ylabel("reward")
            plt.grid(True, linestyle='--', alpha=1.0)
            plt.savefig("training_progress.png")
            plt.close()
            print(f"epoch: {episode + 1}, total reward: {total_reward}, steps survived: {steps}, epsilon: {epsilon}")
        if (episode + 1) % 5000 == 0:
            torch.save(ai, f"dqn/bestaizero{episode+1}.pt")

if __name__ == '__main__':
    train()
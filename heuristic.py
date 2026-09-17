from ai import Heuristic_CNN
from gamelogic import game
from piecesh import pieces as piecesh
import random
import math
import time
import numpy
import heapq
import torch
import torch.nn as nn
import torch.optim as optim

# MAKE SURE ALL THE PIECES ARE THE SAME THROUGHOUT ALL THE DIFFERENT BRANCHES SO THAT THE DICTIONARY WILL BE MORE HELPFUL, make sure you only set the pieces once for each depth where depth%3 == 0
# check if its possible at all for a branch to beat the best score
def dfs_score(new_game:game, max_depth, depth, pieces_pos, visited_states):
    if max_depth <= depth:
        return(new_game, new_game.score)
    if depth % 3 == 0:
        new_game.current_pieces = pieces_pos[depth // 3]
    pieces = new_game.get_pieces()
    valid_moves = []
    board_hash = new_game.board.tobytes()
    state_key = (board_hash, tuple([p[0] for p in pieces if p[0] != -1]), new_game.score)
    if state_key in visited_states:
        return visited_states[state_key]
    for p in range(3):
        if pieces[p][0] == -1:
            continue
        for x in range(8):
            for y in range(8):
                if new_game.can_place(pieces[p], x, y):
                    valid_moves.append((p,x,y))
    if not valid_moves:
        visited_states[state_key] = (None, -float("inf"))
        return (None, -float("inf"))
    best_child = None
    best_child_score = -float("inf")
    better_moves = []
    for move in valid_moves:
        c_game = new_game.deepcopy()
        c_game.one_game_turn_no_reset(move)
        h_score = 0
        if new_game.check_for_holes():
            h_score -= 7
        grid = c_game.board.reshape(8, 8)
        heights = numpy.sum(grid, axis=0)
        jaggedness = numpy.sum(numpy.abs(heights[1:] - heights[:-1]))
        h_score += .3 * jaggedness
        for x in range(8):
            for y in range(8):
                if c_game.can_place(piecesh.all_pieces[26], x, y):
                    h_score += 5
                    break
            else:
                continue
            break
        h_score += + 3*math.log(c_game.score - new_game.score)
        better_moves.append((c_game, move, h_score))
    for move_heu in heapq.nlargest(4 + max(0, (len(better_moves) - 10) // 10), better_moves, key=lambda x: x[2]):
        c_game = move_heu[0]
        move = move_heu[1]
        cur_best_game, cur_best_score = dfs_score(c_game, max_depth, depth + 1, pieces_pos, visited_states)
        if best_child_score <= cur_best_score:
            best_child = cur_best_game
            best_child_score = cur_best_score
    visited_states[state_key] = (best_child, best_child_score)
    return (best_child, best_child_score)

def dfs_cnn(new_game, board_tensor, info_tensor, game_list, visited_states, current_path=None):
    if current_path is None:
        current_path = []
    pieces = new_game.get_pieces()
    valid_moves = []
    board_hash = new_game.board.tobytes()
    available_pieces = tuple([p[0] for p in pieces if p[0] != -1])
    state_key = (board_hash, available_pieces, new_game.score)
    if state_key in visited_states:
        return board_tensor, info_tensor, game_list
    went = False
    for p in range(3):
        if pieces[p][0] == -1:
            continue
        went = True
        for x in range(8):
            for y in range(8):
                if new_game.can_place(pieces[p], y, x):
                    valid_moves.append((p,y,x))
    for move in valid_moves:
        c_game = new_game.deepcopy()
        c_game.one_game_turn_no_reset(move)
        c_game.move_history = current_path + [move]
        board_tensor, info_tensor, game_list = dfs_cnn(c_game, board_tensor, info_tensor, game_list, visited_states, current_path + [move])
    visited_states[state_key] = 1
    if not went:
        if not hasattr(new_game, 'move_history'):
            new_game.move_history = current_path
        cur_board = torch.tensor(new_game.board, dtype=torch.float32).view(-1, 1, 8, 8)
        other_info = torch.tensor([new_game.combo, new_game.combo_counter],dtype=torch.float32).unsqueeze(0)
        game_list = [new_game] + game_list
        board_tensor = torch.cat((cur_board, board_tensor), 0)
        info_tensor = torch.cat((other_info, info_tensor), 0)
        return board_tensor, info_tensor, game_list
    return board_tensor, info_tensor, game_list

def best_score(cur_game, rounds_played, extra_rounds):
    visited_states = {}
    pieces_possible = []
    for i in range(3 + rounds_played + math.ceil(extra_rounds / 3)):
        cur_game.current_pieces = cur_game.random_pieces()
        pieces_possible.append(cur_game.get_pieces())
    best_game, best_score = dfs_score(cur_game, rounds_played * 3 + extra_rounds, 0, pieces_possible, visited_states)
    return best_game

def best_cnn(cur_game, cnn):
    visited_states = {}
    board_tensor = torch.Tensor()
    info_tensor = torch.Tensor()
    game_list = []
    board_tensor, info_tensor, game_list = dfs_cnn(cur_game, board_tensor, info_tensor, game_list, visited_states)
    if len(game_list) > 0:
        result = cnn(board_tensor, info_tensor).flatten()
        max_index = torch.argmax(result).item()
        return game_list[max_index]
    else:
        return cur_game

def get_data(cur_game, cnn, turns):
    best_score_game = best_score(cur_game, 0, turns)
    # print(best_score_game)
    if best_score_game == None:
        return -2
    cnn_failed = 0
    for i in range(10):
        # print(i)
        # print(best_score_game)
        best_score_game.reset_pieces()
        total_game = best_cnn(best_score_game, cnn)
        if best_score_game != total_game:
            break
        cnn_failed += 1.0
    else:
        return math.log(best_score_game.score, 4) / 10
    cur_board = torch.tensor(total_game.board, dtype=torch.float32).view(-1, 1, 8, 8)
    other_info = torch.tensor([total_game.combo, total_game.combo_counter],dtype=torch.float32).unsqueeze(0)
    cnn_result = cnn(cur_board, other_info)
    return math.log(1 + best_score_game.score + .985 * cnn_result.item(), 4) - cnn_failed

def train_loop(cur_game, cnn, turns, games_per):
    total_score = 0
    for i in range(games_per):
        game_score = get_data(cur_game, cnn, turns)
        total_score += game_score
    return total_score / games_per

def choose_random_game(cur_game, depth = 2):
    c_game = cur_game.deepcopy()
    for i in range(depth):
        pieces = c_game.get_pieces()
        valid_moves = []
        for p in range(3):
            if pieces[p][0] == -1:
                continue
            for x in range(8):
                for y in range(8):
                    if c_game.can_place(pieces[p], y, x):
                        valid_moves.append((p, y, x))
        if not valid_moves:
            return None
        move = random.choice(valid_moves)
        c_game.one_game_turn_no_reset(move)
        c_game.reset_pieces()
    return c_game

def play_game(cnn, best_game):
    c_game = game()
    while True:
        c_game.reset_pieces()
        with torch.no_grad():
            next_game = best_cnn(c_game, cnn)
        if next_game == c_game:
            if c_game.score > best_game.score:
                best_game = c_game
            break
        c_game = next_game
    return c_game.score, best_game

def test_models(start, end, tests):
    for i in range(start, end):
        cnn = torch.load(f"model{i}.pt")
        total_score = 0
        best_game = game()
        for j in range(tests):
            score, best_game = play_game(cnn, best_game)
            total_score += score
        print(f"average score for model {i}: {total_score / tests}, best score: {best_game.score}")

if __name__ == "__main__":
    cur_game = game()
    cnn = torch.load("models/model249.pt")
    optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
    EPSILON_DECAY = 0.9999
    loss_fn = nn.SmoothL1Loss()
    epsilon = 0.09
    i = 101
    while True:
        if cur_game == None:
            cur_game = game()
        new_game = None
        with torch.no_grad():
            avg_score = train_loop(cur_game, cnn, 6, 36)
        cur_board = torch.tensor(cur_game.board, dtype=torch.float32).view(-1, 1, 8, 8)
        other_info = torch.tensor([cur_game.combo, cur_game.combo_counter],dtype=torch.float32).unsqueeze(0)
        cnn_result = cnn(cur_board, other_info).view(-1)
        loss = loss_fn(cnn_result, torch.Tensor([avg_score]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epsilon < random.random():
            for j in range(100):
                new_game = choose_random_game(cur_game, 3)
                if new_game != None:
                    break
            if not new_game:
                new_game = game()
        else:
            new_game = best_cnn(cur_game, cnn)
            if new_game == cur_game:
                new_game = game()
        if i % 10 == 0:
            if i % 100 == 0:
                torch.save(cnn, f"models/model{i // 100}.pt")
                print(f"saved model {i // 100}", end= " ")
                test_models(i//100, (i // 100) + 1, 30)
        cur_game = new_game
        epsilon = max(epsilon * EPSILON_DECAY, 0.09)
        i += 1
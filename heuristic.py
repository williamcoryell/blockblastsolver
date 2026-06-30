from ai import Heuristic_CNN
from gamelogic import game
from piecesh import pieces as piecesh
import torch
import random
import math
import time
import numpy
import pickle
import heapq

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
    for move_heu in heapq.nlargest(7 + (len(better_moves) - 10) // 9, better_moves, key=lambda x: x[2]):
        c_game = move_heu[0]
        move = move_heu[1]
        cur_best_game, cur_best_score = dfs_score(c_game, max_depth, depth + 1, pieces_pos, visited_states)
        if best_child_score <= cur_best_score:
            best_child = cur_best_game
            best_child_score = cur_best_score
    visited_states[state_key] = (best_child, best_child_score)
    return (best_child, best_child_score)

def dfs_cnn(new_game:game, board_tensor, info_tensor, game_list, visited_states):
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
    for move in (random.sample(valid_moves, k=192) if len(valid_moves) > 192 else valid_moves):
        c_game = new_game.deepcopy()
        c_game.one_game_turn_no_reset(move)
        board_tensor, info_tensor, game_list = dfs_cnn(c_game, board_tensor, info_tensor, game_list, visited_states)
    visited_states[state_key] = 1
    if not went:
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
    for i in range(1 + rounds_played + math.ceil(extra_rounds / 3)):
        cur_game.current_pieces = cur_game.random_pieces()
        pieces_possible.append(cur_game.get_pieces())
    best_game, best_score = dfs_score(cur_game, rounds_played * 3 + extra_rounds, 0, pieces_possible, visited_states)
    best_game.print_cur_state()
    return best_game

def best_cnn(cur_game, cnn):
    #you NEED to train with a discount factor, otherwise it will not be trained for survival.
    # use log() to normalize the scores as well, because the huge value spikes from the combos could cause a problem
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
#monte_carlo_cnn_test()
# monte_carlo_cnn_test(game(), Heuristic_CNN())
cur_game = game()
with open("pieces.pt", "rb") as f:
    cur_game.current_pieces = pickle.load(f)
starttime = time.time()
best_score(cur_game, 0, 6)
print(f"full time: {time.time() - starttime:.2f}")
    
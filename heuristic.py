from ai import Heuristic_CNN
from gamelogic import game
import torch
import random
import math
import time

# okay we want to look at the maximum possiblilty for score without testing a bunch of possibilities of different pieces being the three that pop up after round 1
# could potentially choose all 3 - 5 rounds of pieces before hand

def monte_carlo_score(new_game:game, max_depth, depth, pieces_pos, visited_states):
    if max_depth <= depth:
        return(new_game, new_game.score)
    new_game.reset_pieces()
    pieces = new_game.get_pieces()
    valid_moves = []
    board_hash = new_game.board.tobytes()
    available_pieces = tuple([p[0] for p in pieces if p[0] != -1])
    state_key = (board_hash, available_pieces, depth)
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
        return (new_game, -999999)
    best_child = None
    best_child_score = -float("inf")
    # for move in (random.sample(valid_moves, k=192) if len(valid_moves) > 192 else valid_moves):
    for move in valid_moves:
        c_game = new_game.deepcopy()
        c_game.one_game_turn_no_reset(move)
        cur_best_game, cur_best_score = monte_carlo_score(c_game, max_depth, depth + 1, pieces_pos, visited_states)
        if best_child_score <= cur_best_score:
            best_child = cur_best_game
            best_child_score = cur_best_score
    visited_states[state_key] = (best_child, best_child_score)
    return (best_child, best_child_score)

def monte_carlo_cnn(new_game:game, board_tensor, info_tensor, game_list, visited_states):
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
        board_tensor, info_tensor, game_list = monte_carlo_cnn(c_game, board_tensor, info_tensor, game_list, visited_states)
    visited_states[state_key] = 1
    if not went:
        cur_board = torch.tensor(new_game.board, dtype=torch.float32).view(-1, 1, 8, 8)
        other_info = torch.tensor([new_game.combo, new_game.combo_counter],dtype=torch.float32).unsqueeze(0)
        game_list = [new_game] + game_list
        board_tensor = torch.cat((cur_board, board_tensor), 0)
        info_tensor = torch.cat((other_info, info_tensor), 0)
        return board_tensor, info_tensor, game_list
    return board_tensor, info_tensor, game_list

def monte_carlo_score_test(cur_game, rounds_played, extra_rounds):
    visited_states = {}
    cur_game.print_cur_state()
    pieces_possible = []
    for i in range(1 + rounds_played + math.ceil(extra_rounds / 3)):
        cur_game.random_pieces()
        pieces_possible.append(cur_game.get_pieces())
    cur_game.random_pieces()
    best_game, best_score = monte_carlo_score(cur_game, rounds_played * 3 + extra_rounds, 0, pieces_possible, visited_states)
    return best_game

def monte_carlo_cnn_test(cur_game, cnn):
    #you NEED to train with a discount factor, otherwise it will not be trained for survival.
    # use log() to normalize the scores as well, because the huge value spikes from the combos could cause a problem
    for i in range(1000):
        visited_states = {}
        board_tensor = torch.Tensor()
        info_tensor = torch.Tensor()
        game_list = []
        board_tensor, info_tensor, game_list = monte_carlo_cnn(cur_game, board_tensor, info_tensor, game_list, visited_states)
        if len(game_list) > 0:
            result = cnn(board_tensor, info_tensor).flatten()
            max_index = torch.argmax(result).item()
            cur_game = game_list[max_index]
            cur_game.reset_pieces()
            cur_game.print_cur_state()
            print(f"turn {i}")
        else:
            print(game_list)
            print(f"ended at turn {i}, final score: {cur_game.score}")
            break
#monte_carlo_cnn_test()
monte_carlo_cnn_test(game(), Heuristic_CNN())


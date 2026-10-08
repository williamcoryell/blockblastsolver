from gamelogic import game
from piecesh import pieces as piecesh
import random
import math
import numpy
import heapq
import torch
import torch.nn as nn
import torch.optim as optim
from ai import Heuristic_CNN

def dfs_score(new_game, board_tensor, info_tensor, max_depth, depth, pieces_pos, game_list, visited_states, beam_k=8):
    if depth % 3 == 0:
        new_game.current_pieces = pieces_pos[depth // 3]
    pieces = new_game.get_pieces()
    board_hash = new_game.board
    available_pieces = tuple([p[0] for p in pieces if p[0] != -1])
    state_key = (board_hash, available_pieces, new_game.combo, new_game.combo_counter)
    if state_key in visited_states:
        return game_list, board_tensor, info_tensor
    valid_games = []
    if depth < max_depth - 1:
        for p in range(3):
            if pieces[p][0] == -1:
                continue
            w, h = piecesh.width_height_dict[pieces[p][0]]
            for x in range(8 - h + 1):
                for y in range(8 - w + 1):
                    if new_game.can_place(pieces[p], y, x):
                        move = (p,y,x)
                        c_game = new_game.deepcopy()
                        c_game.one_game_turn_no_reset(move)
                        h_score = 0
                        if c_game.check_for_holes():
                            h_score -= 7
                        grid = c_game.get_board().reshape(8, 8)
                        heights = numpy.sum(grid, axis=0)
                        jaggedness = numpy.sum(numpy.abs(heights[1:] - heights[:-1]))
                        h_score += .5 * jaggedness
                        for x in range(8):
                            for y in range(8):
                                if c_game.can_place(piecesh.all_pieces[26], x, y):
                                    h_score += 9
                                    break
                            else:
                                continue
                            break
                        h_score += + 3 * math.log(c_game.score - new_game.score)
                        valid_games.append((c_game, h_score))
        for moves in sorted(valid_games, key=lambda x: x[1], reverse=True)[:beam_k]:
            game_list, board_tensor, info_tensor = dfs_score(moves[0], board_tensor, info_tensor, max_depth, depth + 1,
                                    pieces_pos, game_list, visited_states)
    game_list = [new_game] + game_list
    cur_board = torch.tensor(new_game.get_board(), dtype=torch.float32).view(-1, 1, 8, 8)
    other_info = torch.tensor([new_game.combo, new_game.combo_counter],dtype=torch.float32).unsqueeze(0)
    board_tensor = torch.cat((cur_board, board_tensor), 0)
    info_tensor = torch.cat((other_info, info_tensor), 0)
    visited_states[state_key] = 1
    return game_list, board_tensor, info_tensor

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
        cur_board = torch.tensor(new_game.get_board(), dtype=torch.float32).view(-1, 1, 8, 8)
        other_info = torch.tensor([new_game.combo, new_game.combo_counter],dtype=torch.float32).unsqueeze(0)
        game_list = [new_game] + game_list
        board_tensor = torch.cat((cur_board, board_tensor), 0)
        info_tensor = torch.cat((other_info, info_tensor), 0)
        return board_tensor, info_tensor, game_list
    return board_tensor, info_tensor, game_list

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

def get_bellmen_score(cur_game, cnn, rounds_played, extra_turns):
    visited_states = {}
    board_tensor = torch.Tensor()
    info_tensor = torch.Tensor()
    game_list = []
    bellman_scores = []
    pieces_possible = []
    for i in range(3 + rounds_played + math.ceil(extra_turns / 3)):
        cur_game.current_pieces = cur_game.random_pieces()
        pieces_possible.append(cur_game.get_pieces())
    game_list, board_tensor, info_tensor = dfs_score(cur_game, board_tensor, info_tensor, 3 * rounds_played + extra_turns,
                                                     0, pieces_possible, game_list, visited_states)
    if len(game_list) > 0:
        result = cnn(board_tensor, info_tensor).flatten()
        for i in range(len(game_list)):
            bellman_scores.append((game_list[i].score - cur_game.score) + .99 * result[i].item())
        max_index = bellman_scores.index(max(bellman_scores))
        return bellman_scores[max_index]
    else:
        return -1

def training_loop(cur_game, cnn, rounds_played, extra_turns, games_per):
    total_score = 0
    for i in range(games_per):
        game_score = get_bellmen_score(cur_game, cnn, rounds_played, extra_turns)
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
        cnn = torch.load(f"models/model{i}.pt", weights_only=False)
        total_score = 0
        best_game = game()
        for j in range(tests):
            score, best_game = play_game(cnn, best_game)
            total_score += score
        print(f"average score for model {i}: {total_score / tests}, best score: {best_game.score}")

if __name__ == "__main__":
    cur_game = game()
    cnn = Heuristic_CNN()
    optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
    EPSILON_DECAY = 0.9999
    loss_fn = nn.SmoothL1Loss()
    epsilon = 0.2
    i = 0
    while True:
        if cur_game == None:
            cur_game = game()
        new_game = None
        with torch.no_grad():
            avg_score = training_loop(cur_game, cnn, 2, 0, 30)
        cur_board = torch.tensor(cur_game.get_board(), dtype=torch.float32).view(-1, 1, 8, 8)
        other_info = torch.tensor([cur_game.combo, cur_game.combo_counter],dtype=torch.float32).unsqueeze(0)
        cnn_result = cnn(cur_board, other_info).view(-1)
        loss = loss_fn(cnn_result, torch.Tensor([avg_score]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epsilon > random.random():
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
        epsilon = max(epsilon * EPSILON_DECAY, 0.05)
        i += 1
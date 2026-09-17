import torch

def run_dfs(new_game, board_tensor, info_tensor, game_list, visited_states, current_path=None):
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
                if new_game.run_can_place(pieces[p], y, x):
                    valid_moves.append((p,y,x))
    for move in valid_moves:
        c_game = new_game.deepcopy()
        c_game.run_one_game_turn_no_reset(move)
        c_game.move_history = current_path + [move]
        board_tensor, info_tensor, game_list = run_dfs(c_game, board_tensor, info_tensor, game_list, visited_states, current_path + [move])
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

def run_cnn(cur_game, cnn):
    visited_states = {}
    board_tensor = torch.Tensor()
    info_tensor = torch.Tensor()
    game_list = []
    board_tensor, info_tensor, game_list = run_dfs(cur_game, board_tensor, info_tensor, game_list, visited_states)
    if len(game_list) > 0:
        result = cnn(board_tensor, info_tensor).flatten()
        max_index = torch.argmax(result).item()
        return game_list[max_index]
    else:
        return cur_game
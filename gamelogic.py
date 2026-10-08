from random import randint
from piecesh import pieces
import numpy

def col_mask(x):
    mask = 0
    for y in range(8):
        mask |= 1 << (x + 8 * y)
    return mask
_ROW_MASKS = [0xFF << (8 * y) for y in range(8)]
_COL_MASKS = [col_mask(x) for x in range(8)]

class game:

    def __init__(self, board=-1, score=0, current_pieces=-1, combo=0, combo_counter=0):
        if isinstance(board, int):
            if board == -1:
                self.board = 0
            else:
                self.board = board & (1 << 64) - 1
        else:
            bits = 0
            for i, val in enumerate(board):
                if val:
                    bits |= 1 << i
            self.board = bits & (1 << 64) - 1
        self.score = score
        if type(current_pieces) == int:
            self.current_pieces = self.random_pieces()
        else:
            self.current_pieces = list(current_pieces)
        self.combo = combo
        self.combo_counter = combo_counter

    def _cell(self, board, i):
        return (board >> i) & 1

    def get_board(self):
        return numpy.array([(self.board >> i) & 1 for i in range(64)], dtype=int)

    def print_board(self):
        for i in range(8):
            for j in range(8):
                print(self._cell(self.board, i + j * 8), end="  ")
            print("\n")
        print("\n")

    def add_piece(self, piece, x=0, y=0):
        if not self.can_place(piece, x, y):
            return -1
        for x_m, y_m in pieces.cordinate_dict[piece[0]]:
            self.board |= 1 << (x + x_m + 8 * (y + y_m))
        return 4

    def can_place(self, piece, x, y):
        valid = False
        if x + pieces.width_height_dict[piece[0]][0] > 8 or y + pieces.width_height_dict[piece[0]][1] > 8:
            return 0
        for x_m, y_m in pieces.cordinate_dict[piece[0]]:
            if x_m + x >= 0 and y_m + y >= 0 and x_m + x < 8 and y_m + y < 8 and self._cell(self.board, x_m + x + 8 * (y_m + y)) != 1:
                x_a = x_m + x
                y_a = y_m + y
                for h in [-1, 0, 1]:
                    for v in [-1, 0, 1]:
                        if h == v or -h == v:
                            continue
                        x_n = x_a + h
                        y_n = y_a + v
                        if x_n < 0 or x_n >= 8 or y_n < 0 or y_n >= 8 or self._cell(self.board, x_n + 8 * y_n) == 1:
                            valid = True
                            break
                    else:
                        continue
                    break
            else:
                return 0
        if valid:
            return 1
        return 0

    def run_add_piece(self, piece, x=0, y=0):
        if not self.run_can_place(piece, x, y):
            return -1
        for x_m, y_m in pieces.cordinate_dict[piece[0]]:
            self.board |= 1 << (x + x_m + 8 * (y + y_m))
        return 4

    def run_can_place(self, piece, x, y):
        for x_m, y_m in pieces.cordinate_dict[piece[0]]:
            if x_m + x < 0 or y_m + y < 0 or x_m + x >= 8 or y_m + y >= 8 or self._cell(self.board, x_m + x + 8 * (y_m + y)) == 1:
                return 0
        return 1

    def run_place_block(self, piece, x, y):
        place_result = self.run_add_piece(piece, x, y)
        if place_result != -1:
            board_break = self.break_board()
            return board_break
        else:
            return -1

    def run_one_game_turn_no_reset(self, move):
        line_broke = 0
        if self.current_pieces[move[0]][0] == -1:
            return -1
        place_result = self.run_place_block(self.current_pieces[move[0]], move[1], move[2])
        if place_result == -1:
            return -2
        self.current_pieces[move[0]] = (-1, [[0]])
        if place_result >= 1:
            line_broke = 1
            self.combo += 1
            self.combo_counter = 3
            if self.board != 0:
                self.score += 300
            self.score += 5
            if self.combo == 1:
                self.score += 28 * place_result**2
            elif self.combo == 2:
                self.score += 82 * place_result**2
            elif self.combo == 3:
                self.score += 109 * place_result**2
            elif self.combo == 4:
                self.score += 139 * place_result**2
            else:
                self.score += 56 * self.combo * place_result**2
        else:
            self.score += 5
            if self.combo_counter == 1:
                self.combo = 0
                self.combo_counter -= 1
            elif self.combo_counter > 0:
                self.combo_counter -= 1
        return line_broke

    def break_board(self):
        full_rows = [y for y in range(8) if (self.board & _ROW_MASKS[y]) == _ROW_MASKS[y]]
        full_cols = [x for x in range(8) if (self.board & _COL_MASKS[x]) == _COL_MASKS[x]]
        for y in full_rows:
            self.board &= ~_ROW_MASKS[y]
        for x in full_cols:
            self.board &= ~_COL_MASKS[x]
        self.board &= (1 << 64) - 1
        return len(full_rows) + len(full_cols)

    def reset_board(self):
        self.board = 0

    def random_pieces(self):
        random_pieces = [pieces.all_pieces[randint(0, len(pieces.all_pieces) - 1)] for i in range(3)]
        return random_pieces

    def get_pieces(self):
        return self.current_pieces

    def place_block(self, piece, x, y):
        place_result = self.add_piece(piece, x, y)
        if place_result != -1:
            board_break = self.break_board()
            return board_break
        else:
            return -1

    def one_game_turn(self, move):
        line_broke = self.one_game_turn_no_reset(move)
        self.reset_pieces()
        return (self.current_pieces, self.get_board(), self.combo, self.combo_counter, self.score, line_broke)

    def one_game_turn_no_reset(self, move):
        line_broke = 0
        if self.current_pieces[move[0]][0] == -1:
            return -1
        place_result = self.place_block(self.current_pieces[move[0]], move[1], move[2])
        if place_result == -1:
            return -2
        self.current_pieces[move[0]] = (-1, [[0]])
        if place_result >= 1:
            line_broke = 1
            self.combo += 1
            self.combo_counter = 3
            if self.board != 0:
                self.score += 300
            self.score += 5
            if self.combo == 1:
                self.score += 28 * place_result**2
            elif self.combo == 2:
                self.score += 82 * place_result**2
            elif self.combo == 3:
                self.score += 109 * place_result**2
            elif self.combo == 4:
                self.score += 139 * place_result**2
            else:
                self.score += 56 * self.combo * place_result**2
        else:
            self.score += 5
            if self.combo_counter == 1:
                self.combo = 0
                self.combo_counter -= 1
            elif self.combo_counter > 0:
                self.combo_counter -= 1
        return line_broke

    def reset_pieces(self):
        if all(item[0] == -1 for item in self.current_pieces):
            self.current_pieces = self.random_pieces()

    def get_state(self):
        return (self.current_pieces, self.get_board(), self.combo, self.combo_counter, self.score)

    def print_cur_state(self):
        self.print_board()
        for i in range(3):
            pieces.print_piece(self.current_pieces[i], i + 1)
        print(f"Score: {self.score}, Combo: {self.combo}, Combo Countdown: {self.combo_counter}")

    def reset(self):
        self.reset_board()
        self.score = 0
        self.combo = 0
        self.combo_counter = 0
        self.current_pieces = self.random_pieces()
        return (self.current_pieces, self.get_board(), self.combo, self.combo_counter, self.score)

    def deepcopy(self):
        return game(self.board, self.score, self.current_pieces, self.combo, self.combo_counter)

    def check_for_holes(self):
        for x_a in range(8):
            for y_a in range(8):
                if self._cell(self.board, x_a + 8 * y_a) == 1:
                    continue
                counter = 0
                for h in [-1, 0, 1]:
                    for v in [-1, 0, 1]:
                        if h == v or -h == v:
                            continue
                        if x_a + h < 0 or y_a + v < 0 or x_a + h >= 8 or y_a + v >= 8 or self._cell(self.board, x_a + h + 8 * (y_a + v)) == 1:
                            counter += 1
                if counter == 4:
                    return 1
        return 0
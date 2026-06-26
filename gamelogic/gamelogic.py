from random import randint
from piecesh import pieces
import numpy
class game:
    
    def __init__(self, board=-1, score=0, current_pieces=-1, combo=0, combo_counter=0):
        if type(board) == int:
            self.board = numpy.array([numpy.array([0 for i in range(8)]) for j in range(8)])
        else:
            self.board = numpy.copy(board)
        self.score = score
        if type(current_pieces) == int:
            self.current_pieces = self.random_pieces()
        else:
            self.current_pieces = list(current_pieces)
        self.combo = combo
        self.combo_counter = combo_counter

    def get_board(self):
        return self.board

    def print_board(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                print(self.board[i][j], end="  ")
            print("\n")
        print("\n")

    def add_piece(self, piece, x=0, y=0):
        if not self.can_place(piece, x, y):
            return -1
        for x_m, y_m in pieces.cordinate_dict[piece[0]]:
            self.board[x + x_m][y + y_m] = 1
        return 4

    def can_place(self, piece, x, y):
        valid = False
        for x_m, y_m in pieces.cordinate_dict[piece[0]]:
            if x_m + x >= 0 and y_m + y >= 0 and x_m + x < len(self.board) and y_m + y < len(self.board[x_m + x]) and self.board[x_m + x][y_m + y] != 1:
                x_a = x_m + x
                y_a = y_m + y
                for h in [-1, 0, 1]:
                    for v in [-1, 0, 1]:
                        if h == v or -h == v:
                            continue
                        if x_a + h < 0 or y_a + v < 0 or x_a + h >= len(self.board) or y_a + v >= len(self.board[x_a]) or self.board[x_a + h][y_a + v] == 1:
                            valid = True
            else:
                return 0
        if valid:
            return 1
        return 0

    def break_board(self):
        hor_lst = [i for i, r in enumerate(self.board) if 0 not in r]
        vert_lst = [j for j in range(8) if all(self.board[i][j] != 0 for i in range(8))]
        for i in hor_lst:
            self.board[i] = numpy.array([0] * 8)
        for j in vert_lst:
            for i in range(8):
                self.board[i][j] = 0
        return len(hor_lst) + len(vert_lst)
    
    def reset_board(self):
        self.board = numpy.copy([[0 for i in range(8)] for j in range(8)])

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
        return (self.current_pieces, self.board, self.combo, self.combo_counter, self.score, line_broke)
    
    def one_game_turn_no_reset(self, move):
        score_mult = 1
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
            if self.combo == 1:
                self.score += 28 * place_result * score_mult
            elif self.combo == 2:
                self.score += 82 * place_result * score_mult
            elif self.combo == 3:
                self.score += 109 * place_result * score_mult
            elif self.combo == 4:
                self.score += 139 * place_result * score_mult
            else:
                self.score += 56 * self.combo * place_result * score_mult
        elif self.combo_counter == 1:
            self.combo = 0
            self.combo_counter -= 1
        elif place_result == 0:
            self.score += 4
        elif self.combo_counter > 0:
            self.combo_counter -= 1
        return line_broke

    def reset_pieces(self):
        if all(item[0] == -1 for item in self.current_pieces):
            self.current_pieces = self.random_pieces()
    
    def get_state(self):
        return (self.current_pieces, self.board, self.combo, self.combo_counter, self.score)
    
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
        return (self.current_pieces, self.board, self.combo, self.combo_counter, self.score)
    
    def deepcopy(self):
        return game(self.board, self.score, self.current_pieces, self.combo, self.combo_counter)
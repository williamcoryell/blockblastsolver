class block_board:
    
    def __init__(self):
        self.board = [[0 for i in range(8)] for j in range(8)]

    def get_board(self):
        return self.board

    def print_board(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                print(self.board[i][j], end="  ")
            print("\n")
        print("\n")

    def add_piece(self, piece, y=0, x=0):
        for i in range(len(piece)):
            for j in range(len(piece[i])):
                if piece[i][j] == 0:
                    continue
                if i + x < 0 or j + y < 0 or i + x >= len(self.board) or j + y >= len(self.board[i]) or self.board[i + x][j + y] == 1:
                    return -1
        for i in range(len(piece)):
            for j in range(len(piece[i])):
                if piece[i][j] == 0:
                    continue
                self.board[i + x][j + y] += piece[i][j]
        return 4

    def can_place_piece(self, piece, y, x):
        for i in range(len(piece)):
            for j in range(len(piece[i])):
                if piece[i][j] == 0:
                    continue
                if i + x < 0 or j + y < 0 or i + x >= len(self.board) or j + y >= len(self.board[i]) or self.board[i + x][j + y] == 1:
                    return -1
        return 1

    def break_board(self):
        hor_lst = []
        for i in range(len(self.board)):
            full_row = True
            for j in range(len(self.board)):
                if self.board[i][j] == 0:
                    full_row = False
                    break
            if full_row:
                hor_lst.append(i)
        vert_lst = []
        for j in range(len(self.board)):
            full_row = True
            for i in range(len(self.board)):
                if self.board[i][j] == 0:
                    full_row = False
                    break
            if full_row:
                vert_lst.append(j)
        for i in hor_lst:
            for j in range(len(self.board)):
                self.board[i][j] = 0
        for j in vert_lst:
            for i in range(len(self.board)):
                self.board[i][j] = 0
        return len(hor_lst) + len(vert_lst)
    
    def reset_board(self):
        self.board = [[0 for i in range(8)] for j in range(8)]
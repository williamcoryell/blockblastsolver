class pieces:
    #L's
    left_down_l = [
        [1],
        [1],
        [1, 1],
    ]
    right_down_l = [
        [0, 1],
        [0, 1],
        [1, 1]
    ]
    left_up_l = [
        [1, 1],
        [1],
        [1],
    ]
    right_up_l = [
        [1, 1],
        [0, 1],
        [0, 1]
    ]
    #long L's
    long_left_down_l = [
        [1],
        [1, 1, 1]
    ]
    long_right_down_l = [
        [0, 0, 1],
        [1, 1, 1]
    ]
    long_left_up_l = [
        [1, 1, 1],
        [1]
    ]
    long_right_up_l = [
        [1, 1, 1],
        [0, 0, 1]
    ]
    #Sticks
    one = [
        [1]
    ]
    #   Rights
    right_2 = [
        [1, 1]
    ]
    right_3 = [
        [1, 1, 1]
    ]
    right_4 = [
        [1, 1, 1, 1]
    ]
    right_5 = [
        [1, 1, 1, 1, 1]
    ]
    #   Downs
    down_2 = [
        [1],
        [1]
    ]
    down_3 = [
        [1],
        [1],
        [1]
    ]
    down_4 = [
        [1],
        [1],
        [1],
        [1]
    ]
    down_5 = [
        [1],
        [1],
        [1],
        [1],
        [1]
    ]
    #Corners
    #   Large Corners
    left_up_xl_corner = [
        [1, 1, 1],
        [1],
        [1]
    ]
    left_down_xl_corner = [
        [1],
        [1],
        [1, 1, 1]
    ]
    right_up_xl_corner = [
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1]
    ]
    right_down_xl_corner = [
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    #   Small Corners
    right_down_xs_corner = [
        [0, 1],
        [1, 1]
    ]
    right_up_xs_corner = [
        [1, 1],
        [0, 1]
    ]
    left_down_xs_corner = [
        [1],
        [1, 1]
    ]
    left_up_xs_corner = [
        [1, 1],
        [1]
    ]
    #Squares
    square_2x2 = [
        [1, 1],
        [1, 1]
    ]
    square_3x3 = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    square_2x3 = [
        [1, 1],
        [1, 1],
        [1, 1]
    ]
    square_3x2 = [
        [1, 1, 1],
        [1, 1, 1]
    ]
    #S's
    s_0 = [
        [1],
        [1, 1],
        [0, 1]
    ]
    s_90 = [
        [0, 1, 1],
        [1, 1]
    ]
    #   Flip
    s_0_flip = [
        [0, 1],
        [1, 1],
        [1]
    ]
    s_90_flip = [
        [1, 1],
        [0, 1, 1]
    ]
    #T's
    t_down = [
        [1, 1, 1],
        [0, 1]
    ]
    t_up = [
        [0, 1],
        [1, 1, 1]
    ]
    t_right = [
        [1],
        [1, 1],
        [1]
    ]
    t_left = [
        [0, 1],
        [1, 1],
        [0, 1]
    ]
    #Steps
    right_stairs = [
        [0, 0, 1],
        [0, 1],
        [1]
    ]
    left_stairs = [
        [1],
        [0, 1],
        [0, 0, 1]
    ]
    all_pieces_no_keys = [
        left_down_l, right_down_l, left_up_l, right_up_l,
        long_left_down_l, long_right_down_l, long_left_up_l, long_right_up_l,
        one, right_2, right_3, right_4, right_5, down_2, down_3, down_4, down_5,
        left_up_xl_corner, left_down_xl_corner, right_up_xl_corner, right_down_xl_corner,
        right_down_xs_corner, right_up_xs_corner, left_down_xs_corner, left_up_xs_corner,
        square_2x2, square_3x3, square_2x3, square_3x2,
        s_0, s_90, s_0_flip, s_90_flip,
        t_down, t_up, t_right, t_left,
        right_stairs, left_stairs
    ]
    
    all_pieces = list(enumerate(all_pieces_no_keys))

    cordinate_dict = {piece[0] : [(i, j) for i in range(len(piece[1])) for j in range(len(piece[1][i])) if piece[1][i][j] == 1] for piece in all_pieces}

    def print_piece(piece, num=0):
        if type(piece) == int:
            return
        if num != 0:
            print(f"Piece {num}:")
        for i in range(len(piece[1])):
            for j in range(len(piece[1][i])):
                print(piece[1][i][j], end=" ")
            print()

import pygame
import numpy as np
import torch
import sys
import copy
from gamelogic import game
from ai import Heuristic_CNN
from heuristic import best_cnn
from rungame import run_cnn
from piecesh import pieces

pygame.init()
WIDTH, HEIGHT = 1280, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Block Blast AI Solver")
font = pygame.font.SysFont("Arial", 20)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (100, 150, 255)
GREEN = (100, 255, 100)
RED = (255, 100, 100)
DARK_GRAY = (50, 50, 50)

MAIN_CELL_SIZE = 50
PIECE_CELL_SIZE = 30
MINI_CELL_SIZE = 20

main_board = np.zeros((8, 8), dtype=int)
custom_pieces_grids = [np.zeros((5, 5), dtype=int) for _ in range(3)]
best_board_state = None
step_boards = []
is_evaluating = False

current_combo = 0
current_combo_counter = 0

print("Which model?")
model_number = input()
cnn_model = Heuristic_CNN()
try:
    cnn_model = torch.load(f"models/model{model_number}.pt")
    cnn_model.eval()
except Exception:
    print("No trained model found. Using untrained Heuristic_CNN.")

def draw_grid(surface, grid, x_offset, y_offset, cell_size, active_color=BLUE):
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            rect = pygame.Rect(x_offset + c * cell_size, y_offset + r * cell_size, cell_size, cell_size)
            if grid[r, c] == 1:
                pygame.draw.rect(surface, active_color, rect)
            pygame.draw.rect(surface, GRAY, rect, 1)

def handle_click(mouse_pos, grid, x_offset, y_offset, cell_size):
    rows, cols = grid.shape
    x, y = mouse_pos
    if x_offset <= x < x_offset + cols * cell_size and y_offset <= y < y_offset + rows * cell_size:
        c = (x - x_offset) // cell_size
        r = (y - y_offset) // cell_size
        grid[r, c] = 1 - grid[r, c]
        return True
    return False

def extract_custom_piece(grid, dynamic_id):
    rows = np.any(grid, axis=1)
    cols = np.any(grid, axis=0)
    if not np.any(rows):
        return (-1, [[0]])
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    trimmed = grid[rmin:rmax+1, cmin:cmax+1].T.tolist()
    piece_tuple = (dynamic_id, trimmed)
    
    pieces.all_pieces.append(piece_tuple)
    pieces.cordinate_dict[dynamic_id] = [(i, j) for i in range(len(trimmed)) for j in range(len(trimmed[i])) if trimmed[i][j] == 1]
    pieces.width_height_dict[dynamic_id] = (max(map(len, trimmed)), len(trimmed))
    
    return piece_tuple

def draw_counter_ui(surface, label, value, x_pos, y_pos):
    """Draws a label, a value, and +/- buttons, returning the button rects for click detection."""
    surface.blit(font.render(f"{label}: {value}", True, BLACK), (x_pos, y_pos))
    minus_rect = pygame.Rect(x_pos, y_pos + 25, 30, 30)
    plus_rect = pygame.Rect(x_pos + 40, y_pos + 25, 30, 30)
    
    pygame.draw.rect(surface, GRAY, minus_rect)
    pygame.draw.rect(surface, GRAY, plus_rect)
    
    surface.blit(font.render("-", True, BLACK), (minus_rect.x + 10, minus_rect.y + 4))
    surface.blit(font.render("+", True, BLACK), (plus_rect.x + 8, plus_rect.y + 4))
    
    return minus_rect, plus_rect

def evaluate_best_move():
    global best_board_state, step_boards, main_board, current_combo, current_combo_counter
    
    flat_board = main_board.flatten()
    g = game(board=flat_board)
    
    g.combo = current_combo
    g.combo_counter = current_combo_counter
    
    current_pieces = []
    base_id = 1000
    for i, grid in enumerate(custom_pieces_grids):
        current_pieces.append(extract_custom_piece(grid, base_id + i))
    g.current_pieces = current_pieces
    
    with torch.no_grad():
        best_g = best_cnn(g, cnn_model)
        if best_g == g:
            best_g = run_cnn(g, cnn_model)
            if best_g == g:
                print("unsolveable")
    
    best_board_state = best_g.board.reshape((8, 8))
    
    step_boards = []
    temp_g = game(board=flat_board)
    temp_g.current_pieces = copy.deepcopy(current_pieces)
    step_boards.append(temp_g.board.reshape((8, 8)).copy())
    
    if hasattr(best_g, 'move_history'):
        for move in best_g.move_history:
            temp_g.run_one_game_turn_no_reset(move)
            step_boards.append(temp_g.board.reshape((8, 8)).copy())

running = True
btn_rect = pygame.Rect(460, 50, 180, 50) 
result_board_rect = pygame.Rect(50, 500, 8 * MAIN_CELL_SIZE, 8 * MAIN_CELL_SIZE)

combo_minus_btn, combo_plus_btn = None, None
counter_minus_btn, counter_plus_btn = None, None

while running:
    screen.fill(WHITE)
    
    draw_grid(screen, main_board, 50, 50, MAIN_CELL_SIZE)
    
    for i in range(3):
        y_off = 50 + i * (5 * PIECE_CELL_SIZE + 20)
        draw_grid(screen, custom_pieces_grids[i], 650, y_off, PIECE_CELL_SIZE, GREEN)
        
    pygame.draw.rect(screen, BLUE, btn_rect)
    
    combo_minus_btn, combo_plus_btn = draw_counter_ui(screen, "Combo", current_combo, 460, 120)
    counter_minus_btn, counter_plus_btn = draw_counter_ui(screen, "Combo Counter", current_combo_counter, 460, 190)
    
    if best_board_state is not None:
        draw_grid(screen, best_board_state, 50, 500, MAIN_CELL_SIZE, RED)
        
        for step_idx, step_b in enumerate(step_boards):
            x_step = 550 + (step_idx % 2) * (8 * MINI_CELL_SIZE + 20)
            y_step = 590 + (step_idx // 2) * (8 * MINI_CELL_SIZE + 30)
            draw_grid(screen, step_b, x_step, y_step, MINI_CELL_SIZE, DARK_GRAY)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            
            if combo_minus_btn and combo_minus_btn.collidepoint(pos):
                current_combo = max(0, current_combo - 1)
            elif combo_plus_btn and combo_plus_btn.collidepoint(pos):
                current_combo += 1
                
            if counter_minus_btn and counter_minus_btn.collidepoint(pos):
                current_combo_counter = max(0, current_combo_counter - 1)
            elif counter_plus_btn and counter_plus_btn.collidepoint(pos):
                current_combo_counter += 1
            
            handle_click(pos, main_board, 50, 50, MAIN_CELL_SIZE)
            
            for i in range(3):
                y_off = 50 + i * (5 * PIECE_CELL_SIZE + 20)
                handle_click(pos, custom_pieces_grids[i], 650, y_off, PIECE_CELL_SIZE)
            
            if btn_rect.collidepoint(pos):
                evaluate_best_move()
                
            if best_board_state is not None:
                x_res, y_res = 50, 500
                if x_res <= pos[0] < x_res + 8 * MAIN_CELL_SIZE and y_res <= pos[1] < y_res + 8 * MAIN_CELL_SIZE:
                    main_board = best_board_state.copy()
                    best_board_state = None
                    step_boards = []

pygame.quit()
sys.exit()
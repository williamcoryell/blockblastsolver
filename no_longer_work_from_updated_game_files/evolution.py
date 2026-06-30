from board import block_board
from gamelogic import game
from ai import BlockBlastAI
import copy
import torch
import heapq
import matplotlib.pyplot as plt
from multiprocessing import Pool

trails = 10

def eval(ai, c_game):
    while True:
        result = c_game.get_state()
        o_piece_1 = result[0][0][1]
        o_piece_2 = result[0][1][1]
        o_piece_3 = result[0][2][1]
        piece_1 = [i for j in o_piece_1 for i in j]
        piece_2 = [i for j in o_piece_2 for i in j]
        piece_3 = [i for j in o_piece_3 for i in j]
        board_2d = result[1]
        board = [i for j in board_2d.get_board() for i in j]
        combo = result[2]
        combo_counter = result[3]
        score = result[4]
        data_list = piece_1 + piece_2 + piece_3 + board + [combo] + [combo_counter]
        prediction = ai(data_list)
        # if result[0][0][0] == -1:
        #     for k in range(0, 64): 
        #         prediction[k] = -9999999
        # if result[0][1][0] == -1:
        #     for k in range(64, 128): 
        #         prediction[k] = -9999999
        # if result[0][2][0] == -1:
        #     for k in range(128, 192): 
        #         prediction[k] = -9999999
        index = prediction.index(max(prediction))
        piece_index = index // 64
        remainder = index % 64
        x = remainder // 8
        y = remainder % 8
        result = c_game.one_game_turn([piece_index, x, y])
        if type(result) == int:
            row_sums = [sum(board[r*8 : (r+1)*8]) for r in range(8)]
            col_sums = [sum(board[r*8 + c] for r in range(8)) for c in range(8)]
            closeness_bonus = sum((x ** 2) for x in row_sums) + sum((y ** 2) for y in col_sums)
            final_fitness = score + closeness_bonus * 0.08
            return final_fitness # make a metric on how close it was to breaking a line, cause it just isn't breaking shit y
        # else:
        #     if result[5]:
        #         sum_of_combo += 100 * result[5]

def eval_single(ai):
    score = 0
    cur_game = game(block_board())
    for g in range(trails):
        score += eval(ai, cur_game)
    return score

if __name__ == "__main__":
    pop_size = 1000
    population = [BlockBlastAI() for i in range(pop_size)]
    # population = [torch.load("best_ai.pt") for i in range(pop_size)]
    carry_overs = 100
    gen_history = []
    i = 0
    while True:
        scores = []
        with Pool(processes=6) as pool:
            scores = pool.map(eval_single, population)
        best_indexes = heapq.nlargest(carry_overs, range(len(scores)), key=scores.__getitem__)
        best_ais = [population[j] for j in best_indexes]
        top_10_scores = [scores[idx] for idx in best_indexes[:carry_overs]]
        gen_history.append(top_10_scores)
        if i % 5 == 4:
            print(f"generation {i + 1}, best scores: ", end="")
            for m in range(carry_overs - 1):
                print(f"{scores[best_indexes[m]] // trails}", end=", ")
            print(f"{scores[best_indexes[carry_overs - 1]] // trails}")
        if i % 25 == 24:
            plt.figure(figsize=(12, 7))
            rank_lines = list(zip(*(gen_history)))
            for rank, line_data in enumerate(rank_lines):
                if rank == 0:
                    plt.plot(line_data, label="rank 1", linewidth=0.5, color='red')
                else:
                    plt.plot(line_data, label=f"rank {rank+1}", linewidth=0.25, alpha=0.4)
            plt.title("top ten")
            plt.xlabel("generations")
            plt.ylabel("fitness Score")
            plt.legend(loc="upper left")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig("training_progress.png")
            plt.close()
        if i % 100 == 99:
            torch.save(best_ais[0], f"best_ai{i + 1}.pt")
        new_population = []
        for j in range(carry_overs):
            new_population.append(copy.deepcopy(best_ais[j]))
        for j in range(pop_size - carry_overs):
            new_population.append(best_ais[j % carry_overs].mutate_model())
        population = new_population
        i += 1
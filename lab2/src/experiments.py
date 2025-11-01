from two_player_games.games.dots_and_boxes import DotsAndBoxes, Player
from alphabeta import best_move
import numpy as np
import time
import random


GAMES_IN_ONE_CYCLE = 20
MAX_PLAYER = str('S')
MIN_PLAYER = str('L')
DRAW_SITUATION = str('N')

if __name__ == "__main__":
    random.seed('1234')
    sizes = [3, 4, 5, 6]
    depths = [0, 1]
    player_order = [('S', 'L'), ('L', 'S')]
    times_table = np.zeros(shape=(len(sizes), len(player_order), len(depths), GAMES_IN_ONE_CYCLE))
    games_outcomes = np.ndarray(shape=(len(sizes), len(player_order), len(depths), GAMES_IN_ONE_CYCLE), dtype='U1')

    size_idx = 0
    for size in sizes:
        order_idx = 0
        for order in player_order:
            depth_idx = 0
            for depth in depths:
                for game_num in range(GAMES_IN_ONE_CYCLE):
                    game = DotsAndBoxes(size=size, first_player=Player(order[0]), second_player=Player(order[1]))
                    i = 0
                    start = time.time()
                    while not game.is_finished():
                        i += 1
                        if game.state.get_current_player().char == MAX_PLAYER:
                            best_curr_move = best_move(game.state, depth, MAX_PLAYER)
                        else:
                            best_curr_move = best_move(game.state, 0, MIN_PLAYER)

                        game.make_move(best_curr_move)
                        print(f"After {i} turn:")
                        print(game.state)
                        print(" ")
                    end = time.time()
                    times_table[size_idx, order_idx, depth_idx, game_num] = end - start
                    winner = game.get_winner()
                    if winner is None:
                        print('Draw!')
                        games_outcomes[size_idx, order_idx, depth_idx, game_num] = DRAW_SITUATION
                    else:
                        print('Winner: Player ' + winner.char)
                        games_outcomes[size_idx, order_idx, depth_idx, game_num] = winner.char

                depth_idx += 1
            order_idx += 1
        size_idx += 1

    print("Done experimenting")



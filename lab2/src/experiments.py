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
    sizes = [3, 4, 5]
    depths = [3]
    player_order = [('S', 'L'), ('L', 'S')]
    times_table = np.zeros(shape=(len(sizes), len(player_order), len(depths), GAMES_IN_ONE_CYCLE))
    games_outcomes = np.ndarray(shape=(len(sizes), len(player_order), len(depths)), dtype=dict)

    size_idx = 0
    for size in sizes:
        order_idx = 0
        for order in player_order:
            depth_idx = 0
            for depth in depths:
                game_scores = {
                    MAX_PLAYER: 0,
                    MIN_PLAYER: 0,
                    DRAW_SITUATION: 0
                }
                for game_num in range(GAMES_IN_ONE_CYCLE):
                    game = DotsAndBoxes(size=size, first_player=Player(order[0]), second_player=Player(order[1]))
                    i = 0
                    start = time.time()
                    while not game.is_finished():
                        i += 1
                        if game.state.get_current_player().char == MAX_PLAYER:
                            best_curr_move = best_move(game.state, depth, MAX_PLAYER)
                        else:
                            best_curr_move = best_move(game.state, 1, MIN_PLAYER)

                        game.make_move(best_curr_move)
                        print(f"After {i} turn:")
                        print(game.state)
                        print(" ")
                    end = time.time()
                    times_table[size_idx, order_idx, depth_idx, game_num] = end - start
                    winner = game.get_winner()
                    if winner is None:
                        print('Draw!')
                        game_scores[DRAW_SITUATION] += 1
                    else:
                        print('Winner: Player ' + winner.char)
                        game_scores[winner.char] += 1

                games_outcomes[size_idx, order_idx, depth_idx] = game_scores
                depth_idx += 1
            order_idx += 1
        size_idx += 1

    print('\n')
    print("Done experimenting.")
    print('\n')

    for size_idx in range(len(sizes)):
        for depth_idx in range(len(depths)):
            for order_idx in range(len(player_order)):
                time_mean = np.mean(times_table[size_idx, order_idx, depth_idx])
                num_of_wins = games_outcomes[size_idx, order_idx, depth_idx][MAX_PLAYER]
                num_of_losses = games_outcomes[size_idx, order_idx, depth_idx][MIN_PLAYER]
                num_of_draws = games_outcomes[size_idx, order_idx, depth_idx][DRAW_SITUATION]
                print(f"Size: {sizes[size_idx]}, Order: {player_order[order_idx]}, Depth: {depths[depth_idx]} -->")
                print(f"Mean of games times: {time_mean} ; Wins: {num_of_wins} ; Losses: {num_of_losses} ; Draws: {num_of_draws}\n")

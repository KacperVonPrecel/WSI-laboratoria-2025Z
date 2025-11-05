from alphabeta import best_move
from two_player_games.games import dots_and_boxes
from typing import Any, Callable, Dict, Tuple, List


def game_interface(player_name, size, difficulty):
    pass


if __name__ == "__main__":
    game = dots_and_boxes.DotsAndBoxes(size=3)
    i = 0

    while not game.is_finished():
        i += 1
        if game.state.get_current_player().char == '1':
            best_curr_move = best_move(game.state, 3, '1')
        else:
            best_curr_move = best_move(game.state, 2, '2')

        game.make_move(best_curr_move)
        print(f"After {i} turn:")
        print(game.state)
        print(" ")

    winner = game.get_winner()
    if winner is None:
        print('Draw!')
    else:
        print('Winner: Player ' + winner.char)

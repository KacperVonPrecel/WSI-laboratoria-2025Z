from alphabeta import best_move
from two_player_games.games.dots_and_boxes import DotsAndBoxes, Player
from typing import Any, Callable, Dict, Tuple, List
from random import randint


EASY_DEPTH = 1
NORMAL_DEPTH = 2
HARD_DEPTH = 4


def game_interface(player_name: str, size: int, difficulty: str, is_player_first: bool):
    game = DotsAndBoxes(size, Player(player_name), Player("C")) if is_player_first else DotsAndBoxes(size, Player("C"), Player(player_name))
    match difficulty:
        case "easy":
            cp_depth = EASY_DEPTH
        case "normal":
            cp_depth = NORMAL_DEPTH
        case "hard":
            cp_depth = HARD_DEPTH
        case _:
            cp_depth = 0
    i = 0

    while not game.is_finished():

        print(f"After {i} turn:")
        print(game.state)
        print(" ")
        if game.get_current_player().char == player_name:
            print("Your turn. Here's the list of possible moves (h - horizontal, v - vertical):")
            j = 0
            for move in game.state.get_moves():
                print(f"{j}: {move.connection}: {move.loc}")
                j += 1
            good_idx = False
            move_pool = game.state.get_moves()
            while not good_idx:
                print("What move will you choose? (write the index of wanted move):")
                chosen_idx = int(input())
                if chosen_idx >= 0 and chosen_idx < len(game.state.get_moves()):
                    good_idx = True
                else:
                    print("Ups! Wrong index. Try again.")
            game.make_move(move_pool[chosen_idx])

        else:
            best_curr_move = best_move(game.state, cp_depth, 'C')
            game.make_move(best_curr_move)
        i += 1

    print(f"After {i} turn:")
    print(game.state)
    print(" ")
    winner = game.get_winner()
    if winner is None:
        print('Draw!')
    else:
        print('Winner: Player ' + winner.char)


if __name__ == "__main__":
    print("Welcome to the game of dots and boxes!\nPlease enter your name! (one character):")
    name = str(input())
    print("Now enter the size of the board you want to play on. (one number):")
    size = int(input())
    print("Finally choose a difficulty of the Computer! (easy, normal, hard):")
    difficulty = str(input())
    print("Let's choose who starts first!")
    if randint(0, 1) == 1:
        is_first = True
        print(f"Player {name} is first!")
    else:
        is_first = False
        print("Computer C is first!")
    print("Let the game begin!\n")
    game_interface(name, size, difficulty, is_first)

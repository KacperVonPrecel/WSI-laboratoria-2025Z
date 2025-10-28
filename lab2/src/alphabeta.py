import numpy as np
from two_player_games.games import dots_and_boxes
from typing import Any, Callable, Dict, Tuple
import random


def heuristics(state: dots_and_boxes.DotsAndBoxesState) -> int:

    player_max = state.get_players()[0]
    player_min = state.get_players()[1]
    current_score = state.get_scores()

    value = current_score[player_max] - current_score[player_min]

    return value


def alphabeta(
        game_state: dots_and_boxes.DotsAndBoxesState,
        depth: int,
        alfa: int = float('-inf'),
        beta: int = float('inf'),
        maximizing_player: str = "M"):

    if depth == 0 or game_state.is_finished():
        return heuristics(game_state)

    if game_state.get_current_player().char == maximizing_player:
        value = float('-inf')
        for move in game_state.get_moves():
            value = max(value, alphabeta(game_state.make_move(move), depth - 1, alfa, beta, maximizing_player))
            alfa = max(alfa, value)
            if value >= beta:
                break
        return value
    else:
        value = float('inf')
        for move in game_state.get_moves():
            value = min(value, alphabeta(game_state.make_move(move), depth - 1, alfa, beta, maximizing_player))
            beta = min(beta, value)
            if value <= alfa:
                break
        return value


def best_move(state: dots_and_boxes.DotsAndBoxesState, depth: int, player: str) -> dots_and_boxes.DotsAndBoxesMove:
    best_value = float("-inf")
    best_move = None
    idx = 0
    best_idx = 0
    for move in state.get_moves():
        value = alphabeta(game_state=state.make_move(move), depth=depth, maximizing_player=player)

        if value > best_value:
            best_value = value
            best_move = move
            best_idx = idx
        idx += 1

    print(f"Best move index: {best_idx}")
    print(f"Best value: {best_value}")
    return best_move


if __name__ == "__main__":
    game = dots_and_boxes.DotsAndBoxes()
    i = 0

while not game.is_finished():
    i += 1
    if game.state.get_current_player().char == '1':
        best_curr_move = best_move(game.state, 3, '1')
    else:
        best_curr_move = best_move(game.state, 1, '2')

    game.make_move(best_curr_move)
    print(f"After {i} turn:")
    print(game.state)
    print(" ")

winner = game.get_winner()
if winner is None:
    print('Draw!')
else:
    print('Winner: Player ' + winner.char)
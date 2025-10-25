import numpy as np
from two_player_games.games import dots_and_boxes
from typing import Any, Callable, Dict, Tuple


def heuristics(state: dots_and_boxes.DotsAndBoxesState) -> int:
    value = 0

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

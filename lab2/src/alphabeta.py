import numpy as np
from two_player_games.games import dots_and_boxes
from typing import Any, Callable, Dict, Tuple, List
from collections import deque
import random


MAX_NUM_OF_MOVES_TO_CHECK = 10


def get_lines_in_box(state: dots_and_boxes.DotsAndBoxesState, col: int, row: int) -> List[bool]:
    lines = [
        state.horizontals[col][row],
        state.horizontals[col][row + 1],
        state.verticals[row][col],
        state.verticals[row][col + 1]
        ]
    return lines


def find_chains(state: dots_and_boxes.DotsAndBoxesState) -> List[int]:
    visited_boxes = set()
    chains = []

    for row_idx in range(len(state.boxes)):
        for col_idx in range(len(state.boxes[0])):
            if (row_idx, col_idx) in visited_boxes or sum(get_lines_in_box(state, col_idx, row_idx)) != 3:
                continue

            chain_length = 0
            box_queue = deque([(row_idx, col_idx)])
            visited_boxes.add((row_idx, col_idx))

            while box_queue:
                curr_row, curr_col = box_queue.popleft()
                chain_length += 1
                curr_lines = get_lines_in_box(state, curr_col, curr_row)
                for (t_row, t_col), is_filled in zip([(-1, 0), (1, 0), (0, -1), (0, 1)], curr_lines):
                    if is_filled:
                        continue
                    bb_row, bb_col = curr_row + t_row, curr_col + t_col

                    if 0 <= bb_row < len(state.boxes) and 0 <= bb_col < len(state.boxes[0]) and (bb_row, bb_col) not in visited_boxes:
                        bordering_box_lines = get_lines_in_box(state, bb_col, bb_row)
                        if sum(bordering_box_lines) == 2 or sum(bordering_box_lines) == 3:
                            box_queue.append((bb_row, bb_col))
                            visited_boxes.add((bb_row, bb_col))

            chains.append(chain_length)

    return chains


def heuristics(state: dots_and_boxes.DotsAndBoxesState, maximizing_player: str) -> int:
    first_player = state.get_players()[0]
    second_player = state.get_players()[1]
    current_score = state.get_scores()
    found_chains = []
    max_num_of_lines = len(state.horizontals) * len(state.horizontals[0]) + len(state.verticals) * len(state.verticals[0])
    safe_moves = 0

    # Checking board for chains in mid game; In early game counting safe moves
    if len(state.get_moves()) <= max_num_of_lines * 0.5:
        found_chains = find_chains(state)
    else:
        for row in range(len(state.boxes)):
            for col in range(len(state.boxes[0])):
                lines = get_lines_in_box(state, col, row)
                if sum(lines) == 1:
                    safe_moves += 1

    # Evaluating game state by the current score, number of safe moves and found chains
    if first_player.char == maximizing_player:
        value = 10 * (current_score[first_player] - current_score[second_player])
        value += safe_moves
        for length in found_chains:
            value += 3 * (length - 1)

    else:
        value = 10 * (current_score[second_player] - current_score[first_player])
        for length in found_chains:
            value -= 3 * (length - 1)

    # Evaulating game state by the number of 3 wall boxes on the board
    for box_row_idx in range(len(state.boxes)):
        for box_col_idx in range(len(state.boxes[0])):
            lines = get_lines_in_box(state, box_col_idx, box_row_idx)
            if sum(lines) == len(lines) - 1:
                if first_player.char == maximizing_player:
                    value += 5
                else:
                    value -= 8

    return value


def alphabeta(
        game_state: dots_and_boxes.DotsAndBoxesState,
        depth: int,
        alfa: int = float('-inf'),
        beta: int = float('inf'),
        maximizing_player: str = "1"):

    if depth == 0 or game_state.is_finished():
        return heuristics(game_state, maximizing_player)

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
    equal_moves = []
    move_pool = []
    max_num_of_lines = len(state.horizontals) * len(state.horizontals[0]) + len(state.verticals) * len(state.verticals[0])

    if len(state.get_moves()) >= max_num_of_lines * 0.5 and len(state.get_moves()) > MAX_NUM_OF_MOVES_TO_CHECK:
        move_pool = np.random.choice(state.get_moves(), 10, replace=False)
    else:
        move_pool = state.get_moves()

    for move in move_pool:
        value = alphabeta(game_state=state.make_move(move), depth=depth, maximizing_player=player)

        if value > best_value:
            best_value = value
            best_move = move
            equal_moves = []
            equal_moves.append(best_move)
        if value == best_value:
            equal_moves.append(move)

    if len(equal_moves) > 1:
        best_move = np.random.choice(equal_moves)

    return best_move

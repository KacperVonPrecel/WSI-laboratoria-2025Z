import numpy as np
from two_player_games.games import dots_and_boxes
from typing import Any, Callable, Dict, Tuple, List
from collections import deque
import random


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

    for col_idx in range(len(state.boxes)):
        for row_idx in range(len(state.boxes[0])):
            if (col_idx, row_idx) in visited_boxes and sum(get_lines_in_box(state, col_idx, row_idx)) != 3:
                continue
            chain_length = 0
            box_queue = deque([(col_idx, row_idx)])
            visited_boxes.add((col_idx, row_idx))

            while box_queue:
                curr_col, curr_row = box_queue.popleft()
                chain_length += 1
                curr_lines = get_lines_in_box(state, curr_col, curr_row)
                for (t_col, t_row), is_filled in zip([(-1, 0), (1, 0), (0, -1), (0, 1)], curr_lines):
                    if is_filled:
                        continue
                    bb_col, bb_row = curr_col + t_col, curr_row + t_row

                    if 0 <= bb_col < len(state.boxes) and 0 <= bb_row < len(state.boxes[0]):
                        bordering_box_lines = get_lines_in_box(state, bb_col, bb_row)
                        if sum(bordering_box_lines) == 2 or sum(bordering_box_lines) == 3:
                            box_queue.append((bb_col, bb_row))
                            visited_boxes.add((bb_col, bb_row))

            chains.append(chain_length)

    return chains


def heuristics(state: dots_and_boxes.DotsAndBoxesState, maximizing_player: str) -> int:
    first_player = state.get_players()[0]
    second_player = state.get_players()[1]
    current_score = state.get_scores()

    if first_player == maximizing_player:
        value = 10 * (current_score[first_player] - current_score[second_player])
    else:
        value = 10 * (current_score[second_player] - current_score[first_player])

    for box_col_idx in range(len(state.boxes)):
        for box_row_idx in range(len(state.boxes[0])):
            lines = get_lines_in_box(state, box_col_idx, box_row_idx)
            if sum(lines) == len(lines) - 1:
                if first_player == maximizing_player:
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
    # idx = 0
    # best_idx = 0
    for move in state.get_moves():
        value = alphabeta(game_state=state.make_move(move), depth=depth, maximizing_player=player)

        if value > best_value:
            best_value = value
            best_move = move
            equal_moves = []
            equal_moves.append(best_move)
            # best_idx = idx
        if value == best_value:
            equal_moves.append(move)

        # idx += 1

    if len(equal_moves) > 1:
        best_move = np.random.choice(equal_moves)

    # print(f"Best move index: {best_idx}")
    print(f"Best value for player {player}: {best_value}")
    return best_move


if __name__ == "__main__":
    game = dots_and_boxes.DotsAndBoxes(size=3)
    i = 0

while not game.is_finished():
    i += 1
    if game.state.get_current_player().char == '1':
        best_curr_move = best_move(game.state, 5, '1')
    else:
        best_curr_move = best_move(game.state, 3, '2')

    game.make_move(best_curr_move)
    print(f"After {i} turn:")
    print(game.state)
    print(" ")

winner = game.get_winner()
if winner is None:
    print('Draw!')
else:
    print('Winner: Player ' + winner.char)
from two_player_games.games.dots_and_boxes import DotsAndBoxes  # or any other game
import random


game = DotsAndBoxes()
i = 0

while not game.is_finished():
    i += 1
    moves = game.get_moves()
    move = random.choice(moves)
    game.make_move(move)
    print(f"After {i} turn:")
    print(game.state)

winner = game.get_winner()
if winner is None:
    print('Draw!')
else:
    print('Winner: Player ' + winner.char)
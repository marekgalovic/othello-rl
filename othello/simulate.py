from argparse import ArgumentParser
import random

import tensorflow as tf

from agent import Agent
from board import Board
from player import GreedyPlayer, GreedyTreeSearchPlayer, AlphaBetaPlayer, RLPlayer


def main(args):
    board = Board()

    agent = Agent(board.size)
    tf.train.Checkpoint(net=agent).restore(args.checkpoint).expect_partial()

    players = [
        RLPlayer(agent, 0),
        AlphaBetaPlayer(1)
    ]
    curr_player_idx = random.choice([0, 1])

    while True:
        valid_positions = board.valid_positions(curr_player_idx)
        if len(valid_positions) == 0:
            break

        position_id = players[curr_player_idx].move(board)
        board.apply_position(curr_player_idx, valid_positions[position_id])
        print(board)
        curr_player_idx = 1 - curr_player_idx

    print(board.scores())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)

    main(parser.parse_args())


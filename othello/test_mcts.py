from argparse import ArgumentParser
import time

import numpy as np
import tensorflow as tf

from mcts import mcts
from agent import Agent
from board import Board

def main(args):
    board = Board()

    print(board)

    agent = Agent(board.size)
    tf.train.Checkpoint(net=agent).restore(args.checkpoint).expect_partial()

    curr_player_idx = 0
    while True:
        valid_positions = board.valid_positions(curr_player_idx)
        if len(valid_positions) == 0:
            break

        t = time.time()
        root, mcts_p, action_p, _ = mcts(board, agent, curr_player_idx, n_iter=args.n_iter)
        print('Time: %.4f' % float(time.time() - t))
        action_idx = np.random.choice(len(mcts_p), p=mcts_p)
        position_key = (int(action_idx / board.size), int(action_idx % board.size))

        board.apply_position(curr_player_idx, valid_positions[position_key])
        print(board)

        curr_player_idx = 1 - curr_player_idx


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n-iter', type=int, default=50)

    main(parser.parse_args())

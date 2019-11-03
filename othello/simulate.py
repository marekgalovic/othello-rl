from argparse import ArgumentParser
import random
import time

import tensorflow as tf
import numpy as np

from agent import Agent
from board import Board
from player import GreedyPlayer, GreedyTreeSearchPlayer, AlphaBetaPlayer, RLPlayer, RandomPlayer


def play_game(rl_player, opponent_player):
    players = (rl_player, opponent_player)

    board = Board()
    curr_player_idx = random.choice([0, 1])

    while True:
        valid_positions = board.valid_positions(curr_player_idx)
        if len(valid_positions) == 0:
            break

        position_id = players[curr_player_idx].move(board)
        board.apply_position(curr_player_idx, valid_positions[position_id])
        curr_player_idx = 1 - curr_player_idx

    return board.scores()


def main(args):
    # np.random.seed(19970617)

    board = Board()

    agent = Agent(board.size)
    tf.train.Checkpoint(net=agent).restore(args.checkpoint).expect_partial()

    rl_player = RLPlayer(agent, 0, mcts=True, mcts_iter=args.mcts_iter)
    opponents = [
        RandomPlayer(1),
        GreedyPlayer(1),
        GreedyTreeSearchPlayer(1),
        AlphaBetaPlayer(1)
    ]

    for opponent in opponents:
        print("%s" % opponent.__class__.__name__)
        wins, losses = 0, 0
        start_at = time.time()
        for i in range(args.n_games):
            scores = play_game(rl_player, opponent)
            print("\tgame: %d (%d, %d)" % (i, scores[0], scores[1]))
            wins += int(scores[0] > scores[1])
            losses += int(scores[1] > scores[0])

        print('\twins: %d (%.2f), losses: %d (%.2f), draws: %d (%.2f), time: %.2f\n' % (wins, wins / args.n_games, losses, losses / args.n_games, args.n_games - wins - losses, 1 - (wins + losses) / args.n_games, float(time.time() - start_at)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n-games', type=int, default=10)
    parser.add_argument('--mcts-iter', type=int, default=30)

    main(parser.parse_args())


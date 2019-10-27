import random

import ray
import tensorflow as tf

from board import Board
from agent import Agent
from player import RLPlayer, RandomPlayer, GreedyPlayer


@ray.remote
def _play_benchmark_game(agent_checkpoint, opponent_class, board_size, mcts_iter):
    agent = Agent(board_size)
    tf.train.Checkpoint(net=agent).restore(agent_checkpoint).expect_partial()

    players = [
        RLPlayer(agent, 0, mcts_iter=mcts_iter),
        opponent_class(1),
    ]

    board = Board(board_size)
    curr_player_idx = random.choice([0, 1])

    while True:
        valid_positions = board.valid_positions(curr_player_idx)
        if len(valid_positions) == 0:
            break

        position_id = players[curr_player_idx].move(board)
        board.apply_position(curr_player_idx, valid_positions[position_id])
        curr_player_idx = 1 - curr_player_idx

    scores = board.scores()
    if scores[0] > scores[1]:
        return 1
    elif scores[0] < scores[1]:
        return -1

    return 0


def benchmark_agent(agent_checkpoint, board_size, mcts_iter, n_games=20):
    ref_agents = {
        'greedy': GreedyPlayer,
        'random': RandomPlayer,
    }

    # Run benchmarks
    benchmark_futures = []
    benchmark_future_names = {}
    for name, player in ref_agents.items():
        for _ in range(n_games):
            future_id = _play_benchmark_game.remote(agent_checkpoint, player, board_size, mcts_iter)
            benchmark_futures.append(future_id)
            benchmark_future_names[future_id] = name

    player_stats = {name: [0, 0] for name in ref_agents.keys()}
    while benchmark_futures:
        done_ids, benchmark_futures = ray.wait(benchmark_futures)

        for future_id in done_ids:
            result = ray.get(future_id)
            name = benchmark_future_names[future_id]
            if result == 1:
                player_stats[name][0] += 1
            elif result == -1:
                player_stats[name][1] += 1

    return player_stats

import os
from argparse import ArgumentParser
from datetime import datetime
import time
import random
from multiprocessing import cpu_count

import ray
import numpy as np
import tensorflow as tf

from board import Board
from agent import Agent
from mcts import mcts, TerminalStateException
from player import RLPlayer, GreedyPlayer, GreedyTreeSearchPlayer, AlphaBetaPlayer
from utils import sample_checkpoint, get_state


def play_game(agent0, agent1, mcts_iter):
    board = Board()

    steps = 0
    agents = (agent0, agent1)
    # states = ([], [])
    curr_agent_idx = random.choice([0, 1])
    samples_buffer = []
    while True:
        steps += 1
        agent = agents[curr_agent_idx]

        try:
            valid_positions, valid_positions_ids, position_values, state, action_p, value = mcts(board, agent, curr_agent_idx, n_iter=mcts_iter)
        except TerminalStateException:
            break

        position_idx = np.argmax(position_values)
        position = valid_positions[position_idx]

        if curr_agent_idx == 0:
            samples_buffer.append([state, action_p[position_idx], valid_positions_ids[position_idx], value])

        board.apply_position(curr_agent_idx, position)

        # state, valid_positions, valid_positions_mask = get_state(board, curr_agent_idx)

        # if len(valid_positions) == 0:
        #     break

        # action_p, value = agent(tf.convert_to_tensor([state], dtype=tf.float32))
        # action_p = action_p[0].numpy() * valid_positions_mask.reshape((-1,))
        # value = value[0].numpy()
        # action_idx = np.random.choice(len(action_p), p=action_p / np.sum(action_p))

        # if curr_agent_idx == 0:
        #     samples_buffer.append([state, action_p[action_idx], action_idx, value])

        # position = (int(action_idx / board.size), int(action_idx % board.size))
        # board.apply_position(curr_agent_idx, valid_positions[position])

        curr_agent_idx = 1 - curr_agent_idx

    reward = 0
    player0_score, player1_score = board.scores()
    if player0_score < player1_score:
        reward = -1
    elif player1_score < player0_score:
        reward = 1

    return samples_buffer, reward, steps


@ray.remote
def play_games(agent0_checkpoint, agent1_checkpoint, board_size, n_games, mcts_iter, gamma):
    agent0 = Agent(board_size)
    tf.train.Checkpoint(net=agent0).restore(agent0_checkpoint).expect_partial()
    agent1 = Agent(board_size)
    tf.train.Checkpoint(net=agent1).restore(agent1_checkpoint).expect_partial()

    samples  = []
    total_steps, total_wins, total_losses = 0, 0, 0
    for _ in range(n_games):
        game_samples, reward, game_steps = play_game(agent0, agent1, mcts_iter)
        total_steps += game_steps
        total_wins += max(0, reward)
        total_losses += -min(0, reward)

        # Append discounted reward
        for i in range(len(game_samples)):
            game_samples[i].append(reward * (gamma ** (len(game_samples) - i)))

        samples.extend(game_samples)

    return samples, total_steps, total_wins, total_losses


@ray.remote
def run_benchmark(agent_checkpoint, opponent_class, board_size):
    agent = Agent(board_size)
    tf.train.Checkpoint(net=agent).restore(agent_checkpoint).expect_partial()

    players = [
        RLPlayer(agent, 0),
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


def collect_samples(checkpoints, board_size, n_games=1, mcts_iter=10, n_partitions=1, gamma=0.99):
    partition_games = int(np.ceil(n_games / n_partitions))

    futures = []
    for _ in range(n_partitions):
        futures.append(play_games.remote(checkpoints[-1], sample_checkpoint(checkpoints, p_latest=0.9), board_size, partition_games, mcts_iter, gamma))

    samples  = []
    total_steps, total_wins, total_losses = 0, 0, 0
    while futures:
        done_ids, futures = ray.wait(futures)

        for future_id in done_ids:
            (game_samples, steps, wins, losses) = ray.get(future_id)

            total_steps += steps
            total_wins += wins
            total_losses += losses
            samples.extend(game_samples)

    total_games = n_partitions * partition_games

    return samples, {
        'avg_game_length': total_steps / float(total_games),
        'win_rate': total_wins / float(total_games),
        'loss_rate': total_losses / float(total_games),
    }


def batches(samples, batch_size):
    random.shuffle(samples)

    for i in range(int(len(samples) / float(batch_size))):
        yield zip(*samples[i*batch_size:i*batch_size+batch_size])


def gather_action_probabilities(p, action_ids):
    gather_indices = tf.stack([
        tf.range(tf.shape(action_ids)[0]),
        action_ids
    ], -1)

    return tf.gather_nd(p, gather_indices)


def ppo_loss(new_values, values, p, p_old, action_ids, rewards, eps=0.2, c=1.0):
    advantage = rewards - values

    p = gather_action_probabilities(p, action_ids)
    r = p / p_old

    l_pg = tf.reduce_min([r * advantage, tf.clip_by_value(r, 1-eps, 1+eps) * advantage], axis=0)
    l_v = tf.square(new_values - rewards)

    return tf.reduce_mean(-l_pg + c*l_v)


def train(agent, optimizer, states, old_action_p, action_indices, state_values, rewards):
    with tf.GradientTape() as t:
        action_p, new_state_values = agent(states)

        loss = ppo_loss(new_state_values, state_values, action_p, old_action_p, action_indices, rewards)

    grads = t.gradient(loss, agent.trainable_variables)
    optimizer.apply_gradients(zip(grads, agent.trainable_variables))

    return loss


def main(args):
    job_dir = os.path.join(args.job_dir, datetime.now().strftime('%Y%m%d%H%M%s'))
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    print('Job dir: %s' % job_dir)

    board = Board()
    agent = Agent(board.size)

    optimizer = tf.keras.optimizers.Adam(lr=args.lr)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64), optimizer=optimizer, net=agent)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(job_dir, 'checkpoints'), max_to_keep=None)
    metrics_writer = tf.summary.create_file_writer(os.path.join(job_dir, 'metrics'))

    ref_agents = {
        'greedy': GreedyPlayer,
        'greedy_tree_search': GreedyTreeSearchPlayer,
        'alpha_beta': AlphaBetaPlayer,
    }

    try:
        ray.init(num_cpus=args.num_cpus)
        checkpoint_manager.save()

        with metrics_writer.as_default():
            for e in range(args.epochs):
                print('Epoch: %d' % e)
                t = time.time()
                samples, stats = collect_samples(checkpoint_manager.checkpoints, board.size, args.epoch_games, args.mcts_iter, n_partitions=args.num_cpus)
                print('Time to collect samples: %.4f' % (time.time() - t))

                for key, val in stats.items():
                    tf.summary.scalar(key, val, step=checkpoint.step)

                for (states, action_probabilities, action_indices, state_values, rewards) in batches(samples, args.batch_size):
                    loss = train(
                        agent,
                        optimizer,
                        tf.convert_to_tensor(states, dtype=tf.float32),
                        tf.convert_to_tensor(action_probabilities, dtype=tf.float32),
                        tf.convert_to_tensor(action_indices, dtype=tf.int32),
                        tf.convert_to_tensor(state_values, dtype=tf.float32),
                        tf.convert_to_tensor(rewards, dtype=tf.float32),
                    )
                    tf.summary.scalar('loss', loss, step=checkpoint.step)
                    tf.summary.scalar('mean_advantage', tf.reduce_mean(tf.convert_to_tensor(rewards, dtype=tf.float32) - tf.convert_to_tensor(state_values, dtype=tf.float32)), step=checkpoint.step)
                    checkpoint.step.assign_add(1)
                
                checkpoint_manager.save()

                if e > 0 and e % 5 == 0:
                    t = time.time()

                    # Run benchmarks
                    benchmark_futures = []
                    benchmark_future_names = {}
                    for name, player in ref_agents.items():
                        for _ in range(args.benchmark_games):
                            future_id = run_benchmark.remote(checkpoint_manager.latest_checkpoint, player, board.size)
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

                    for name, (wins, losses) in player_stats.items():
                        tf.summary.scalar('benchmark/%s/wins' % name, wins / args.benchmark_games, step=checkpoint.step)
                        tf.summary.scalar('benchmark/%s/losses' % name, losses / args.benchmark_games, step=checkpoint.step)

                    print('Time to run benchmarks: %.4f' % (time.time() - t))

    finally:
        ray.shutdown()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--job-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--epoch-games', type=int, default=5)
    parser.add_argument('--mcts-iter', type=int, default=30)
    parser.add_argument('--benchmark-games', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-cpus', type=int, default=cpu_count())

    main(parser.parse_args())

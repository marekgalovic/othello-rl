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
from agent import Agent, train
from mcts import MCTS, TerminalStateException
from player import RLPlayer
from benchmark import benchmark_agent
from utils import sample_checkpoint, get_state


def play_game(agent0, agent1, mcts_iter):
    board = Board()

    steps = 0
    # agents = (agent0, agent1)
    agents = (
        (agent0, MCTS(agent0, n_iter=mcts_iter)),
        (agent1, MCTS(agent1, n_iter=mcts_iter))
    )
    curr_agent_idx = random.choice([0, 1])
    samples_buffer = []
    while True:
        steps += 1
        
        # MCTS
        agent, mcts = agents[curr_agent_idx]
        try:
            root_node, mcts_p, action_p, value = mcts.search(board, curr_agent_idx)
            # root_node, mcts_p, action_p, value = mcts(board, agent, curr_agent_idx, n_iter=mcts_iter)
        except TerminalStateException:
            break

        state, valid_positions, valid_positions_mask = root_node.state
        if steps <= 20:
            action_idx = np.random.choice(len(mcts_p), p=mcts_p)
        else:
            action_idx = np.argmax(mcts_p)

        # /MCTS

        # No mcts
        # agent = agents[curr_agent_idx]
        # state, valid_positions, valid_positions_mask = get_state(board, curr_agent_idx)

        # if len(valid_positions) == 0:
        #     break

        # action_p, value = agent(tf.convert_to_tensor([state], dtype=tf.float32))
        # action_p = action_p[0].numpy() * valid_positions_mask.reshape((-1,))
        # value = value[0].numpy()
        # action_idx = np.random.choice(len(action_p), p=action_p / np.sum(action_p))
        # /No mcts

        if curr_agent_idx == 0:
            samples_buffer.append([state, action_p[action_idx], action_idx, value])

        position_key = (int(action_idx / board.size), int(action_idx % board.size))
        board.apply_position(curr_agent_idx, valid_positions[position_key])

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


def collect_samples(checkpoints, board_size, n_games=1, mcts_iter=10, n_partitions=1, gamma=0.99, checkpoint_gamma=0.2):
    partition_games = int(np.ceil(n_games / n_partitions))

    futures = []
    for _ in range(n_partitions):
        futures.append(play_games.remote(checkpoints[-1], sample_checkpoint(checkpoints, gamma=checkpoint_gamma), board_size, partition_games, mcts_iter, gamma))

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


@ray.remote
def play_comparison_game(old_checkpoint, new_checkpoint, mcts_iter):
    board = Board()

    old_agent = Agent(board.size)
    tf.train.Checkpoint(net=old_agent).restore(old_checkpoint).expect_partial()
    new_agent = Agent(board.size)
    tf.train.Checkpoint(net=new_agent).restore(new_checkpoint).expect_partial()

    players = (
        RLPlayer(old_agent, 0, mcts=True, mcts_iter=mcts_iter),
        RLPlayer(new_agent, 1, mcts=True, mcts_iter=mcts_iter)
    )

    curr_player_idx = random.choice([0, 1])
    while True:
        valid_positions = board.valid_positions(curr_player_idx)
        if len(valid_positions) == 0:
            break

        position_id = players[curr_player_idx].move(board)
        board.apply_position(curr_player_idx, valid_positions[position_id])
        curr_player_idx = 1 - curr_player_idx

    scores = board.scores()
    if scores[1] > scores[0]:
        return 1
    if scores[0] > scores[1]:
        return -1
    return 0


def compare_agents(old_checkpoint, new_checkpoint, n_games, mcts_iter):
    futures = []
    for _ in range(n_games):
        futures.append(play_comparison_game.remote(old_checkpoint, new_checkpoint, mcts_iter))

    new_wins, old_wins = 0, 0
    while futures:
        done_ids, futures = ray.wait(futures)
        for future_id in done_ids:
            result = ray.get(future_id)
            if result == 1:
                new_wins += 1
            if result == -1:
                old_wins += 1

    return new_wins, old_wins


def batches(samples, batch_size):
    random.shuffle(samples)

    for i in range(int(len(samples) / float(batch_size))):
        yield zip(*samples[i*batch_size:i*batch_size+batch_size])


def main(args):
    job_dir = args.job_dir if args.job_dir.startswith('gs') else os.path.join(args.job_dir, datetime.now().strftime('%Y%m%d%H%M%s'))
    if not tf.io.gfile.exists(job_dir):
        tf.io.gfile.makedirs(job_dir)
    print('Job dir: %s' % job_dir)

    board = Board()
    agent = Agent(board.size)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr,
        int((args.epoch_games * 60 * args.lr_decay_epochs) / args.batch_size),
        args.lr_decay
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    metrics_step = tf.Variable(1, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64), optimizer=optimizer, net=agent)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(job_dir, 'checkpoints'), max_to_keep=None)
    if args.contest_to_update:
        temp_checkpoint_manager = tf.train.CheckpointManager(checkpoint, os.path.join(job_dir, 'temp_checkpoint'), max_to_keep=1)
    metrics_writer = tf.summary.create_file_writer(os.path.join(job_dir, 'metrics'))

    try:
        ray.init(num_cpus=args.num_cpus)
        checkpoint_manager.save()

        with metrics_writer.as_default():
            for e in range(args.epochs):
                if args.contest_to_update:
                    # Restore the last accepted agent parameters
                    checkpoint.restore(checkpoint_manager.latest_checkpoint)
                # Benchmark
                if e % 5 == 0:
                    t = time.time()
                    for name, (wins, losses) in benchmark_agent(checkpoint_manager.latest_checkpoint, board.size, args.mcts_iter, n_games=args.benchmark_games).items():
                        tf.summary.scalar('benchmark/%s/wins' % name, wins / args.benchmark_games, step=metrics_step)
                        tf.summary.scalar('benchmark/%s/losses' % name, losses / args.benchmark_games, step=metrics_step)
                        tf.summary.scalar('benchmark/%s/draws' % name, (args.benchmark_games - wins - losses) / args.benchmark_games, step=metrics_step)

                    ttrb = float(time.time() - t)
                    tf.summary.scalar('perf/time_to_run_benchmarks', ttrb, step=metrics_step)
                    print('Time to run benchmarks: %.4f' % ttrb)

                # Collect epoch samples
                print('Epoch: %d' % e)
                t = time.time()
                samples, stats = collect_samples(
                    checkpoint_manager.checkpoints,
                    board.size,
                    args.epoch_games,
                    mcts_iter=args.mcts_iter,
                    n_partitions=args.num_cpus,
                    gamma=args.reward_gamma,
                    checkpoint_gamma=args.checkpoint_gamma
                )
                ttcs = float(time.time() - t)
                for key, val in stats.items():
                    tf.summary.scalar('game_metrics/%s' % key, val, step=metrics_step)
                tf.summary.scalar('perf/time_to_collect_samples', ttcs, step=metrics_step)
                print('Time to collect samples: %.4f' % ttcs)

                for (states, action_probabilities, action_indices, state_values, rewards) in batches(samples, args.batch_size):
                    if np.any(np.isnan(action_probabilities)):
                        raise ValueError('NaN Action P')

                    loss = train(
                        agent,
                        optimizer,
                        tf.convert_to_tensor(states, dtype=tf.float32),
                        tf.convert_to_tensor(action_probabilities, dtype=tf.float32),
                        tf.convert_to_tensor(action_indices, dtype=tf.int32),
                        tf.convert_to_tensor(state_values, dtype=tf.float32),
                        tf.convert_to_tensor(rewards, dtype=tf.float32),
                    )
                    tf.summary.scalar('train/loss', loss, step=metrics_step)
                    tf.summary.scalar('train/mean_advantage', tf.reduce_mean(tf.convert_to_tensor(rewards, dtype=tf.float32) - tf.convert_to_tensor(state_values, dtype=tf.float32)), step=metrics_step)
                    metrics_step.assign_add(1)
                    checkpoint.step.assign_add(1)

                if args.contest_to_update:
                    # Update parameters only if the new agent beats the old one.
                    temp_checkpoint_manager.save()
                    t = time.time()
                    new_wins, old_wins = compare_agents(checkpoint_manager.latest_checkpoint, temp_checkpoint_manager.latest_checkpoint, cpu_count(), args.mcts_iter)
                    tf.summary.scalar('perf/time_to_compare_agents', float(time.time() - t), step=metrics_step)
                    tf.summary.scalar('train/new_agent_win_rate', new_wins / (new_wins + old_wins), step=metrics_step)
                    if ((new_wins + old_wins) > 0) and (new_wins / (new_wins + old_wins) >= args.win_rate_threshold):
                        checkpoint_manager.save()
                else:
                    checkpoint_manager.save()

    finally:
        ray.shutdown()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--job-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epoch-games', type=int, default=5)
    parser.add_argument('--mcts-iter', type=int, default=50)
    parser.add_argument('--benchmark-games', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-decay', type=float, default=1.0)
    parser.add_argument('--lr-decay-epochs', type=int, default=5)
    parser.add_argument('--reward-gamma', type=float, default=0.99)
    parser.add_argument('--num-cpus', type=int, default=cpu_count())
    parser.add_argument('--checkpoint-gamma', type=float, default=0.2)
    parser.add_argument('--contest-to-update', type=bool, default=False)
    parser.add_argument('--win-rate-threshold', type=float, default=0.6)

    main(parser.parse_args())

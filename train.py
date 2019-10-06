from argparse import ArgumentParser
import random

import numpy as np
import tensorflow as tf

from board import Board
from agent import Agent


def get_state(board, color):
    valid_positions = board.valid_positions(color)
    valid_positions_mask = np.zeros(shape=(board.size, board.size, 1), dtype=np.uint8)
    valid_positions_mask[tuple(zip(*valid_positions.keys()))] = 1

    state = np.concatenate([
        valid_positions_mask,
        board.to_tensor(color)
    ], axis=-1)

    return state, valid_positions, valid_positions_mask


def play_game(agent0, agent1):
    board = Board()

    agents = (agent0, agent1)
    # states = ([], [])
    curr_agent_idx = random.choice([0, 1])
    samples_buffer = []
    while True:
        agent = agents[curr_agent_idx]
        state, valid_positions, valid_positions_mask = get_state(board, curr_agent_idx)

        if len(valid_positions) == 0:
            break

        action_p, value = agent(tf.convert_to_tensor([state], dtype=tf.float32))
        action_p = action_p[0].numpy() * valid_positions_mask.reshape((-1,))
        value = value[0].numpy()
        action_idx = np.random.choice(len(action_p), p=action_p / np.sum(action_p))

        if curr_agent_idx == 0:
            samples_buffer.append([state, action_p[action_idx], action_idx, value])

        position = (int(action_idx / board.size), int(action_idx % board.size))
        board.apply_position(curr_agent_idx, valid_positions[position])

        curr_agent_idx = 1 - curr_agent_idx

    reward = 0
    player0_score, player1_score = board.scores()
    if player0_score < player1_score:
        reward = -1
    elif player1_score < player0_score:
        reward = 1

    return samples_buffer, reward


def collect_samples(agent0, agent1, n_games=1, gamma=0.99):
    samples = []
    for _ in range(n_games):
        game_samples, reward = play_game(agent0, agent1)

        # Append discounted reward
        for i in range(len(game_samples)):
            game_samples[i].append(reward * (gamma ** (len(game_samples) - i)))

        samples.extend(game_samples)

    return samples


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

    return loss.numpy()


def main(args):
    board = Board()
    agent0 = Agent(board.size)
    agent1 = Agent(board.size)

    optimizer = tf.keras.optimizers.Adam(lr=args.lr)

    for e in range(args.epochs):
        samples = collect_samples(agent0, agent1, args.epoch_games)

        for (states, action_probabilities, action_indices, state_values, rewards) in batches(samples, args.batch_size):
            loss = train(
                agent0,
                optimizer,
                tf.convert_to_tensor(states, dtype=tf.float32),
                tf.convert_to_tensor(action_probabilities, dtype=tf.float32),
                tf.convert_to_tensor(action_indices, dtype=tf.int32),
                tf.convert_to_tensor(state_values, dtype=tf.float32),
                tf.convert_to_tensor(rewards, dtype=tf.float32),
            )

            print(loss)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--epoch_games', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    main(parser.parse_args())

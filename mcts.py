from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from agent import Agent
from board import Board
from utils import get_state


class Node:

    def __init__(self, board, color):
        self._board = board
        self._color = color

        self._wins = 0
        self._visits = 0
        self._children = []

    @property
    def board(self):
        return self._board

    @property
    def color(self):
        return self._color

    @property
    def wins(self):
        return self._wins

    @property
    def visits(self):
        return self._visits
    
    @property
    def children(self):
        return self._children

    def set_children(self, children):
        self._children = children

    def increment_wins(self):
        self._wins += 1

    def increment_visits(self):
        self._visits += 1


def mcts(board, agent, color, n_iter=100, c=np.sqrt(2)):
    root = Node(board.copy(), color)

    for i in range(n_iter):
        # print('Iter: %d' % i)
        _traverse(root, agent, color, c)

    return np.asarray(_child_scores(root, c=c))


def _child_scores(node, c=np.sqrt(2)):
    def _uct(child):
        if child.visits == 0:
            return np.inf

        return child.wins / child.visits + c * np.sqrt(np.log(node.visits) / child.visits)

    return list(map(_uct, node.children))


def _select_child(node, c=np.sqrt(2)):
    idx = np.argmax(_child_scores(node, c))
    return node.children[idx]


def _traverse(node, agent, root_color, c):
    # print("Visits: %d, Wins: %d, Children visits: %s" % (node.visits, node.wins, ','.join(list(map(lambda c: str(c.visits), node.children)))))
    node.increment_visits()

    if len(node.children) == 0:
        # Expand
        valid_positions = node.board.valid_positions(node.color)
        if len(valid_positions) == 0:
            # Terminal node
            scores = node.board.scores()
            if scores[root_color] > scores[1 - root_color]:
                node.increment_wins()
            return scores

        children = []
        for position in valid_positions.values():
            child_board = node.board.copy()
            child_board.apply_position(node.color, position)
            children.append(Node(child_board, 1 - node.color))

        node.set_children(children)
        scores = _simulate(_select_child(node, c), agent, root_color)
        if scores[root_color] > scores[1 - root_color]:
            node.increment_wins()
        return scores

    scores = _traverse(_select_child(node, c), agent, root_color, c)
    if scores[root_color] > scores[1 - root_color]:
        node.increment_wins()
    return scores


def _simulate(node, agent, root_color):
    node.increment_visits()

    board = node.board.copy()
    curr_color = node.color

    while True:
        state, valid_positions, valid_positions_mask = get_state(board, curr_color)

        if len(valid_positions) == 0:
            break

        action_p, _ = agent(tf.convert_to_tensor([state], dtype=tf.float32))
        action_p = action_p[0].numpy() * valid_positions_mask.reshape((-1,))
        action_idx = np.random.choice(len(action_p), p=action_p / np.sum(action_p))

        position_key = (int(action_idx / board.size), int(action_idx % board.size))
        board.apply_position(curr_color, valid_positions[position_key])

        curr_color = 1 - curr_color

    scores = board.scores()
    if scores[root_color] > scores[1 - root_color]:
        node.increment_wins()

    return scores


def main(args):
    board = Board()

    agent = Agent(board.size)
    tf.train.Checkpoint(net=agent).restore(args.checkpoint).expect_partial()

    curr_player_idx = 0

    while True:
        valid_positions = board.valid_positions(curr_player_idx)
        if len(valid_positions) == 0:
            break

        print(mcts(board, agent, curr_player_idx))
        raise ValueError


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)

    main(parser.parse_args())

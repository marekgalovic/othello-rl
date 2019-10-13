# from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from utils import get_state


class TerminalStateException(Exception): pass


class Node:

    def __init__(self, board, color):
        self._board = board
        self._color = color

        self._wins = 0
        self._visits = 0
        self._children = []

        self._state = None
        self._valid_positions = None
        self._valid_positions_ids = None
        self._p_v = None

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

    @property
    def state(self):
        if self._state is None:
            self._state = get_state(self.board, self.color)

        return self._state

    @property
    def valid_positions(self):
        if self._valid_positions is None:
            valid_positions = self.board.valid_positions(self.color)
            self._valid_positions = list(sorted(valid_positions.values(), key=lambda p: p.r_i * self.board.size + p.c_i))

        return self._valid_positions

    @property
    def valid_positions_ids(self):
        if self._valid_positions_ids is None:
            self._valid_positions_ids = [p.r_i * self.board.size + p.c_i for p in self.valid_positions]

        return self._valid_positions_ids

    def get_p_v(self, agent):
        if self._p_v is None:
            p, v = agent(tf.convert_to_tensor([self.state[0]], dtype=tf.float32))
            p = p[0].numpy()[self.valid_positions_ids]
            v = v[0].numpy()
            self._p_v = (p, v)

        return self._p_v

    def set_children(self, children):
        self._children = children

    def increment_wins(self):
        self._wins += 1

    def increment_visits(self):
        self._visits += 1


def mcts(board, agent, color, n_iter=10, c=np.sqrt(2)):
    root = Node(board.copy(), color)

    if len(root.valid_positions) == 0:
        raise TerminalStateException()

    for i in range(n_iter):
        _traverse(root, agent, color, c)

    raw_action_p, raw_state_v = root.get_p_v(agent)
    position_values = _child_values(root, agent, c)
    
    return root.valid_positions, root.valid_positions_ids, position_values, root.state[0], raw_action_p, raw_state_v

# def _uct_child_scores(node, c=np.sqrt(2)):
#     def _uct(child):
#         if child.visits == 0:
#             return np.inf

#         return child.wins / child.visits + c * np.sqrt(np.log(node.visits) / child.visits)

#     return np.asarray(list(map(_uct, node.children)))


# def _select_child(node, c=np.sqrt(2)):
#     idx = np.argmax(_uct_child_scores(node, c))
#     return node.children[idx]


def _child_values(node, agent, c):
    child_visit_counts = np.asarray([c.visits for c in node.children], dtype=np.float32)
    p, _ = node.get_p_v(agent)

    return c * (p / np.sum(p)) * (np.sqrt(node.visits) / (1 + child_visit_counts))


def _select_child(node, agent, c, greedy=True):
    child_values = _child_values(node, agent, c)

    if greedy:
        return node.children[np.argmax(child_values)]

    return node.children[np.random.choice(len(p), p=child_values/np.sum(child_values))]


def _traverse(node, agent, root_color, c):
    # print("Visits: %d, Wins: %d, Children visits: %s" % (node.visits, node.wins, ','.join(list(map(lambda c: str(c.visits), node.children)))))
    node.increment_visits()

    if len(node.children) == 0:
        # Expand
        if len(node.valid_positions) == 0:
            # Terminal node
            scores = node.board.scores()
            if scores[root_color] > scores[1 - root_color]:
                node.increment_wins()
            return scores

        children = []
        for position in node.valid_positions:
            child_board = node.board.copy()
            child_board.apply_position(node.color, position)
            children.append(Node(child_board, 1 - node.color))

        node.set_children(children)
        # scores = _simulate(_select_child(node, agent, c), agent, root_color)
        scores = _estimate_outcome(_select_child(node, agent, c), agent, root_color)
        if scores[root_color] > scores[1 - root_color]:
            node.increment_wins()
        return scores

    scores = _traverse(_select_child(node, agent, c), agent, root_color, c)
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


def _estimate_outcome(node, agent, root_color):
    node.increment_visits()

    state, _, _ = get_state(node.board, root_color)
    _, value = agent(tf.convert_to_tensor([state], dtype=tf.float32))

    scores = [0,0]
    scores[root_color] = value[0].numpy()
    scores[1 - root_color] = 1 - scores[root_color]

    if scores[root_color] > scores[1 - root_color]:
        node.increment_wins()

    return scores


# def main(args):
#     import time
#     from agent import Agent
#     from board import Board
#     board = Board()

#     print(board)

#     agent = Agent(board.size)
#     tf.train.Checkpoint(net=agent).restore(args.checkpoint).expect_partial()

#     curr_player_idx = 0

#     print('Play')

#     while True:
#         valid_positions = board.valid_positions(curr_player_idx)
#         if len(valid_positions) == 0:
#             break

#         t = time.time()
#         mcts(board, agent, curr_player_idx, n_iter=args.n_iter)
#         print('Time: %.4f' % float(time.time() - t))
#         raise ValueError


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--checkpoint', type=str, required=True)
#     parser.add_argument('--n-iter', type=int, default=100)

#     main(parser.parse_args())

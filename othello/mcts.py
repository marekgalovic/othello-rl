import numpy as np
import tensorflow as tf

from utils import get_state


class TerminalStateException(Exception): pass


class Node:

    def __init__(self, board, color):
        self._board = board
        self._color = color

        self._visits = 0
        self._children = []
        self._child_values = None
        self._child_visits = None

        self._state = None
        self._valid_positions = None
        self._valid_positions_indices = None
        self._p_v = None

    @property
    def board(self):
        return self._board
    
    @property
    def color(self):
        return self._color

    @property
    def visits(self):
        return self._visits
    
    @property
    def children(self):
        return self._children

    @property
    def child_values(self):
        return self._child_values

    @property
    def child_visits(self):
        return self._child_visits
    
    @property
    def state(self):
        if self._state is None:
            self._state = get_state(self.board, self.color)

        return self._state

    @property
    def valid_positions(self):
        if self._valid_positions is None:
            valid_positions = self.board.valid_positions(self.color)
            self._valid_positions = list(sorted(valid_positions.values(), key=lambda p: p.action_index(self.board)))

        return self._valid_positions

    @property
    def valid_positions_indices(self):
        if self._valid_positions_indices is None:
            self._valid_positions_indices = [p.action_index(self.board) for p in self.valid_positions]

        return self._valid_positions_indices

    def get_p_v(self, agent):
        if self._p_v is None:
            p, v = agent(tf.convert_to_tensor([self.state[0]], dtype=tf.float32))
            p = p[0].numpy()
            v = v[0].numpy()
            self._p_v = (p, v)

        return self._p_v

    def set_children(self, children):
        self._children = children
        self._child_values = np.zeros(len(children), dtype=np.float32)
        self._child_visits = np.zeros(len(children), dtype=np.int32)

    def increment_visits(self):
        self._visits += 1

    def update_value(self, child_idx, value):
        self._child_visits[child_idx] += 1
        self._child_values[child_idx] += value


class MCTS:

    def __init__(self, agent, n_iter=50, c=4.):
        self._agent = agent
        self._n_iter = int(n_iter)
        self._c = float(c)

        self._nodes = {}

    @property
    def agent(self):
        return self._agent

    @property
    def color(self):
        return self._color
    
    def search(self, board, color, tau=1.):
        board_key = board.to_canonical(color).tostring()
        if board_key not in self._nodes:
            self._nodes[board_key] = Node(board.copy(), color)

        root = self._nodes[board_key]
        if len(root.valid_positions) == 0:
            raise TerminalStateException()

        for _ in range(self._n_iter):
            self._traverse(root)

        p, v = root.get_p_v(self.agent)

        return root, _mcts_p(root, tau), p, v


    def _traverse(self, node):
        node.increment_visits()

        if len(node.children) == 0:
            if len(node.valid_positions) == 0:
                # Terminal state
                return -_get_value_from_scores(node.board.scores(), node.color)

            # Expand
            children = []
            for position in node.valid_positions:
                child_board = node.board.copy()
                child_board.apply_position(node.color, position)
                child_board_key = child_board.to_canonical(1 - node.color).tostring()
                if child_board_key not in self._nodes:
                    self._nodes[child_board_key] = Node(child_board, 1 - node.color)
                children.append(self._nodes[child_board_key])
            node.set_children(children)

            # Rollout
            idx = np.argmax(_ucb_score(self.agent, node, self._c))
            value = _estimate_outcome(self.agent, node.children[idx])
            node.update_value(idx, value)
            return -value

        # Continue traversal
        idx = np.argmax(_ucb_score(self.agent, node, self._c))
        value = self._traverse(node.children[idx])
        node.update_value(idx, value)
        return -value


def mcts(board, agent, color, n_iter=50, c=4., tau=1.):
    root = Node(board.copy(), color)

    if len(root.valid_positions) == 0:
        raise TerminalStateException()

    for i in range(n_iter):
        _traverse(agent, root, c)

    p, v = root.get_p_v(agent)

    return root, _mcts_p(root, tau), p, v


def _traverse(agent, node, c):
    node.increment_visits()

    if len(node.children) == 0:
        if len(node.valid_positions) == 0:
            # Terminal state
            return -_get_value_from_scores(node.board.scores(), node.color)

        # Expand
        children = []
        for position in node.valid_positions:
            child_board = node.board.copy()
            child_board.apply_position(node.color, position)
            children.append(Node(child_board, 1 - node.color))
        node.set_children(children)

        idx = np.argmax(_ucb_score(agent, node, c))
        value = _estimate_outcome(agent, node.children[idx])
        node.update_value(idx, value)
        return -value

    # Continue traversal
    idx = np.argmax(_ucb_score(agent, node, c))
    value = _traverse(agent, node.children[idx], c)
    node.update_value(idx, value)
    return -value


def _estimate_outcome(agent, node):
    # Use value function to estimate game outcome
    node.increment_visits()

    _, v = node.get_p_v(agent)
    return -v


def _simulate_truncated(agent, node, frac_occupied=.5):
    # Rollout until (board.size ** 2) * frac_occupied positions are occupied by players' stones
    node.increment_visits()

    board = node.board.copy()
    curr_color = node.color

    while (np.sum(board.scores()) / (board.size ** 2)) < frac_occupied:
        state, valid_positions, valid_positions_mask = get_state(board, curr_color)
 
        if len(valid_positions) == 0:
            break

        action_p, _ = agent(tf.convert_to_tensor([state], dtype=tf.float32))
        action_p = action_p[0].numpy() * valid_positions_mask.reshape((-1,))
        action_idx = np.argmax(action_p)

        position_key = (int(action_idx / board.size), int(action_idx % board.size))
        board.apply_position(curr_color, valid_positions[position_key])
        curr_color = 1 - curr_color

    valid_positions = board.valid_positions(node.color)
    if len(valid_positions) == 0:
        return -_get_value_from_scores(board.scores(), node.color)

    _, v = node.get_p_v(agent)
    return -v


def _simulate(agent, node):
    # Rollout until a terminal state is reached and return true win/loss value.
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

    return -_get_value_from_scores(board.scores(), node.color)


def _mcts_p(node, tau):
    mcts_p = np.zeros(node.board.size ** 2, dtype=np.float32)
    for i, visits in enumerate(node.child_visits):
        mcts_p[node.valid_positions_indices[i]] = visits ** (1 / tau)

    return mcts_p / np.sum(mcts_p)


def _ucb_score(agent, node, c):
    p, _ = node.get_p_v(agent)
    p = p[node.valid_positions_indices]
    if np.isclose(np.sum(p), 0):
        p = np.ones_like(p) / len(p)
    else:
        p /= np.sum(p)

    return (node.child_values / (node.child_visits + 1)) + c * p * (np.sqrt(node.visits) / (node.child_visits + 1))


def _get_value_from_scores(scores, color, v=1):
    if scores[color] > scores[1 - color]:
        return v
    elif scores[color] < scores[1 - color]:
        return -v
    return 0

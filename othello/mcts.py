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
        self._value = 0
        self._children = {}

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
    def wins(self):
        return self._wins

    @property
    def visits(self):
        return self._visits

    @property
    def value(self):
        return self._value
    
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

    def increment_wins(self):
        self._wins += 1

    def increment_visits(self):
        self._visits += 1

    def update_value(self, scores):
        self._value += scores[self.color]
        if scores[self.color] > 0:
            self.increment_wins()


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
            scores = _normalize_scores(node.board.scores(), node.color)
            node.update_value(scores)
            return scores

        # Expand
        children = []
        for position in node.valid_positions:
            child_board = node.board.copy()
            child_board.apply_position(node.color, position)
            children.append(Node(child_board, 1 - node.color))
        node.set_children(children)

        # Rollout
        scores = _estimate_outcome(agent, _select_child(agent, node, c))
        node.update_value(scores)
        return scores

    # Continue traversal
    scores = _traverse(agent, _select_child(agent, node, c), c)
    node.update_value(scores)
    return scores


def _estimate_outcome(agent, node):
    # Use value function to estimate game outcome
    node.increment_visits()

    _, v = node.get_p_v(agent)

    scores = [0,0]
    scores[node.color] = v
    scores[1 - node.color] = -v
    node.update_value(scores)

    return scores


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

    scores = _normalize_scores(board.scores(), node.color)
    node.update_value(scores)
    return scores


def _normalize_scores(scores, color, v=1):
    normalized = [0, 0]
    if scores[color] > scores[1 - color]:
        normalized[color] = v
    elif scores[color] < scores[1 - color]:
        normalized[color] = -v

    normalized[1 - color] = -normalized[color]
    return normalized


def _mcts_p(node, tau):
    mcts_p = np.zeros(node.board.size ** 2, dtype=np.float32)
    for i, child in enumerate(node.children):
        mcts_p[node.valid_positions_indices[i]] = child.visits ** (1 / tau)

    return mcts_p / np.sum(mcts_p)


def _ucb_score(agent, node, c):
    child_visit_counts = np.empty(len(node.children), dtype=np.float32)
    child_values = np.empty(len(node.children), dtype=np.float32)
    for i, child in enumerate(node.children):
        child_visit_counts[i] = child.visits
        child_values = -child.value # - here because the value is from opponents' perspective

    # State policy
    p, _ = node.get_p_v(agent)
    p = p[node.valid_positions_indices]

    return (child_values / (1 + child_visit_counts)) + c * (p / np.sum(p)) * (np.sqrt(node.visits) / (1 + child_visit_counts))


def _select_child(agent, node, c):
    ucb = _ucb_score(agent, node, c)

    return node.children[np.argmax(ucb)]
    # return node.children[np.random.choice(len(ucb), p=ucb/np.sum(ucb))]

import time

import numpy as np
import tensorflow as tf

from mcts import MCTS, TerminalStateException
from utils import get_state


class BasePlayer:

    TIME_LIMIT_MS = 950

    def __init__(self, color):
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def opponent_color(self):
        return 1 - self._color
    
    def move(self, board):
        raise NotImplementedError

    def _get_sorted_positions(self, board, color, n=None):
        # Returns valid positions sorted by score.
        valid_positions = sorted(board.valid_positions(color).values(), key=lambda p: p.total_steps, reverse=True)

        if n is not None:
            return valid_positions[:n]

        return valid_positions


class RLPlayer(BasePlayer):

    def __init__(self, agent, color, mcts_iter=30):
        super(RLPlayer, self).__init__(color)
        self._agent = agent
        self._mcts = MCTS(agent, n_iter=mcts_iter)
        # self._mcts_iter = int(mcts_iter)

    @property
    def agent(self):
        return self._agent

    @property
    def mcts(self):
        return self._mcts
    
    def move(self, board):
        # MCTS
        try:
            root_node, mcts_p, action_p, value = self.mcts.search(board, self.color)
            # root_node, mcts_p, action_p, value = mcts(board, self.agent, self.color, n_iter=self._mcts_iter)
        except TerminalStateException:
            return

        action_idx = np.argmax(mcts_p)

        # Pure RL
        # state, valid_positions, valid_positions_mask = get_state(board, self.color)
        # if len(valid_positions) == 0:
        #     return

        # action_p, _ = self.agent(tf.convert_to_tensor([state], dtype=tf.float32))
        # action_p = action_p[0].numpy() * valid_positions_mask.reshape((-1,))
        # action_idx = np.random.choice(len(action_p), p=action_p / np.sum(action_p))

        return (int(action_idx / board.size), int(action_idx % board.size))


class GreedyPlayer(BasePlayer):
    '''
        Greedy player always selects a position that maximizes 
        the total number of flipped opponent's positions.
    '''

    def move(self, board):
        best_score, best_position = 0, None
        for position in board.valid_positions(self.color).values():
            if position.total_steps > best_score:
                best_score = position.total_steps
                best_position = (position.r_i, position.c_i)

        return best_position


class GreedyTreeSearchPlayer(BasePlayer):
    '''
        Greedy tree search player is similar to the greedy player in a sense
        that it uses the number of flipped opponent's stones as a value function.
        However, it doesn't consider just the very next step but rather runs
        a simulation from a given board state (playing against itself) and then
        selects a move based on expected number of flipped opponent's positions.
    '''

    TOP_N_POSITIONS = 3
    MAX_ROLLOUT_DEPTH = 8

    def move(self, board):
        start_at = time.time()
        best_score, best_position = 0, None
        for idx, position in enumerate(self._get_sorted_positions(board, self.color, n=self.TOP_N_POSITIONS + 1)):
            score, depth = self._rollout(board.copy(), position, self.color, 1, start_at)

            if (score is not None) and (score > best_score) and (depth > 0):
                best_score = score
                best_position = (position[0], position[1])

        return best_position

    def _rollout(self, board, position, color, depth, start_at):
        if (time.time() - start_at) * 1000 > self.TIME_LIMIT_MS:
            return None, 0

        if depth == self.MAX_ROLLOUT_DEPTH:
            return board.score(self.color), depth

        board.apply_position(color, position)

        scores, max_depth = [], -1
        for position in self._get_sorted_positions(board, 1 - color, n=self.TOP_N_POSITIONS):
            score, rollout_depth = self._rollout(board.copy(), position, 1 - color, depth + 1, start_at)

            if score is not None:
                scores.append(score)

            if rollout_depth > max_depth:
                max_depth = rollout_depth

        if len(scores) == 0:
            return board.score(self.color), depth

        return np.mean(scores), max_depth


class AlphaBetaPlayer(BasePlayer):
    '''
        Alpha-beta player implements minimax algorithm with alpha-beta pruning.
        At levels of the tree that represent our move, algorithm selects a max
        of node's children as a node value. Similarly, at levels of the tree that
        represent opponent's move, algorithm selects min of node's children as
        a node value. The intuition is that we are playing our most aggressive
        strategy whilst expecting our opponent to play its least harmful strategy.
        Alpha-beta pruning is used to reduce the search space.
    '''

    TOP_N_POSITIONS = 3
    MAX_ROLLOUT_DEPTH = 8

    def move(self, board):
        start_at = time.time()
        alpha, beta = -float('inf'), float('inf')
        best_score, best_position = 0, None
        for idx, position in enumerate(self._get_sorted_positions(board, self.color, n=self.TOP_N_POSITIONS + 1)):
            score = self._min(board.copy(), position, 1, alpha, beta, start_at)
            if score > best_score:
                best_score = score
                best_position = (position[0], position[1])

        return best_position

    def _min(self, board, position, depth, alpha, beta, start_at):
        if (time.time() - start_at) * 1000.0 >= self.TIME_LIMIT_MS:
            return beta

        if depth == self.MAX_ROLLOUT_DEPTH:
            return board.score(self.color)

        board.apply_position(self.color, position)

        least_score = None
        for position in self._get_sorted_positions(board, self.opponent_color, n=self.TOP_N_POSITIONS):
            score = self._max(board.copy(), position, depth + 1, alpha, beta, start_at)
            beta = min(beta, score)

            if (least_score is None) or (score < least_score):
                least_score = score

            if least_score < alpha:
                # Parent maximizer would already choose alpha as it's its
                # best option.
                break

        if least_score is None:
            return board.score(self.color)

        return least_score

    def _max(self, board, position, depth, alpha, beta, start_at):
        if (time.time() - start_at) * 1000.0 >= self.TIME_LIMIT_MS:
            return alpha

        if depth == self.MAX_ROLLOUT_DEPTH:
            return board.score(self.color)

        board.apply_position(self.opponent_color, position)

        best_score = None
        for position in self._get_sorted_positions(board, self.color, n=self.TOP_N_POSITIONS):
            score = self._min(board.copy(), position, depth + 1, alpha, beta, start_at)
            alpha = max(alpha, score)

            if (best_score is None) or (score > best_score):
                best_score = score

            if best_score > beta:
                # Parent minimizer would already choose beta as it's its
                # best option.
                break

        if best_score is None:
            return board.score(self.color)

        return best_score

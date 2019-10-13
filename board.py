import random
from collections import namedtuple

import numpy as np


class Position(namedtuple('Position', ('r_i', 'c_i', 'directions', 'total_steps'))):
    pass


def _generate_directions(board_size, r_i, c_i):
    for change_r_i in [-1, 0, 1]:
        if (r_i + change_r_i) < 0 or (r_i + change_r_i) > board_size - 1:
            continue
        for change_c_i in [-1, 0, 1]:
            if (c_i + change_c_i) < 0 or (c_i + change_c_i) > board_size - 1:
                continue
            if change_r_i == 0 and change_c_i == 0:
                continue

            yield change_r_i, change_c_i


class Board:

    def __init__(self, size=8, board=None):
        assert isinstance(size, int) and (size % 2 == 0)
        self._size = int(size)

        self._valid_positions_cache = [None, None]
        if board is None:
            self._board = np.zeros(2 * (self._size ** 2), dtype=np.bool)
            self._init_empty()
        else:
            self._board = board

    @property
    def size(self):
        return self._size

    @property
    def raw(self):
        return self._board

    def copy(self):
        return Board(board=np.copy(self._board))

    def scores(self):
        occupancy = np.asarray(list(self._board[::2]), dtype=np.int8)
        player_positions = np.asarray(list(self._board[1::2]), dtype=np.int8)

        return np.dot(occupancy, 1 - player_positions), np.dot(occupancy, player_positions)

    def score(self, color):
        return self.scores()[color]

    def to_tensor(self, color):
        t = np.empty(shape=(self._size, self._size, 3), dtype=np.uint8)
        player_positions = np.asarray(list(self._board[1::2]), dtype=np.uint8).reshape((self._size, self._size))

        # Occupancy
        t[:,:,0] = np.asarray(list(self._board[::2]), dtype=np.uint8).reshape((self._size, self._size))
        # My positions
        t[:,:,1] = (player_positions == color) * t[:,:,0]
        # Opponent positions
        t[:,:,2] = (1 - t[:,:,1]) * t[:,:,0]

        return t

    def apply_position(self, color, position):
        for direction in position.directions:
            r_i = position.r_i
            c_i = position.c_i
            for _ in range(direction[2]):
                idx = 2 * (r_i * self._size + c_i)
                self._board[idx] = 1
                self._board[idx+1] = color
                r_i += direction[0]
                c_i += direction[1]

        self._valid_positions_cache = [None, None]

    def valid_positions(self, color):
        if self._valid_positions_cache[color] is not None:
            return self._valid_positions_cache[color]

        valid_positions = {}
        for r_i in range(self._size):
            for c_i in range(self._size):
                if self._board[2 * (r_i * self._size + c_i)]:
                    continue

                directions = []
                total_steps = 0
                for (r_direction, c_direction) in _generate_directions(self._size, r_i, c_i):
                    num_steps = self._traverse_direction(color, r_i, c_i, r_direction, c_direction)

                    if num_steps > 0:
                        total_steps += num_steps
                        directions.append((r_direction, c_direction, num_steps))

                if len(directions) > 0:
                    valid_positions[(r_i, c_i)] = Position(r_i, c_i, directions, total_steps)

        self._valid_positions_cache[color] = valid_positions
        return valid_positions

    def _get_position(self, r_i, c_i):
        [occupied, color] = self._board[2*(r_i*self._size+c_i):2*(r_i*self._size+c_i)+2]
        return bool(occupied), int(color)

    def _set_position(self, r_i, c_i, occupied, color):
        self._board[2*(r_i*self._size+c_i):2*(r_i*self._size+c_i)+2] = [occupied, color]

    def _init_empty(self):
        player_a = random.choice([0, 1])
        idx = int(self._size / 2 - 1)

        self._set_position(idx, idx, 1, player_a)
        self._set_position(idx, idx+1, 1, 1-player_a)
        self._set_position(idx+1, idx, 1, 1-player_a)
        self._set_position(idx+1, idx+1, 1, player_a)

    def _traverse_direction(self, color, r_i, c_i, r_direction, c_direction):
        num_steps = 1
        r_i += r_direction
        c_i += c_direction
        idx = 2 * (r_i * self._size + c_i)
        [occupied, pos_color] = self._board[idx:idx+2]
        while occupied and pos_color == (1 - color):
            r_i += r_direction
            c_i += c_direction
            if r_i < 0 or r_i > self._size - 1:
                return 0
            if c_i < 0 or c_i > self._size - 1:
                return 0

            num_steps += 1 
            idx = 2 * (r_i * self._size + c_i)
            [occupied, pos_color] = self._board[idx:idx+2]

        if occupied and pos_color == color and num_steps > 1:
            return num_steps

        return 0

    def __str__(self):
        rows, scores = [], [0, 0]
        rows.append('\t' + '\t'.join(['%d' % i for i in range(self._size)]))
        rows.append('\t' + '\t'.join('-' for _ in range(self._size)))
        for r_i in range(self._size):
            cols = []
            for c_i in range(self._size):
                occupied, color = self._get_position(r_i, c_i)
                if occupied:
                    scores[int(color)] += 1
                    cols.append('%d' % int(color))
                else:
                    cols.append('.')
            rows.append(('%d |\t' % r_i) + '\t'.join(cols))

        return '\n'.join(rows) + '\nPlayer 0: %d, Player 1: %d' % tuple(scores)

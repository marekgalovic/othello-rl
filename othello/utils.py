import numpy as np


def sample_checkpoint(checkpoints, gamma=0.2, last_n=None):
    assert len(checkpoints) > 0

    if len(checkpoints) == 1:
        return checkpoints[0]

    if last_n is not None:
        p = gamma ** np.flip(np.arange(len(checkpoints[-last_n:])))
        return np.random.choice(checkpoints[-last_n:], p=p / np.sum(p))

    p = gamma ** np.flip(np.arange(len(checkpoints)))
    return np.random.choice(checkpoints, p=p / np.sum(p))


def get_state(board, color):
    valid_positions = board.valid_positions(color)
    valid_positions_mask = np.zeros(shape=(board.size, board.size, 1), dtype=np.uint8)
    valid_positions_mask[tuple(zip(*valid_positions.keys()))] = 1

    state = np.concatenate([
        valid_positions_mask,
        board.to_tensor(color)
    ], axis=-1)

    return state, valid_positions, valid_positions_mask

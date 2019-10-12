import numpy as np


def generate_directions(board_size, r_i, c_i):
    for change_r_i in [-1, 0, 1]:
        if (r_i + change_r_i) < 0 or (r_i + change_r_i) > board_size - 1:
            continue
        for change_c_i in [-1, 0, 1]:
            if (c_i + change_c_i) < 0 or (c_i + change_c_i) > board_size - 1:
                continue
            if change_r_i == 0 and change_c_i == 0:
                continue

            yield change_r_i, change_c_i


def sample_checkpoint(checkpoints, p_latest=1.0):
    assert len(checkpoints) > 0

    if np.random.uniform() <= p_latest:
        return checkpoints[-1]

    p = 1 / (1 + np.arange(len(checkpoints)))
    p /= np.sum(p)

    return np.random.choice(checkpoints, p=p)


def get_state(board, color):
    valid_positions = board.valid_positions(color)
    valid_positions_mask = np.zeros(shape=(board.size, board.size, 1), dtype=np.uint8)
    valid_positions_mask[tuple(zip(*valid_positions.keys()))] = 1

    state = np.concatenate([
        valid_positions_mask,
        board.to_tensor(color)
    ], axis=-1)

    return state, valid_positions, valid_positions_mask

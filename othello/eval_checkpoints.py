import os
import json

import ray
import numpy as np
import tensorflow as tf

from player import RandomPlayer, GreedyPlayer, AlphaBetaPlayer, GreedyTreeSearchPlayer
from benchmark import benchmark_agent

EVERY_NTH = 5


def get_checkpoint_files(checkpoints_dir):
    for i in range(0, 51, 5):
        yield os.path.join(checkpoints_dir, 'ckpt-%d' % (i + 1))


def write_result(result, output_dir):
    serializable = dict(map(lambda kv: (kv[0], kv[1].tolist()), result.items()))
    f = tf.io.gfile.GFile(os.path.join(output_dir, 'eval_result.json'), mode='w')
    json.dump(serializable, f)
    f.close()


def main(args):
    checkpoints = list(get_checkpoint_files(os.path.join(args.job_dir, 'checkpoints')))

    try:
        ray.init(num_cpus=args.num_cpus)

        ref_agents = {
            'random': RandomPlayer,
            'greedy': GreedyPlayer,
            'greedy_tree_search': GreedyTreeSearchPlayer,
            'alpha_beta': AlphaBetaPlayer,
        }
        result = {
            name: np.empty(len(checkpoints), dtype=np.float32)
            for name in ref_agents.keys()
        }

        for i, checkpoint_path in enumerate(checkpoints):
            print('Checkpoint: %s' % checkpoint_path)
            for opponent_name, r in benchmark_agent(checkpoint_path, 8, args, ref_agents=ref_agents).items():
                result[opponent_name][i] = (r[0] / args.benchmark_games)

        write_result(result, args.job_dir)

    finally:
        ray.shutdown()

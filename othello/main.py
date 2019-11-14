from argparse import ArgumentParser
from multiprocessing import cpu_count

from train import main as train_main
from eval_checkpoints import main as eval_main


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--job-dir', type=str, required=True)
    parser.add_argument('--agent-net-size', type=int, default=256)
    parser.add_argument('--agent-net-conv', type=int, default=5)
    # parser.add_argument('--agent-net-dropout', type=float, default=0.2)
    parser.add_argument('--mcts-iter', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epoch-games', type=int, default=5)
    parser.add_argument('--benchmark-games', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-decay', type=float, default=1.0)
    parser.add_argument('--lr-decay-epochs', type=int, default=5)
    parser.add_argument('--reward-gamma', type=float, default=0.99)
    parser.add_argument('--num-cpus', type=int, default=cpu_count())
    parser.add_argument('--checkpoint-gamma', type=float, default=0.2)
    parser.add_argument('--checkpoint-last-n', type=int, default=None)
    parser.add_argument('--contest-to-update', type=bool, default=False)
    parser.add_argument('--win-rate-threshold', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--eval', type=bool, default=False)

    args = parser.parse_args()
    if args.eval:
        eval_main(args)
    else:
        train_main(args)


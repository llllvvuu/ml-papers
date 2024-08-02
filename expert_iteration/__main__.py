import argparse
import logging

from .hex import Hex
from .hex_net import HexNet
from .train import train


def main():
    parser = argparse.ArgumentParser(description="Expert Iteration for Hex")
    _ = parser.add_argument("command", choices=["train"], help="Command to execute")
    _ = parser.add_argument(
        "output", type=str, help="Output file path for the trained model"
    )
    _ = parser.add_argument(
        "--iters", type=int, default=100, help="Number of training iterations"
    )
    _ = parser.add_argument(
        "--board_size", type=int, default=5, help="Size of the Hex board"
    )
    _ = parser.add_argument(
        "--checkpoint", type=str, help="Path to load a checkpoint from"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.command == "train":
        train(
            HexNet(args.board_size),
            Hex(args.board_size),
            args.output,
            args.iters,
            args.checkpoint,
        )


if __name__ == "__main__":
    main()

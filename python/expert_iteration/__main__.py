import argparse
import logging

from .hex import Hex
from .hex_convnet import HexConvnet
from .hex_transformer import HexTransformer
from .train import train
from .frontend import hex


def main():
    parser = argparse.ArgumentParser(description="Expert Iteration for Hex")
    _ = parser.add_argument(
        "command", choices=["train", "play"], help="Command to execute"
    )
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
    _ = parser.add_argument(
        "--arch",
        choices=["convnet", "transformer"],
        default="convnet",
        help="Neural network architecture to use",
    )
    _ = parser.add_argument("--loss_csv", type=str, help="Path save loss csv to")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.command == "train":
        train(
            HexTransformer(args.board_size)
            if args.arch == "transformer"
            else HexConvnet(args.board_size),
            Hex(args.board_size),
            args.output,
            args.iters,
            args.checkpoint,
            args.loss_csv,
        )
    elif args.command == "play":
        mode = input("Enter mode ([apprentice], [e]xpert): ")
        if mode.lower().startswith("e"):
            mode = "expert"
        hex.play_against_exit(args.arch, args.board_size, args.output, mode)


if __name__ == "__main__":
    main()

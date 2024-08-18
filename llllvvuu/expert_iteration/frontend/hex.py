import torch

from ...mcts.Hex import HexColor

from ..hex import Hex
from ..hex_convnet import HexConvnet
from ..hex_transformer import HexTransformer
from ..expert_iteration import ExpertIteration


def play_against_exit(arch: str, board_size: int, model_path: str, mode: str):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = (
        HexTransformer(board_size) if arch == "transformer" else HexConvnet(board_size)
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    _ = model.load_state_dict(checkpoint["model_state_dict"])
    _ = model.eval()

    exit = ExpertIteration(Hex(board_size), model)

    player_color = input("Play as? ([B]lue, [R]ed, [Selfplay]): ").upper()
    is_blue_player = player_color.startswith("B")
    is_selfplay = not is_blue_player and not player_color.startswith("R")

    game = Hex(board_size)

    while not game.is_terminal():
        print(game)
        print(" " * (game.size + 1) + " ".join(str(i) for i in range(game.size)))

        if (game.blue_turn == is_blue_player) and not is_selfplay:
            while True:
                try:
                    row = int(input(f"Enter row (0-{board_size-1}): "))
                    col = int(input(f"Enter column (0-{board_size-1}): "))
                    move = row * board_size + col
                    if game.board[row][col] == HexColor.Empty:
                        game = game.from_network_output(move)
                        break
                    else:
                        raise ValueError("Cell is already occupied")
                except (ValueError, IndexError) as e:
                    print(f"Invalid input: {e}")
        else:
            if mode == "expert":
                move, value, policy = exit.expert_move(game)
            else:
                move, value, policy = exit.apprentice_move(game)
            print("Value:", value)
            print("Policy:")
            for i in range(0, len(policy), board_size):
                row = policy[i : i + board_size]
                print(" ".join(f"{x:.8f}" for x in row))
            row, col = divmod(move, board_size)
            if game.board[row][col] == HexColor.Empty:
                game = game.from_network_output(move)
            else:
                print("AI played an invalid move, game ends!")
                break

    print(game)
    if game.reward() == 1:
        print("Blue wins!")
    elif game.reward() == -1:
        print("Red wins!")

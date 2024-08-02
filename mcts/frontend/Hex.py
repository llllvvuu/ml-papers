from ..MCTS import MCTSNode
from ..Hex import Hex, HexColor

DEFAULT_SIMS = 1000


def user_move(node: MCTSNode[Hex], row: int, col: int) -> MCTSNode[Hex]:
    new_board = [row.copy() for row in node.state.board]
    if new_board[row][col] != HexColor.Empty:
        raise ValueError("Cell is already occupied")

    new_board[row][col] = HexColor.Blue if node.state.blue_turn else HexColor.Red

    if new_board not in map(lambda x: x.board, node.state.next_states()):
        raise ValueError("Invalid move")

    return next(
        filter(lambda x: x.state.board == new_board, node.children),
        MCTSNode(Hex(node.state.size, new_board, not node.state.blue_turn), node),
    )


def pve(blue: bool, size: int, sims: int = DEFAULT_SIMS):
    node = MCTSNode(Hex(size), None)
    if not blue:
        for _ in range(sims):
            node.run()
        node = node.best_child()
    while node:
        next_node = None
        while not next_node:
            try:
                print(node.state)
                print(
                    " " * (node.state.size + 1)
                    + " ".join(str(i) for i in range(node.state.size))
                )
                row = int(input(f"Enter row (0-{node.state.size-1}): "))
                col = int(input(f"Enter column (0-{node.state.size-1}): "))
                next_node = user_move(node, row, col)
            except (ValueError, IndexError) as e:
                print(f"⚠️ Invalid move: {e}")
        node = next_node
        if node.state.is_terminal():
            print(node.state)
            if node.state.terminal_reward() == 0:
                print("😐 Draw...")
            else:
                print("🎉 You Win!")
            break
        for _ in range(sims):
            node.run()
        print("Visits:", node.visits)
        print(
            "Rewards:",
            [
                child.reward / child.visits if child.visits else 0
                for child in node.children
            ],
        )
        node = node.best_child()
        if not node:
            print("You win! (AI forfeit)")
        elif node.state.is_terminal():
            print(node.state)
            if node.state.terminal_reward() == 0:
                print("😐 Draw...")
            else:
                print("😭 You Lose!")
            break


def selfplay(size: int, sims: int = DEFAULT_SIMS):
    node = MCTSNode(Hex(size), None)
    while node:
        print(node.state)
        if node.state.is_terminal():
            if node.state.terminal_reward() == 1:
                print("🎉 Blue Wins!")
            elif node.state.terminal_reward() == -1:
                print("🎉 Red Wins!")
            else:
                print("😐 Draw...")
            break
        for _ in range(sims):
            node.run()
        print("Visits:", node.visits)
        print(
            "Rewards:",
            [
                child.reward / child.visits if child.visits else 0
                for child in node.children
            ],
        )
        node = node.best_child()

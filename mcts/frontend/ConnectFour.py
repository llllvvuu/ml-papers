from ..MCTS import MCTSNode
from ..ConnectFour import ConnectFour, ConnectFourPiece

DEFAULT_SIMS = 1000


def user_move(node: MCTSNode[ConnectFour], col: int) -> MCTSNode[ConnectFour]:
    new_board = [row.copy() for row in node.state.board]
    for row in range(ConnectFour.ROWS - 1, -1, -1):
        if new_board[row][col] == ConnectFourPiece.Empty:
            new_board[row][col] = (
                ConnectFourPiece.Red if node.state.red_turn else ConnectFourPiece.Yellow
            )
            break
    else:
        raise ValueError("Column is full")

    if new_board not in map(lambda x: x.board, node.state.next_states()):
        raise ValueError("Invalid move")

    return next(
        filter(lambda x: x.state.board == new_board, node.children),
        MCTSNode(ConnectFour(new_board, not node.state.red_turn), node),
    )


def pve(red: bool, sims: int = DEFAULT_SIMS):
    node = MCTSNode(ConnectFour(), None)
    if not red:
        for _ in range(sims):
            node.run()
        node = node.best_child()
    while node:
        next_node = None
        while not next_node:
            try:
                print(node.state)
                print(" ".join(str(i) for i in range(node.state.COLS)))
                col = int(input(f"Enter column (0-{ConnectFour.COLS-1}): "))
                next_node = user_move(node, col)
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


def selfplay(sims: int = DEFAULT_SIMS):
    node = MCTSNode(ConnectFour(), None)
    while node:
        print(node.state)
        if node.state.is_terminal():
            if node.state.terminal_reward() == 1:
                print("🎉 Red Wins!")
            elif node.state.terminal_reward() == -1:
                print("🎉 Yellow Wins!")
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

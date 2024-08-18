from ..MCTS import MCTSNode
from ..TicTacToe import TicTacToe, TicTacToeSquare

DEFAULT_SIMS = 1000


def user_move(node: MCTSNode[TicTacToe], index: int) -> MCTSNode[TicTacToe]:
    board = node.state.board.copy()
    board[index] = TicTacToeSquare.X if node.state.X else TicTacToeSquare.O
    if board not in map(lambda x: x.board, node.state.next_states()):
        raise ValueError("Square is occupied")
    return next(
        filter(lambda x: x.state.board == board, node.children),
        MCTSNode(TicTacToe(board, not node.state.X), node),
    )


def pve(X: bool, sims: int = DEFAULT_SIMS):
    node = MCTSNode(TicTacToe(), None)
    if not X:
        for _ in range(sims):
            node.run()
        node = node.best_child()
    while node:
        next_node = None
        while not next_node:
            try:
                print(node.state)
                print("0 1 2\n3 4 5\n6 7 8")
                index = int(input("Enter move (0-8): "))
                next_node = user_move(node, index)
            except (ValueError, IndexError) as e:
                print(f"⚠️ Invalid move: {e}")
        node = next_node
        if node.state.is_terminal():
            if node.state.reward() == 0:
                print("😐 Draw...")
            else:
                print("🎉 You Win!")
            break
        for _ in range(sims):
            node.run()
        print("Visits:", node.visits)
        print("Rewards:", list(map(lambda x: x.reward / x.visits, node.children)))
        node = node.best_child()
        if not node:
            print("You win! (AI forfeit)")
        elif node.state.is_terminal():
            print(node.state)
            if node.state.reward() == 0:
                print("😐 Draw...")
            else:
                print("😭 You Lose!")
            break


def selfplay(sims: int = DEFAULT_SIMS):
    node = MCTSNode(TicTacToe(), None)
    while node:
        print(node.state)
        if node.state.is_terminal():
            if node.state.reward() == 1:
                print("🎉 X Wins!")
            elif node.state.reward() == -1:
                print("🎉 O Wins!")
            else:
                print("😐 Draw...")
            break
        for _ in range(sims):
            node.run()
        print("Visits:", node.visits)
        print("Rewards:", list(map(lambda x: x.reward / x.visits, node.children)))
        node = node.best_child()

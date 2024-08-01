from .frontend import ConnectFour, TicTacToe

if __name__ == "__main__":
    game = input("Enter game ([TicTacToe]/[C]onnectFour: ")
    if game.startswith("C"):
        player = input("Play as? (R, Y, [selfplay]): ")
        if player == "R":
            ConnectFour.pve(True)
        elif player == "Y":
            ConnectFour.pve(False)
        else:
            ConnectFour.selfplay()
    else:
        player = input("Play as? (X, O, [selfplay]): ")
        if player == "X":
            TicTacToe.pve(True)
        elif player == "O":
            TicTacToe.pve(False)
        else:
            TicTacToe.selfplay()

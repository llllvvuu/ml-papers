from .frontend import ConnectFour, Hex, TicTacToe

if __name__ == "__main__":
    game = input("Enter game ([TicTacToe]/[C]onnectFour/[H]ex): ")
    if game.startswith("C"):
        player = input("Play as? ([R]ed, [Y]ellow, [selfplay]): ")
        if player == "R":
            ConnectFour.pve(True)
        elif player == "Y":
            ConnectFour.pve(False)
        else:
            ConnectFour.selfplay()
    elif game.startswith("H"):
        size = None
        while not size:
            try:
                size = int(input("Enter size (3-9): "))
            except ValueError as e:
                print(f"Invalid size: {e}")
            if size and (size < 3 or size > 9):
                print("Size must be between 3 and 9")
                size = None
        player = input("Play as? ([B]lue, [R]ed, [selfplay]): ")
        if player == "B":
            Hex.pve(True, size)
        elif player == "R":
            Hex.pve(False, size)
        else:
            Hex.selfplay(size)
    else:
        player = input("Play as? (X, O, [selfplay]): ")
        if player == "X":
            TicTacToe.pve(True)
        elif player == "O":
            TicTacToe.pve(False)
        else:
            TicTacToe.selfplay()

from enum import Enum
from random import choice
from typing import Self, cast, override
from .MCTS import MCTSState


class ConnectFourPiece(Enum):
    Empty = 0
    Red = 1
    Yellow = 2

    @override
    def __str__(self) -> str:
        return "⬜🔴🟡"[self.value]


class ConnectFour(MCTSState):
    ROWS = 6
    COLS = 7

    def __init__(
        self, board: list[list[ConnectFourPiece]] | None = None, red_turn: bool = True
    ):
        if not board:
            board = [
                [ConnectFourPiece.Empty for _ in range(self.COLS)]
                for _ in range(self.ROWS)
            ]
        self.board = board
        self.red_turn = red_turn

    def wins(self, player: ConnectFourPiece) -> bool:
        # Check horizontal
        for row in range(self.ROWS):
            for col in range(self.COLS - 3):
                if all(self.board[row][col + i] == player for i in range(4)):
                    return True

        # Check vertical
        for row in range(self.ROWS - 3):
            for col in range(self.COLS):
                if all(self.board[row + i][col] == player for i in range(4)):
                    return True

        # Check diagonal (positive slope)
        for row in range(self.ROWS - 3):
            for col in range(self.COLS - 3):
                if all(self.board[row + i][col + i] == player for i in range(4)):
                    return True

        # Check diagonal (negative slope)
        for row in range(3, self.ROWS):
            for col in range(self.COLS - 3):
                if all(self.board[row - i][col + i] == player for i in range(4)):
                    return True

        return False

    @override
    def next_states(self) -> list[Self]:
        states: list[MCTSState] = []
        for col in range(self.COLS):
            if self.board[0][col] == ConnectFourPiece.Empty:
                new_board = [row.copy() for row in self.board]
                for row in range(self.ROWS - 1, -1, -1):
                    if new_board[row][col] == ConnectFourPiece.Empty:
                        new_board[row][col] = (
                            ConnectFourPiece.Red
                            if self.red_turn
                            else ConnectFourPiece.Yellow
                        )
                        break
                states.append(ConnectFour(new_board, not self.red_turn))
        return cast(list[Self], states)

    def random_next_state(self) -> Self:
        col = choice([col for col in range(self.COLS) if self.board[0][col] == ConnectFourPiece.Empty])
        new_board = [row.copy() for row in self.board]
        for row in range(self.ROWS - 1, -1, -1):
            if new_board[row][col] == ConnectFourPiece.Empty:
                new_board[row][col] = (
                    ConnectFourPiece.Red if self.red_turn else ConnectFourPiece.Yellow
                )
                break
        return cast(Self, ConnectFour(new_board, not self.red_turn))

    @override
    def random_rollout(self) -> tuple[float, int]:
        state = self
        depth = 0
        while not state.is_terminal():
            state = state.random_next_state()
            depth += 1
        return state.reward(), depth

    @override
    def is_terminal(self) -> bool:
        return (
            self.wins(ConnectFourPiece.Red)
            or self.wins(ConnectFourPiece.Yellow)
            or all(
                self.board[0][col] != ConnectFourPiece.Empty for col in range(self.COLS)
            )
        )

    @override
    def reward(self) -> float:
        if self.wins(ConnectFourPiece.Red):
            return 1
        if self.wins(ConnectFourPiece.Yellow):
            return -1
        return 0

    @override
    def reward_perspective(self, reward: object) -> float:
        assert isinstance(reward, float) or isinstance(reward, int)
        if self.red_turn:
            return reward
        return -reward

    @override
    def __str__(self) -> str:
        return "\n".join("".join(str(piece) for piece in row) for row in self.board)

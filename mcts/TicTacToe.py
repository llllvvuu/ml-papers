from enum import Enum
from random import choice
from typing import Self, cast, override
from .MCTS import MCTSState


class TicTacToeSquare(Enum):
    Empty = 0
    X = 1
    O = 2  # noqa: E741

    @override
    def __str__(self) -> str:
        return "🟨❌⭕"[self.value]


class TicTacToe(MCTSState):
    def __init__(self, board: list[TicTacToeSquare] | None = None, X: bool = True):
        if not board:
            board = [TicTacToeSquare.Empty] * 9
        self.board = board
        self.X = X

    def wins(self, player: TicTacToeSquare) -> bool:
        for i in range(3):
            if all(self.board[i * 3 + j] == player for j in range(3)):
                return True
            if all(self.board[j * 3 + i] == player for j in range(3)):
                return True
        if all(self.board[i * 4] == player for i in range(3)):
            return True
        if all(self.board[i * 2 + 2] == player for i in range(3)):
            return True
        return False

    @override
    def next_states(self) -> list[Self]:
        states: list[MCTSState] = []
        for i, square in enumerate(self.board):
            if square == TicTacToeSquare.Empty:
                new_board = self.board.copy()
                new_board[i] = TicTacToeSquare.X if self.X else TicTacToeSquare.O
                states.append(TicTacToe(new_board, not self.X))
        return cast(list[Self], states)

    def random_next_state(self) -> Self:
        empty_indices = [i for i, square in enumerate(self.board) if square == TicTacToeSquare.Empty]
        new_board = self.board.copy()
        new_board[choice(empty_indices)] = TicTacToeSquare.X if self.X else TicTacToeSquare.O
        return cast(Self, TicTacToe(new_board, not self.X))

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
            self.wins(TicTacToeSquare.X)
            or self.wins(TicTacToeSquare.O)
            or all(square != TicTacToeSquare.Empty for square in self.board)
        )

    @override
    def reward(self) -> float:
        if self.wins(TicTacToeSquare.X):
            return 1
        if self.wins(TicTacToeSquare.O):
            return -1
        return 0

    @override
    def reward_perspective(self, reward: object) -> float:
        assert isinstance(reward, float) or isinstance(reward, int)
        if self.X:
            return reward
        return -reward

    @override
    def __str__(self) -> str:
        return "\n".join(
            "".join(str(square) for square in self.board[i * 3 : i * 3 + 3])
            for i in range(3)
        )

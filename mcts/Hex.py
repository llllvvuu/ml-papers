from enum import Enum
from typing import Self, cast, override
from .MCTS import MCTSState


class HexColor(Enum):
    Empty = 0
    Blue = 1
    Red = 2

    @override
    def __str__(self) -> str:
        return "⬡BR"[self.value]


class Hex(MCTSState):
    def __init__(
        self,
        size: int,
        board: list[list[HexColor]] | None = None,
        blue_turn: bool = True,
    ):
        self.size = size
        if not board:
            board = [
                [HexColor.Empty for _ in range(self.size)] for _ in range(self.size)
            ]
        self.board = board
        self.blue_turn = blue_turn

    def wins(self, player: HexColor) -> bool:
        visited = [[False for _ in range(self.size)] for _ in range(self.size)]

        def dfs(row: int, col: int) -> bool:
            if (player == HexColor.Blue and col == self.size - 1) or (
                player == HexColor.Red and row == self.size - 1
            ):
                return True

            visited[row][col] = True
            directions = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (
                    0 <= new_row < self.size
                    and 0 <= new_col < self.size
                    and not visited[new_row][new_col]
                    and self.board[new_row][new_col] == player
                ):
                    if dfs(new_row, new_col):
                        return True

            return False

        if player == HexColor.Blue:
            for i in range(self.size):
                if self.board[i][0] == player and dfs(i, 0):
                    return True

        if player == HexColor.Red:
            for i in range(self.size):
                if self.board[0][i] == player and dfs(0, i):
                    return True

        return False

    @override
    def next_states(self) -> list[Self]:
        states: list[MCTSState] = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == HexColor.Empty:
                    new_board = [row.copy() for row in self.board]
                    new_board[row][col] = (
                        HexColor.Blue if self.blue_turn else HexColor.Red
                    )
                    states.append(Hex(self.size, new_board, not self.blue_turn))
        return cast(list[Self], states)

    @override
    def is_terminal(self) -> bool:
        return (
            self.wins(HexColor.Blue)
            or self.wins(HexColor.Red)
            or all(
                self.board[row][col] != HexColor.Empty
                for row in range(self.size)
                for col in range(self.size)
            )
        )

    @override
    def terminal_reward(self) -> float:
        if self.wins(HexColor.Blue):
            return 1
        if self.wins(HexColor.Red):
            return -1
        return 0

    @override
    def reward_perspective(self, reward: object) -> float:
        assert isinstance(reward, float) or isinstance(reward, int)
        if self.blue_turn:
            return reward
        return -reward

    @override
    def __str__(self) -> str:
        return "\n".join(
            str(i) + " " * (i + 1) + " ".join(str(piece) for piece in row)
            for i, row in enumerate(self.board)
        )

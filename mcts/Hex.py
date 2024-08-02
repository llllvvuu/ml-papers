from enum import Enum
import random
from typing import Self, cast, override
from .MCTS import MCTSState


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.size -= 1
        return True

    def get_size(self) -> int:
        return self.size

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


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
        self._is_terminal: bool | None = None
        self._reward: float | None = None
        self._blue_wins: bool | None = None
        self._red_wins: bool | None = None

    def wins(self, player: HexColor) -> bool:
        # XXX: slow un-amortized algo, since we haven't persisted any union-find from parent state
        if player == HexColor.Blue and self._blue_wins is not None:
            return self._blue_wins
        if player == HexColor.Red and self._red_wins is not None:
            return self._red_wins

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

    def distance_from_center(self, row: int, col: int) -> int:
        center = self.size // 2
        return abs(row - center) + abs(col - center)

    @override
    def next_states(self) -> list[Self]:
        # XXX: slow, no sharing between states. memcpy overhead could be reduced with bitboards
        moves: list[tuple[int, int, int]] = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == HexColor.Empty:
                    distance = self.distance_from_center(row, col)
                    moves.append((distance, row, col))

        moves.sort(key=lambda x: x[0])

        states: list[MCTSState] = []
        for _, row, col in moves:
            new_board = [row.copy() for row in self.board]
            new_board[row][col] = HexColor.Blue if self.blue_turn else HexColor.Red
            states.append(Hex(self.size, new_board, not self.blue_turn))

        return cast(list[Self], states)

    def union(
        self,
        board: list[list[HexColor]],
        blue_uf: UnionFind,
        red_uf: UnionFind,
        blue_turn: bool,
        row: int,
        col: int,
    ):
        if blue_turn:
            node = row * self.size + col
            if col == 0:
                _ = blue_uf.union(node, self.size * self.size)
            if col == self.size - 1:
                _ = blue_uf.union(node, self.size * self.size + 1)
            for dr, dc in [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.size and 0 <= new_col < self.size:
                    if board[new_row][new_col] == HexColor.Blue:
                        print("blue union of", row, col, "and", new_row, new_col)
                        _ = blue_uf.union(node, new_row * self.size + new_col)
        else:
            node = row * self.size + col
            if row == 0:
                _ = red_uf.union(node, self.size * self.size)
            if row == self.size - 1:
                _ = red_uf.union(node, self.size * self.size + 1)
            for dr, dc in [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.size and 0 <= new_col < self.size:
                    if board[new_row][new_col] == HexColor.Red:
                        print("red union of", row, col, "and", new_row, new_col)
                        _ = red_uf.union(node, new_row * self.size + new_col)

    @override
    def random_rollout(self) -> tuple[float, int]:
        """
        Iteratively mutates a copy of the board, rather than iteratively
        copying and applying the non-amortized
        `wins()`/`is_terminal()`/`reward()` methods.
        i.e., iteratively mutates a union-find data structure,
        rather than running DFS every iteration.
        """
        if self.is_terminal():
            return self.reward(), 0
        board = [row.copy() for row in self.board]
        blue_uf = UnionFind(self.size * self.size + 2)
        red_uf = UnionFind(self.size * self.size + 2)
        print("initializing", self)
        for row in range(self.size):
            for col in range(self.size):
                if board[row][col] == HexColor.Blue:
                    self.union(board, blue_uf, red_uf, True, row, col)
                elif board[row][col] == HexColor.Red:
                    self.union(board, blue_uf, red_uf, False, row, col)
        print("initialized")
        blue_turn = self.blue_turn
        empty = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if board[row][col] == HexColor.Empty
        ]
        random.shuffle(empty)
        depth = 0
        reward: float = 0.0
        while reward == 0.0:
            if red_uf.connected(self.size * self.size, self.size * self.size + 1):
                print("red wins")
                reward = -1.0
            if blue_uf.connected(self.size * self.size, self.size * self.size + 1):
                print("blue wins")
                reward = 1.0
            if reward != 0.0 or len(empty) == 0:
                print("breaking")
                break
            row, col = empty.pop()
            print("playing", row, col, ", turn:", "blue" if blue_turn else "red")
            self.union(board, blue_uf, red_uf, blue_turn, row, col)
            board[row][col] = HexColor.Blue if blue_turn else HexColor.Red
            blue_turn = not blue_turn
            depth += 1
        return reward, depth

    @override
    def is_terminal(self) -> bool:
        if self._is_terminal is None:
            self._is_terminal = (
                self.wins(HexColor.Blue)
                or self.wins(HexColor.Red)
                or all(
                    self.board[row][col] != HexColor.Empty
                    for row in range(self.size)
                    for col in range(self.size)
                )
            )
        return self._is_terminal

    @override
    def reward(self) -> float:
        if self._reward is None:
            if self.wins(HexColor.Blue):
                self._reward = 1.0
            elif self.wins(HexColor.Red):
                self._reward = -1.0
            else:
                self._reward = 0.0
        return self._reward

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

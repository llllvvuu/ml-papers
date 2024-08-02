import torch
from typing import Self, cast, override
from .expert_iteration import ExItState
from ..mcts.Hex import Hex as HexMCTS, HexColor


class Hex(HexMCTS, ExItState):
    @override
    def to_network_input(self) -> torch.Tensor:
        input_tensor = torch.zeros((self.size + 2) * (self.size + 2), 6)
        player = HexColor.Blue if self.blue_turn else HexColor.Red
        opponent = HexColor.Red if self.blue_turn else HexColor.Blue

        player_start: set[tuple[int, int]] = set()
        player_end: set[tuple[int, int]] = set()
        opponent_start: set[tuple[int, int]] = set()
        opponent_end: set[tuple[int, int]] = set()

        if self.blue_turn:
            player_start = {(i, 0) for i in range(self.size)}
            player_end = {(i, self.size - 1) for i in range(self.size)}
            opponent_start = {(0, i) for i in range(self.size)}
            opponent_end = {(self.size - 1, i) for i in range(self.size)}
            for i in range(self.size + 2):
                input_tensor[i * (self.size + 2)][0] = 1.0
                input_tensor[i * (self.size + 2)][2] = 1.0
                input_tensor[i * (self.size + 2) + self.size + 1][0] = 1.0
                input_tensor[i * (self.size + 2) + self.size + 1][4] = 1.0
                input_tensor[i][1] = 1.0
                input_tensor[i][3] = 1.0
                input_tensor[(self.size + 1) * (self.size + 2) + i][1] = 1.0
                input_tensor[(self.size + 1) * (self.size + 2) + i][5] = 1.0
        else:
            player_start = {(0, i) for i in range(self.size)}
            player_end = {(self.size - 1, i) for i in range(self.size)}
            opponent_start = {(i, 0) for i in range(self.size)}
            opponent_end = {(i, self.size - 1) for i in range(self.size)}
            for i in range(self.size + 2):
                input_tensor[i * (self.size + 2)][1] = 1.0
                input_tensor[i * (self.size + 2)][3] = 1.0
                input_tensor[i * (self.size + 2) + self.size + 1][1] = 1.0
                input_tensor[i * (self.size + 2) + self.size + 1][5] = 1.0
                input_tensor[i][0] = 1.0
                input_tensor[i][2] = 1.0
                input_tensor[(self.size + 1) * (self.size + 2) + i][0] = 1.0
                input_tensor[(self.size + 1) * (self.size + 2) + i][4] = 1.0

        def flood_fill(
            start_set: set[tuple[int, int]], color: HexColor
        ) -> set[tuple[int, int]]:
            visited: set[tuple[int, int]] = set()
            stack = list(start_set)
            while stack:
                row, col = stack.pop()
                if (row, col) not in visited and self.board[row][col] == color:
                    visited.add((row, col))
                    for dr, dc in [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            stack.append((nr, nc))
            return visited

        player_start_connected = flood_fill(player_start, player)
        player_end_connected = flood_fill(player_end, player)
        opponent_start_connected = flood_fill(opponent_start, opponent)
        opponent_end_connected = flood_fill(opponent_end, opponent)

        for row in range(self.size):
            for col in range(self.size):
                idx = (row + 1) * (self.size + 2) + (col + 1)
                if self.board[row][col] == player:
                    input_tensor[idx][0] = 1.0
                elif self.board[row][col] == opponent:
                    input_tensor[idx][1] = 1.0
                input_tensor[idx][2] = float((row, col) in player_start_connected)
                input_tensor[idx][3] = float((row, col) in opponent_start_connected)
                input_tensor[idx][4] = float((row, col) in player_end_connected)
                input_tensor[idx][5] = float((row, col) in opponent_end_connected)

        return input_tensor

    @override
    def from_network_output(self, output: int) -> Self:
        """
        WARNING: This does not check if `output` is a valid move,
        and will return an invalid state in that case.
        """
        row = output // self.size
        col = output % self.size
        new_board = [row.copy() for row in self.board]
        new_board[row][col] = HexColor.Blue if self.blue_turn else HexColor.Red
        return cast(Self, Hex(self.size, new_board, not self.blue_turn))

    @override
    def network_output_size(self) -> int:
        return self.size * self.size

    @override
    def reward_unperspective(self, reward: float) -> object:
        return self.reward_perspective(reward)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HexMCTS):
            return NotImplemented
        return (
            self.size == other.size
            and self.board == other.board
            and self.blue_turn == other.blue_turn
        )

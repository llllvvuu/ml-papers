from enum import Enum
from typing import Self, cast, override
from .MCTS import MCTSState


class ChessPiece(Enum):
    Empty = 0
    WhitePawn = 1
    WhiteKnight = 2
    WhiteBishop = 3
    WhiteRook = 4
    WhiteQueen = 5
    WhiteKing = 6
    BlackPawn = 7
    BlackKnight = 8
    BlackBishop = 9
    BlackRook = 10
    BlackQueen = 11
    BlackKing = 12

    @override
    def __str__(self) -> str:
        return "·♙♘♗♖♕♔♟♞♝♜♛♚"[self.value]


class Chess(MCTSState):
    ROWS = 8
    COLS = 8

    def __init__(
        self, board: list[list[ChessPiece]] | None = None, white_turn: bool = True
    ):
        if not board:
            board = self.initial_board()
        self.board = board
        self.white_turn = white_turn

    @staticmethod
    def initial_board() -> list[list[ChessPiece]]:
        board = [
            [ChessPiece.Empty for _ in range(Chess.COLS)] for _ in range(Chess.ROWS)
        ]

        for col in range(Chess.COLS):
            board[1][col] = ChessPiece.WhitePawn
            board[6][col] = ChessPiece.BlackPawn

        back_row: list[str] = [
            "Rook",
            "Knight",
            "Bishop",
            "Queen",
            "King",
            "Bishop",
            "Knight",
            "Rook",
        ]

        for col, piece in enumerate(back_row):
            board[0][col] = getattr(ChessPiece, f"White{piece}")
            board[7][col] = getattr(ChessPiece, f"Black{piece}")

        return board

    def is_in_check(self, white: bool) -> bool:
        king = ChessPiece.WhiteKing if white else ChessPiece.BlackKing
        king_pos = self.find_piece(king)
        if not king_pos:
            return False

        for row in range(self.ROWS):
            for col in range(self.COLS):
                piece = self.board[row][col]
                if piece != ChessPiece.Empty and (piece.value > 6) == white:
                    if self.is_valid_move((row, col), king_pos):
                        return True
        return False

    def find_piece(self, piece: ChessPiece) -> tuple[int, int] | None:
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if self.board[row][col] == piece:
                    return (row, col)
        return None

    def is_valid_move(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        start_row, start_col = start
        end_row, end_col = end
        piece = self.board[start_row][start_col]

        if piece == ChessPiece.Empty:
            return False

        if not (0 <= end_row < self.ROWS and 0 <= end_col < self.COLS):
            return False

        if (piece.value <= 6) == (
            self.board[end_row][end_col].value <= 6
            and self.board[end_row][end_col].value != 0
        ):
            return False

        # TODO: Implement move rules for each piece type

        return True

    @override
    def next_states(self) -> list[Self]:
        states: list[MCTSState] = []
        for start_row in range(self.ROWS):
            for start_col in range(self.COLS):
                piece = self.board[start_row][start_col]
                if piece != ChessPiece.Empty and (piece.value <= 6) == self.white_turn:
                    for end_row in range(self.ROWS):
                        for end_col in range(self.COLS):
                            if self.is_valid_move(
                                (start_row, start_col), (end_row, end_col)
                            ):
                                new_board = [row.copy() for row in self.board]
                                new_board[end_row][end_col] = new_board[start_row][
                                    start_col
                                ]
                                new_board[start_row][start_col] = ChessPiece.Empty
                                new_state = Chess(new_board, not self.white_turn)
                                if not new_state.is_in_check(self.white_turn):
                                    states.append(new_state)
        return cast(list[Self], states)

    @override
    def is_terminal(self) -> bool:
        return len(self.next_states()) == 0

    @override
    def terminal_reward(self) -> float:
        if self.is_in_check(self.white_turn):
            return -1 if self.white_turn else 1
        return 0

    @override
    def reward_perspective(self, reward: object) -> float:
        assert isinstance(reward, float) or isinstance(reward, int)
        if self.white_turn:
            return reward
        return -reward

    @override
    def __str__(self) -> str:
        return (
            "\n".join("".join(str(piece) for piece in row) for row in self.board)
        )

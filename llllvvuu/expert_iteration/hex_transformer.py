import torch
import torch.nn as nn
from typing import override


DEFAULT_DIM = {}
DEFAULT_NUM_LAYERS = {}


# TODO: Generalize RoPE to 2D
class HexPositionalEmbedding(nn.Module):
    EPSILON = 2.0 / 3

    def __init__(self, board_size: int, dim: int) -> None:
        super().__init__()
        self.positional_encoding = nn.Parameter(
            self.EPSILON * torch.randn(1, (board_size + 2) * (board_size + 2), dim)
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional_encoding


class HexTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            dim,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm(x)
        x = x + self.feed_forward(x)
        return self.norm(x)


class HexTransformer(nn.Module):
    INPUT_DIM = 6  # see Hex.to_network_input()

    def __init__(
        self,
        board_size: int,
        dim: int | None = None,
        nhead: int = 2,
        num_layers: int | None = None,
    ) -> None:
        if dim is None:
            dim = DEFAULT_DIM.get(board_size, 16)
        if num_layers is None:
            num_layers = DEFAULT_NUM_LAYERS.get(board_size, board_size)
        super().__init__()
        self.board_size = board_size
        self.dim = dim
        self.nhead = nhead
        self.num_layers = num_layers

        self.base = nn.Sequential(
            nn.Linear(self.INPUT_DIM, dim, bias=False),
            HexPositionalEmbedding(board_size, dim),
            nn.LayerNorm(dim),
            *(HexTransformerBlock(dim, nhead) for _ in range(num_layers)),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3),
            nn.Flatten(),
        )
        self.policy_softmax = nn.Softmax(dim=1)
        self.value_head = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Flatten(),
            nn.Linear((board_size + 2) * (board_size + 2), 1),
            nn.Tanh(),
        )

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input: (batch_size, (board_size + 2) * (board_size + 2), INPUT_DIM)
        Outputs: (batch_size, board_size * board_size), (batch_size, 1)
        """
        occupied = x.transpose(1, 2)
        occupied = occupied.reshape(
            -1, self.INPUT_DIM, self.board_size + 2, self.board_size + 2
        )
        occupied = (occupied[:, 0, 1:-1, 1:-1] + occupied[:, 1, 1:-1, 1:-1]).flatten(1)
        x = self.base(x)
        policy = x.transpose(1, 2)
        policy: torch.Tensor = policy.reshape(
            -1, self.dim, self.board_size + 2, self.board_size + 2
        )
        policy = self.policy_head(policy) - occupied * torch.finfo(torch.float16).max
        return self.policy_softmax(policy), self.value_head(x)

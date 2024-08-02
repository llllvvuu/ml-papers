import torch
import torch.nn as nn
from typing import override

class HexNetTransformerBlock(nn.Module):
    def __init__(self, d_embed: int, d_ff: int, nhead: int, dropout_attn: float = 0.1, dropout_ff: float = 0.5) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_embed, nhead, batch_first=True, dropout=dropout_attn)
        self.norm = nn.LayerNorm(d_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_embed, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout_ff),
            nn.Linear(d_ff, d_embed),
            nn.Dropout(dropout_ff),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm(x)
        x = x + self.feed_forward(x)
        return self.norm(x)

class HexNet(nn.Module):
    def __init__(self, board_size: int, nhead: int = 4, num_layers: int = 3) -> None:
        super().__init__()
        self.board_size = board_size
        self.input_size = 2 * board_size * board_size
        self.output_size = board_size * board_size + 1
        self.d_embed = 4 * board_size * board_size
        self.d_ff = self.d_embed
        self.nhead = nhead
        self.num_layers = num_layers

        self.embedding = nn.Sequential(
            nn.Linear(self.input_size, self.d_embed),
            nn.SiLU(),
        )
        self.transformer_blocks = nn.Sequential(*(
            HexNetTransformerBlock(self.d_embed, self.d_ff, nhead) for _ in range(num_layers)
        ))
        self.policy_head = nn.Linear(self.d_embed, self.output_size - 1)
        self.value_head = nn.Linear(self.d_embed, 1)

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, self.input_size)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer_blocks(x)
        x = x.squeeze(1)
        policy = torch.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

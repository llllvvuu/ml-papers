from typing import override
import torch
from torch import nn


DEFAULT_NUM_FILTERS = {
    4: 16,
}
DEFAULT_NUM_LAYERS = {
    4: 2,
}


class HexConvnet(nn.Module):
    INPUT_CHANNELS = 6  # see Hex.to_network_input()

    def __init__(
        self,
        board_size: int,
        num_filters: int | None = None,
        num_layers: int | None = None,
    ):
        if num_filters is None:
            num_filters = DEFAULT_NUM_FILTERS.get(board_size, 32)
        if num_layers is None:
            num_layers = DEFAULT_NUM_LAYERS.get(board_size, board_size)
        super().__init__()
        self.board_size = board_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.base = nn.Sequential(
            nn.Conv2d(self.INPUT_CHANNELS, num_filters, 3, padding=1),
            nn.SiLU(),
            nn.LayerNorm([num_filters, board_size + 2, board_size + 2]),
            *(
                num_layers
                * [
                    nn.Conv2d(num_filters, num_filters, 3, padding=1),
                    nn.SiLU(),
                    nn.LayerNorm([num_filters, board_size + 2, board_size + 2]),
                ]
            ),
            nn.Conv2d(num_filters, num_filters, 3),
            nn.SiLU(),
            nn.LayerNorm([num_filters, board_size, board_size]),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, 1),
            nn.Flatten(),
        )
        self.policy_softmax = nn.Softmax(dim=1)
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters * board_size * board_size, 1),
            nn.Tanh(),
        )

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input: (batch_size, (board_size + 2) * (board_size + 2), INPUT_CHANNELS)
        Outputs: (batch_size, board_size * board_size), (batch_size, 1)
        """
        x = x.transpose(1, 2)
        x = x.reshape(-1, self.INPUT_CHANNELS, self.board_size + 2, self.board_size + 2)
        occupied = (x[:, 0, 1:-1, 1:-1] + x[:, 1, 1:-1, 1:-1]).flatten(1)
        x = self.base(x)
        policy: torch.Tensor = self.policy_head(x)
        policy = policy - occupied * torch.finfo(torch.float16).max
        return self.policy_softmax(policy), self.value_head(x)

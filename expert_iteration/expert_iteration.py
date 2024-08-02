import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Protocol, Self, TypeVar, cast

from ..mcts.MCTS import MCTSNode, MCTSState


class ExItState(MCTSState, Protocol):
    def to_network_input(self) -> torch.Tensor: ...
    def from_network_output(self, output: int) -> Self: ...
    def network_output_size(self) -> int: ...


TState = TypeVar("TState", bound=ExItState)


class ExpertIteration[TState]:
    def __init__(
        self,
        initial_state: TState,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.initial_node = MCTSNode(initial_state, None)
        self.node = MCTSNode(initial_state, None)
        self.optimizer = optimizer or optim.AdamW(model.parameters())
        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = model.to(self.device)

    def train_apprentice(
        self,
    ) -> tuple[
        int, list[float], float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        expert_policy, expert_value = self.expert()
        _ = self.model.train()
        self.optimizer.zero_grad()
        state = cast(ExItState, self.node.state).to_network_input().to(self.device)
        apprentice_policy, apprentice_value = cast(
            tuple[torch.Tensor, torch.Tensor], self.model(state)
        )
        policy_loss = nn.functional.cross_entropy(
            apprentice_policy.flatten(),
            torch.FloatTensor(expert_policy).to(self.device),
        )
        value_loss = nn.functional.mse_loss(
            apprentice_value.flatten(),
            torch.FloatTensor([expert_value]).to(self.device),
        )
        _ = (policy_loss + value_loss).backward()
        _ = self.optimizer.step()
        sampled_action = random.choices(
            range(len(expert_policy)), weights=expert_policy, k=1
        )[0]
        self.node = self._get_child(sampled_action) or self.initial_node
        if cast(ExItState, self.node.state).is_terminal():
            self.node = self.initial_node
        return (
            sampled_action,
            expert_policy,
            expert_value,
            apprentice_policy,
            apprentice_value,
            policy_loss,
            value_loss,
        )

    def expert(self, n_sims: int = 1000) -> tuple[list[float], float]:
        # TODO: bias the tree policy using the apprentice policy
        # \frac{1}{n_i} \sum_{t=1}^{n_i} r_t + c_{\text{UCT}} \sqrt{\frac{\ln N}{n_i}} + c_{\pi} \frac{\pi(a_i | s)}{n_i + 1}
        # https://duvenaud.github.io/learning-to-search/slides/week3/MCTSintro.pdf
        _ = self.model.eval()
        node = self.node
        for _ in range(n_sims):
            node.run()
        policy: list[float] = []
        for i in range(cast(ExItState, node.state).network_output_size()):
            child = self._get_child(i)
            policy.append(child.visits / node.visits if child else 0)
        return policy, node.reward / node.visits

    def best_move(self, state: TState) -> float:
        # TODO: use expert
        _ = self.model.eval()
        with torch.no_grad():
            input = cast(ExItState, state).to_network_input().to(self.device)
            return cast(torch.Tensor, self.model(input)).max().item()

    def _get_child(self, i: int) -> MCTSNode[TState] | None:
        state = cast(ExItState, self.node.state).from_network_output(i)
        return next(
            (child for child in self.node.children if child.state == state), None
        )

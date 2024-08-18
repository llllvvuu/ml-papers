from enum import Enum
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import NamedTuple, Protocol, Self, TypeVar, cast

from ..mcts.MCTS import MCTSNode, MCTSState


class ExItState(MCTSState, Protocol):
    def to_network_input(self) -> torch.Tensor: ...
    def from_network_output(self, output: int) -> Self: ...
    def network_output_size(self) -> int: ...
    def reward_unperspective(self, reward: float) -> object:
        """
        Converts from e.g. "player wins" / "opponent wins" to e.g. "blue wins" / "red wins"
        """


class PlayerType(Enum):
    """
    This wasn't in original expert iteration nor AlphaZero. In original ExIt,
    all games are played between two copies of the latest apprentice. In AlphaZero,
    all games are played between two copies of the latest expert.
    Similar to ExIt, we exclude the expert from the set of players to
    save on computation. However, we also include more unpredictable players
    in the hopes of helping the apprentice generalize to more states.
    """

    APPRENTICE = 0
    EPSILON_SOFT_APPRENTICE = 1
    EPSILON_GREEDY_APPRENTICE = 2
    RANDOM = 3
    MCTS = 4


class Step(NamedTuple):
    state_tensor: torch.Tensor
    state: ExItState
    action: int
    apprentice_value: float | None


class Experience(NamedTuple):
    state_tensor: torch.Tensor
    policy_target: list[float]
    value_target: float


class ExperienceDiagnostics(NamedTuple):
    player1: PlayerType
    player2: PlayerType
    state: ExItState
    action: int
    expert_action: int
    expert_policy: list[float]
    apprentice_value: float | None
    expert_value: float
    value_target: float


class TrainingDiagnostics(NamedTuple):
    batch_size: int
    avg_policy_loss: float
    avg_value_loss: float


class ExpertEvaluation(NamedTuple):
    policy: list[float]
    value: float


TState = TypeVar("TState", bound=ExItState)


class ExpertIteration[TState]:
    EPSILON = 0.1
    PLAYER_PROBABILITIES = [0.6, 0.1, 0.1, 0.1, 0.1]

    def __init__(
        self,
        initial_state: TState,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        device: torch.device | None = None,
        # state_tensor, Q, terminal_value
        replay_buffer: list[Experience] | None = None,
    ) -> None:
        self.initial_state = initial_state
        self.optimizer = optimizer or optim.AdamW(model.parameters())
        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = model.to(self.device)
        self.replay_buffer = replay_buffer or []

    def experience(
        self, use_policy_network: bool, use_value_network: bool
    ) -> ExperienceDiagnostics:
        """
        Play one game and add one random state from that game to the replay buffer.
        This is similar to ExIt and dissimilar to AlphaZero,
        which adds all states from a game to the replay buffer.
        """
        player1 = random.choices(
            list(PlayerType), weights=self.PLAYER_PROBABILITIES, k=1
        )[0]
        player2 = random.choices(
            list(PlayerType), weights=self.PLAYER_PROBABILITIES, k=1
        )[0]
        if not use_policy_network:
            player1 = PlayerType.MCTS
            player2 = PlayerType.MCTS
        episode = self.play_game(player1, player2)
        state_tensor, state, action, apprentice_value = random.choice(episode)
        expert_policy, expert_value = self.expert(
            MCTSNode(cast(TState, state), None),
            2000 if use_policy_network else 50000,
            use_policy_network,
            use_value_network,
        )
        expert_action = expert_policy.index(max(expert_policy))
        value_target = self.greedy_rollout(state, use_policy_network)
        self.replay_buffer.append(Experience(state_tensor, expert_policy, value_target))
        return ExperienceDiagnostics(
            player1,
            player2,
            state,
            action,
            expert_action,
            expert_policy,
            apprentice_value,
            expert_value,
            value_target,
        )

    def play_game(self, player1: PlayerType, player2: PlayerType) -> list[Step]:
        state = cast(ExItState, self.initial_state)
        episode: list[Step] = []
        player1_turn = True
        _ = self.model.eval()
        with torch.no_grad():
            while not state.is_terminal():
                player = player1 if player1_turn else player2
                state_tensor = state.to_network_input()
                apprentice_value = None
                if player == PlayerType.MCTS:
                    exploration_policy = self.expert(
                        MCTSNode(cast(TState, state), None), 1000, False, False
                    )[0]
                elif player == PlayerType.RANDOM or (
                    random.random() < self.EPSILON
                    and (
                        player == PlayerType.EPSILON_GREEDY_APPRENTICE
                        or player == PlayerType.EPSILON_SOFT_APPRENTICE
                    )
                ):
                    next_states = state.next_states()
                    exploration_policy = [
                        float(state.from_network_output(i) in next_states)
                        for i in range(state.network_output_size())
                    ]
                else:
                    policy, value = cast(
                        tuple[torch.Tensor, torch.Tensor],
                        self.model(state_tensor.unsqueeze(0).to(self.device)),
                    )
                    apprentice_value = cast(float, value.item())
                    exploration_policy = cast(list[float], policy.flatten().tolist())
                if player == PlayerType.EPSILON_GREEDY_APPRENTICE:
                    action = exploration_policy.index(max(exploration_policy))
                else:
                    action = random.choices(
                        range(len(exploration_policy)), weights=exploration_policy, k=1
                    )[0]
                episode.append(Step(state_tensor, state, action, apprentice_value))
                state = state.from_network_output(action)
                player1_turn = not player1_turn
        return episode

    def greedy_rollout(self, state: ExItState, use_policy_network: bool) -> float:
        greedy_state = state
        while not greedy_state.is_terminal():
            if use_policy_network:
                greedy_state_tensor = greedy_state.to_network_input()
                policy = cast(
                    torch.Tensor,
                    self.model(greedy_state_tensor.unsqueeze(0).to(self.device))[0],
                )
                exploration_policy = cast(list[float], policy.flatten().tolist())
            else:
                exploration_policy = self.expert(
                    MCTSNode(cast(TState, greedy_state), None), 1000, False, False
                )[0]
            greedy_action = exploration_policy.index(max(exploration_policy))
            greedy_state = greedy_state.from_network_output(greedy_action)
        return state.reward_perspective(greedy_state.reward())

    def experience_replay(self, batch_size: int, n_batches: int) -> TrainingDiagnostics:
        batch_size = min(len(self.replay_buffer), batch_size)
        total_policy_loss, total_value_loss = 0.0, 0.0
        _ = self.model.train()
        for _ in range(n_batches):
            batch = random.sample(self.replay_buffer, batch_size)
            state_tensors = torch.stack([state_tensor for state_tensor, _, _ in batch])
            policy_targets = torch.FloatTensor([policy for _, policy, _ in batch])
            value_targets = torch.FloatTensor([[value] for _, _, value in batch])

            self.optimizer.zero_grad()
            apprentice_policy, apprentice_value = cast(
                tuple[torch.Tensor, torch.Tensor],
                self.model(state_tensors.to(self.device)),
            )
            policy_loss = nn.functional.cross_entropy(
                apprentice_policy.flatten(1),
                policy_targets.to(self.device),
            )
            value_loss = nn.functional.mse_loss(
                apprentice_value.flatten(1),
                value_targets.to(self.device),
            )
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            _ = (policy_loss + value_loss).backward()
            _ = self.optimizer.step()
        return TrainingDiagnostics(
            batch_size, total_policy_loss / n_batches, total_value_loss / n_batches
        )

    def policy_improvement(
        self,
        policy: list[float],
        Q: list[float],
        n_sims: int,
        c_puct: float = 5.0,
    ) -> list[float]:
        """
        Implementation of $\\overline{\\pi}$ from
        "Monte-Carlo tree search as regularized policy optimization":
        https://arxiv.org/pdf/2007.12509

        Currently unused.
        """
        lambda_N = c_puct / math.sqrt(n_sims)
        a_min = max(Q[i] + lambda_N * policy[i] for i in range(len(policy)))
        a_max = max(Q[i] for i in range(len(policy))) + lambda_N
        improved_policy = policy
        for _ in range(32):
            a = (a_min + a_max) / 2
            improved_policy = [
                lambda_N * policy[i] / (a - Q[i]) for i in range(len(policy))
            ]
            if sum(improved_policy) <= 1:
                a_max = a
            else:
                a_min = a
        return improved_policy

    @staticmethod
    def ExIt_UCT(
        node: MCTSNode[TState], child: MCTSNode[TState], policy: float, n_sims: int
    ) -> float:
        return (
            child.reward / child.visits
            + math.sqrt(2 * math.log(node.visits) / child.visits)
            + (n_sims / len(node.children)) * policy / child.visits
        )

    @staticmethod
    def AlphaZero_PUCT(
        node: MCTSNode[TState], child: MCTSNode[TState], policy: float, n_sims: int
    ) -> float:
        """
        Currently unused.

        This is probably unusable without adding Dirichlet noise,
        since unlike `ExIt_UCT` which adds the prior to the exploration term,
        `AlphaZero_PUCT` multiples the two, so a prior of 0 will
        wipe out the exploration term.
        """
        c_puct = 2.5 * math.sqrt(n_sims / 800)
        return (
            child.reward / child.visits
            + c_puct * math.sqrt(node.visits) * policy / child.visits
        )

    def tree_policy(self, node: MCTSNode[TState], n_sims: int) -> MCTSNode[TState]:
        """Adds a bias of `w_a * policy[i] / visits[i]` to the UCT."""
        state = cast(ExItState, node.state)
        if len(node.unexplored) != 0 or len(node.children) == 0 or state.is_terminal():
            return node
        if getattr(node, "_policy", None) is None:
            input = state.to_network_input().to(self.device).unsqueeze(0)
            node._policy = cast(list[float], self.model(input)[0].flatten().tolist())
        policy = cast(list[float], node._policy)
        best_child = node.children[0]
        best_uct = float("-inf")
        for i in range(state.network_output_size()):
            child = self._get_child(node, i)
            if child is not None:
                uct = self.ExIt_UCT(node, child, policy[i], n_sims)
                if uct > best_uct:
                    best_uct = uct
                    best_child = child
        return self.tree_policy(best_child, n_sims)

    def expert(
        self,
        node: MCTSNode[TState],
        n_sims: int,
        use_policy_network: bool,
        use_value_network: bool,
    ) -> ExpertEvaluation:
        _ = self.model.eval()
        with torch.no_grad():
            for _ in range(n_sims):
                if use_policy_network:
                    child = self.tree_policy(node, n_sims).expand()
                else:
                    child = node.select().expand()
                if use_value_network and not cast(ExItState, child.state).is_terminal():
                    input = (
                        cast(ExItState, child.state)
                        .to_network_input()
                        .to(self.device)
                        .unsqueeze(0)
                    )
                    apprentice_policy, apprentice_reward = cast(
                        tuple[torch.Tensor, torch.Tensor], self.model(input)
                    )
                    policy = cast(list[float], apprentice_policy.flatten().tolist())
                    child._policy = policy
                    state = cast(ExItState, child.state)
                    greedy_child = state.from_network_output(policy.index(max(policy)))
                    reward, depth = greedy_child.random_rollout()
                    depth += 1
                    network_reward, network_depth = reward, depth
                    if not greedy_child.is_terminal():
                        network_reward, network_depth = (
                            state.reward_unperspective(apprentice_reward.item()),
                            0,
                        )
                else:
                    reward, depth = child.simulate()
                    network_reward, network_depth = reward, depth
                child.backpropagate(reward, depth)
                child.backpropagate(network_reward, network_depth)
        policy: list[float] = []
        for i in range(cast(ExItState, node.state).network_output_size()):
            child = self._get_child(node, i)
            policy.append(child.visits / node.visits if child else 0)
        return ExpertEvaluation(policy, node.reward / node.visits)

    @staticmethod
    def recalculate_expert_policy(rewards: list[float], n_sims: int) -> list[float]:
        """
        Currently unused.

        This is an experiment that wasn't present in ExIt nor AlphaZero.
        The idea is to remove the apprentice bias from the expert policy
        before putting it into the replay buffer.

        This avoids regularizing towards old policies, but it also makes it
        an unregularized policy update altogether, so it's not necessarily good.
        """
        visits = [float(r != float("-inf")) for r in rewards]
        n = len(list(filter(lambda v: v > 0, visits)))
        while n < n_sims:
            n += 1
            i = max(
                range(len(rewards)),
                key=lambda i: rewards[i]
                + (math.sqrt(2 * math.log(n) / visits[i]) if visits[i] > 0 else 0),
            )
            visits[i] += 1
        return [v / n for v in visits]

    def apprentice_move(self, state: TState) -> tuple[int, float, list[float]]:
        _ = self.model.eval()
        with torch.no_grad():
            input = (
                cast(ExItState, state).to_network_input().to(self.device).unsqueeze(0)
            )
            policy, value = cast(tuple[torch.Tensor, torch.Tensor], self.model(input))
            return int(policy.argmax().item()), value.item(), policy.flatten().tolist()

    def expert_move(
        self, state: TState, n_sims: int = 2000
    ) -> tuple[int, float, list[float]]:
        policy, value = self.expert(MCTSNode(state, None), n_sims, True, True)
        return policy.index(max(policy)), value, policy

    @staticmethod
    def _get_child(node: MCTSNode[TState], i: int) -> MCTSNode[TState] | None:
        state = cast(ExItState, node.state).from_network_output(i)
        return next((child for child in node.children if child.state == state), None)

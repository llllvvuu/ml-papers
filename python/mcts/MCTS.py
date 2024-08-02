import math
from typing import Protocol, Self, TypeVar, cast


class MCTSState(Protocol):
    def next_states(self) -> list[Self]: ...
    def random_rollout(self) -> tuple[float, int]:
        """returns: reward, depth"""
        ...

    # TODO: Replace `next_states()` and `random_rollout()` with these,
    # for efficiency (current impl wastes a lot of allocations eagerly creating states)
    # def legal_actions(self) -> list[int]: ...
    # def act(self, action: int) -> Self: ...
    # def act_in_place(self, action: int) -> None: ...
    # def random_rollout_in_place(self) -> tuple[float, int]: ...

    def is_terminal(self) -> bool: ...
    def default_parent_reward_perspective(self, reward: object) -> float:
        """
        Given e.g. "blue wins" or "red wins", determine e.g. whether
        the last player wins.
        """
        ...

    def reward(self) -> object: ...

    def reward_perspective(self, reward: object) -> float:
        """
        Given e.g. "blue wins" or "red wins", determine e.g. whether
        the upcoming player wins.
        """
        ...


TState = TypeVar("TState", bound=MCTSState)


class MCTSNode[TState]:
    def __init__(self, state: TState, parent: "MCTSNode[TState] | None"):
        self.state = state
        self.parent = parent
        self.children: list[MCTSNode[TState]] = []
        self.unexplored = cast(list[TState], cast(MCTSState, state).next_states())
        self.visits: int = 0
        self.reward: float = 0

    def UCT(self, c: float = 1.4) -> float:
        if self.parent is None or self.visits == 0 or self.parent.visits == 0:
            return 0
        return self.reward / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self) -> "MCTSNode[TState] | None":
        return max(
            self.children,
            default=None,
            key=lambda x: x.reward / x.visits,
        )

    def select(self, c: float = 1.4) -> "MCTSNode[TState]":
        if (
            len(self.unexplored) != 0
            or len(self.children) == 0
            or cast(MCTSState, self.state).is_terminal()
        ):
            return self
        return max(self.children, key=lambda x: x.UCT(c)).select(c)

    def expand(self) -> "MCTSNode[TState]":
        if len(self.unexplored) == 0:
            return self
        child = MCTSNode(self.unexplored.pop(), self)
        self.children.append(child)
        return child

    def simulate(self) -> tuple[object, int]:
        return cast(MCTSState, self.state).random_rollout()

    def backpropagate(
        self,
        reward: object,
        depth: int = 0,
        log_discount: float = 0,
    ):
        self.visits += 1
        if self.parent is None:
            self.reward += cast(
                MCTSState, self.state
            ).default_parent_reward_perspective(reward) * math.exp(
                -log_discount * depth
            )
        if self.parent is not None:
            self.reward += cast(MCTSState, self.parent.state).reward_perspective(
                reward
            ) * math.exp(-log_discount * depth)
            self.parent.backpropagate(
                reward,
                depth + 1,
                log_discount,
            )

    def run(self, c: float = 1.4, log_discount: float = 0) -> None:
        node = self.select(c).expand()
        node.backpropagate(*node.simulate(), log_discount)

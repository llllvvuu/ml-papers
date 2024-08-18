import torch
import math
from pathlib import Path
from tqdm import tqdm
import logging
import csv

from .expert_iteration import ExItState, Experience, ExpertIteration


torch.serialization.add_safe_globals([Experience])


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: list[tuple[torch.Tensor, list[float], torch.Tensor]],
    policy_loss: float,
    value_loss: float,
    iteration: int,
):
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "replay_buffer": replay_buffer,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        },
        path,
    )
    logging.info(f"Checkpoint saved to {path}")


def train(
    model: torch.nn.Module,
    initial_state: ExItState,
    output_path: str,
    iters: int,
    checkpoint_path_str: str | None,
    loss_csv_path: str | None,
) -> None:
    logging.info(
        f"Training model with {sum(p.numel() for p in model.parameters())} parameters"
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        # XXX: https://github.com/pytorch/pytorch/issues/132596
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    start_iter = 0
    replay_buffer = None
    if checkpoint_path_str:
        checkpoint_path = Path(checkpoint_path_str)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            _ = model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_iter: int = checkpoint["iteration"]
            replay_buffer = checkpoint["replay_buffer"]
            logging.info(f"Loaded checkpoint from {checkpoint_path_str}")
        else:
            logging.info(
                f"Checkpoint file {checkpoint_path_str} not found. Starting with a fresh model."
            )
    expert_iteration = ExpertIteration(
        initial_state, model, optimizer, device, replay_buffer
    )

    if loss_csv_path:
        csv_file = open(loss_csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["iter", "policy_loss", "value_loss"])

    avg_policy_loss, avg_value_loss = 0, 0
    avg_count = 0

    for i in tqdm(range(start_iter, start_iter + iters)):
        use_policy_network = i > 500
        use_value_network = i > 1000
        (
            player1,
            player2,
            state,
            explored_action,
            expert_action,
            expert_policy,
            apprentice_value,
            expert_value,
            terminal_value,
        ) = expert_iteration.experience(use_policy_network, use_value_network)
        batch_size, policy_loss, value_loss = expert_iteration.experience_replay(64, 32)

        avg_policy_loss += policy_loss
        avg_value_loss += value_loss
        avg_count += 1

        logging.info(
            f"Iteration {i+1}: Using Policy Network = {use_policy_network}, Using Value Network = {use_value_network}"
        )
        logging.info(f"Iteration {i+1}: Players = {player1}, {player2}")
        logging.info(
            f"Iteration {i+1}: Batch Size = {batch_size}, Dataset Size = {len(expert_iteration.replay_buffer)}"
        )
        expert_entropy = sum(-p * math.log(p) if p > 0 else 0 for p in expert_policy)
        logging.info(f"Iteration {i+1}: Expert Policy Entropy = {expert_entropy}")
        logging.info(
            f"Iteration {i+1}: Apprentice Value = {apprentice_value}, Expert Value = {expert_value}, Terminal Value = {terminal_value}"
        )
        logging.info(
            f"Iteration {i+1}: Explored Action = {explored_action}, Expert Action = {expert_action}"
        )
        logging.info(
            f"Iteration {i+1}: Policy Loss = {policy_loss}, Value Loss = {value_loss}"
        )
        logging.info("\n" + str(state))

        if (i + 1 - start_iter) % 25 == 0:
            avg_policy_loss /= avg_count
            avg_value_loss /= avg_count
            save_checkpoint(
                output_path,
                model,
                optimizer,
                expert_iteration.replay_buffer,
                avg_policy_loss,
                avg_value_loss,
                i,
            )
            print(f"Checkpoint saved to {output_path}")
            if loss_csv_path:
                csv_writer.writerow([i + 1, avg_policy_loss, avg_value_loss])
                csv_file.flush()
                print(f"Losses saved to {loss_csv_path}")

            avg_policy_loss, avg_value_loss = 0, 0
            avg_count = 0

    if avg_count > 0:
        avg_policy_loss /= avg_count
        avg_value_loss /= avg_count
        save_checkpoint(
            output_path,
            model,
            optimizer,
            expert_iteration.replay_buffer,
            avg_policy_loss,
            avg_value_loss,
            start_iter + iters,
        )
        print(f"Checkpoint saved to {output_path}")
        if loss_csv_path:
            csv_writer.writerow([start_iter + iters, avg_policy_loss, avg_value_loss])
            csv_file.flush()
            print(f"Losses saved to {loss_csv_path}")

    if loss_csv_path:
        csv_file.close()

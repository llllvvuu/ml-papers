import torch
from pathlib import Path
from tqdm import tqdm
import logging

from .expert_iteration import ExItState, ExpertIteration


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    policy_loss: float,
    value_loss: float,
    iteration: int,
):
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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
) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    start_iter = 0
    if checkpoint_path_str:
        checkpoint_path = Path(checkpoint_path_str)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            _ = model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_iter: int = checkpoint["iteration"]
            logging.info(f"Loaded checkpoint from {checkpoint_path_str}")
        else:
            logging.info(
                f"Checkpoint file {checkpoint_path_str} not found. Starting with a fresh model."
            )
    expert_iteration = ExpertIteration(initial_state, model, optimizer, device)
    policy_loss, value_loss = 0, 0
    for i in tqdm(range(start_iter, start_iter + iters)):
        (
            sampled_action,
            expert_policy,
            expert_value,
            apprentice_policy,
            apprentice_value,
            policy_loss,
            value_loss,
        ) = expert_iteration.train_apprentice()
        policy_loss, value_loss = policy_loss.item(), value_loss.item()
        logging.info(f"Iteration {i+1}: Sampled Action = {sampled_action}")
        logging.info(
            f"Iteration {i+1}: Expert Policy = {expert_policy}, Expert Value = {expert_value}"
        )
        logging.info(
            f"Iteration {i+1}: Apprentice Policy = {apprentice_policy}, Apprentice Value = {apprentice_value}"
        )
        logging.info(
            f"Iteration {i+1}: Policy Loss = {policy_loss}, Value Loss = {value_loss}"
        )
        if i % 25 == 0:
            save_checkpoint(output_path, model, optimizer, policy_loss, value_loss, i)
    save_checkpoint(
        output_path, model, optimizer, policy_loss, value_loss, start_iter + iters
    )

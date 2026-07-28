from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExactStateCarry(nn.Module):
    """Projection-free recurrent language model on the unit hypersphere.

    The state dimension equals the vocabulary size. Each state coordinate
    corresponds directly to one token coordinate, so the normalized hidden
    state is also the prediction vector.
    """

    def __init__(self, state_size: int, eps: float = 1.0e-8) -> None:
        super().__init__()
        if state_size <= 1:
            raise ValueError("state_size must be greater than 1")

        self.state_size = state_size
        self.eps = eps

        self.W_f = nn.Linear(state_size, state_size, bias=True)
        self.U_f = nn.Linear(state_size, state_size, bias=False)
        self.W_i = nn.Linear(state_size, state_size, bias=True)
        self.U_i = nn.Linear(state_size, state_size, bias=False)

        nn.init.xavier_uniform_(self.W_f.weight)
        nn.init.xavier_uniform_(self.U_f.weight)
        nn.init.zeros_(self.W_f.bias)

        nn.init.xavier_uniform_(self.W_i.weight)
        nn.init.orthogonal_(self.U_i.weight)
        nn.init.zeros_(self.W_i.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, time]")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long dtype")

        batch_size, time_steps = token_ids.shape
        dtype = self.W_f.weight.dtype
        device = token_ids.device

        if h_prev is None:
            h_prev = torch.zeros(
                batch_size,
                self.state_size,
                dtype=dtype,
                device=device,
            )
        elif h_prev.shape != (batch_size, self.state_size):
            raise ValueError(
                f"h_prev must have shape {(batch_size, self.state_size)}"
            )

        x_sequence = F.one_hot(
            token_ids,
            num_classes=self.state_size,
        ).to(dtype=dtype)

        states: list[torch.Tensor] = []

        for timestep in range(time_steps):
            x_t = x_sequence[:, timestep]

            gate = torch.sigmoid(
                self.W_f(x_t) + self.U_f(h_prev)
            )
            content = torch.tanh(
                self.W_i(x_t) + self.U_i(h_prev)
            )

            q_t = h_prev + gate * content
            h_prev = q_t / torch.linalg.vector_norm(
                q_t,
                ord=2,
                dim=-1,
                keepdim=True,
            ).clamp_min(self.eps)

            states.append(h_prev)

        return torch.stack(states, dim=1), h_prev


def exact_cosine_loss(
    targets: torch.Tensor,
    states: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Cosine loss against one-hot target vectors.

    With unit states and one-hot unit targets, this is exactly
    1 - states[..., target_id], averaged over batch and time.
    """
    if targets.shape != states.shape[:2]:
        raise ValueError("targets must match states batch/time dimensions")

    target_vectors = F.one_hot(
        targets,
        num_classes=states.shape[-1],
    ).to(states.dtype)

    target_norm = torch.linalg.vector_norm(
        target_vectors,
        ord=2,
        dim=-1,
    ).clamp_min(eps)

    state_unit = states / torch.linalg.vector_norm(
        states,
        ord=2,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)

    cosine = (
        target_vectors * state_unit
    ).sum(dim=-1) / target_norm

    return (1.0 - cosine).mean()

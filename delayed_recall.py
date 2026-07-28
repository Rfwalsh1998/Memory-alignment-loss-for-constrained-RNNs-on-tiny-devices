#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exact_state_carry import ExactStateCarry, exact_cosine_loss


ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXY"
    "0123456789"
    " .,;:!?-/()[]{}"
    "\n"
)
STATE_SIZE = 77
assert len(ALPHABET) == STATE_SIZE
assert len(set(ALPHABET)) == STATE_SIZE

STOI = {character: index for index, character in enumerate(ALPHABET)}
ITOS = list(ALPHABET)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic delayed-key benchmark for Exact State Carry."
    )
    parser.add_argument("--train-seconds", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--train-records", type=int, default=10000)
    parser.add_argument("--validation-records", type=int, default=2000)
    parser.add_argument("--validation-chunks", type=int, default=32)
    parser.add_argument("--probe-horizon", type=int, default=100000)
    return parser.parse_args()


def make_delayed_trash(
    records: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    trash_words = [
        "zibble", "mork", "talven", "quux", "brindle", "narp",
        "vek", "sprocket", "lum", "drindle", "kex", "womble",
        "florp", "ziggle", "wibble", "snorp", "glimmer", "plonk",
        "azure", "violet", "silent", "amber", "oblique", "hollow",
    ]

    characters: list[str] = []
    delayed_mask: list[bool] = []

    def append(value: str, marked: bool = False) -> None:
        characters.extend(value)
        delayed_mask.extend([marked] * len(value))

    append(ALPHABET)
    append(ALPHABET[::-1])

    for record in range(records):
        key = f"{rng.randrange(10000):04d}"
        filler_words = rng.randrange(18, 60)

        append(f"packet[{key}] ")

        for position in range(filler_words):
            word = trash_words[
                (
                    rng.randrange(len(trash_words))
                    + record
                    + position * 7
                ) % len(trash_words)
            ]
            append(word)
            append("; " if position % 9 == 8 else " ")

        append("return[")
        append(key, marked=True)
        append("]\n")

    text = "".join(characters)
    unknown = sorted(set(text) - set(ALPHABET))
    if unknown:
        raise ValueError(f"Unexpected characters: {unknown!r}")

    encoded = torch.tensor(
        [STOI[character] for character in text],
        dtype=torch.long,
    )
    mask = torch.tensor(delayed_mask, dtype=torch.bool)
    return encoded, mask


class StreamBatcher:
    def __init__(
        self,
        data: torch.Tensor,
        delayed_mask: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        stream_length = len(data) // batch_size
        usable = stream_length * batch_size
        self.tokens = data[:usable].view(batch_size, stream_length)
        self.delayed = delayed_mask[:usable].view(batch_size, stream_length)
        self.sequence_length = sequence_length
        self.steps = (stream_length - 1) // sequence_length

    def chunks(self):
        for step in range(self.steps):
            start = step * self.sequence_length
            end = start + self.sequence_length
            yield (
                self.tokens[:, start:end],
                self.tokens[:, start + 1:end + 1],
                self.delayed[:, start + 1:end + 1],
            )


@torch.inference_mode()
def evaluate(
    model: ExactStateCarry,
    batches: StreamBatcher,
    max_chunks: int,
) -> dict[str, float | int | None]:
    model.eval()
    hidden = None
    loss_sum = 0.0
    accuracy_sum = 0.0
    delayed_correct = 0
    delayed_total = 0
    delayed_coordinate_sum = 0.0
    maximum_norm_error = 0.0
    chunks = 0

    for inputs, targets, delayed_mask in batches.chunks():
        states, hidden = model(inputs, hidden)
        hidden = hidden.detach()

        loss_sum += exact_cosine_loss(targets, states).item()
        predictions = states.argmax(dim=-1)
        accuracy_sum += (
            predictions == targets
        ).float().mean().item()

        if bool(delayed_mask.any()):
            delayed_correct += int(
                (
                    predictions[delayed_mask]
                    == targets[delayed_mask]
                ).sum()
            )
            delayed_total += int(delayed_mask.sum())

            target_coordinates = states.gather(
                dim=-1,
                index=targets.unsqueeze(-1),
            ).squeeze(-1)
            delayed_coordinate_sum += float(
                target_coordinates[delayed_mask].sum()
            )

        norms = torch.linalg.vector_norm(states, ord=2, dim=-1)
        maximum_norm_error = max(
            maximum_norm_error,
            float((norms - 1.0).abs().max()),
        )

        chunks += 1
        if chunks >= max_chunks:
            break

    model.train()
    return {
        "validation_loss": loss_sum / chunks,
        "validation_accuracy": accuracy_sum / chunks,
        "delayed_key_accuracy": (
            delayed_correct / delayed_total
            if delayed_total else None
        ),
        "delayed_key_mean_target_coordinate": (
            delayed_coordinate_sum / delayed_total
            if delayed_total else None
        ),
        "delayed_characters_scored": delayed_total,
        "maximum_norm_error": maximum_norm_error,
    }


def encode(text: str) -> torch.Tensor:
    return torch.tensor(
        [[STOI[character] for character in text]],
        dtype=torch.long,
    )


@torch.inference_mode()
def retention_probe(
    model: ExactStateCarry,
    maximum_horizon: int,
    seed: int,
) -> list[dict[str, float | int]]:
    model.eval()
    _, state_a = model(encode("packet[1234] zibble mork talven quux "))
    _, state_b = model(encode("packet[9876] zibble mork talven quux "))

    requested = {
        horizon for horizon in
        [0, 1, 10, 100, 1000, 10000, 100000]
        if horizon <= maximum_horizon
    }
    requested.add(maximum_horizon)

    rng = random.Random(seed)
    common_characters = "zibble mork talven quux "
    measurements = []

    def measure(horizon: int) -> None:
        measurements.append({
            "common_tokens_after_key": horizon,
            "state_cosine_similarity": float(
                F.cosine_similarity(state_a, state_b, dim=-1)
            ),
            "state_l2_separation": float(
                torch.linalg.vector_norm(state_a - state_b, ord=2)
            ),
        })

    if 0 in requested:
        measure(0)

    for horizon in range(1, maximum_horizon + 1):
        character = common_characters[
            rng.randrange(len(common_characters))
        ]
        token = torch.tensor([[STOI[character]]], dtype=torch.long)
        _, state_a = model(token, state_a)
        _, state_b = model(token, state_b)

        if horizon in requested:
            measure(horizon)

    model.train()
    return measurements


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_data, train_mask = make_delayed_trash(
        args.train_records,
        args.seed + 10,
    )
    validation_data, validation_mask = make_delayed_trash(
        args.validation_records,
        args.seed + 20,
    )

    train_batches = StreamBatcher(
        train_data,
        train_mask,
        args.batch_size,
        args.seq_len,
    )
    validation_batches = StreamBatcher(
        validation_data,
        validation_mask,
        args.batch_size,
        args.seq_len,
    )

    model = ExactStateCarry(STATE_SIZE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    started = time.perf_counter()
    steps = 0
    passes = 0
    last_loss = None

    while time.perf_counter() - started < args.train_seconds:
        hidden = None

        for inputs, targets, _ in train_batches.chunks():
            if time.perf_counter() - started >= args.train_seconds:
                break

            states, hidden = model(inputs, hidden)
            loss = exact_cosine_loss(targets, states)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            hidden = hidden.detach()
            steps += 1
            last_loss = float(loss.detach())

        passes += 1

    metrics = evaluate(
        model,
        validation_batches,
        args.validation_chunks,
    )
    retention = retention_probe(
        model,
        args.probe_horizon,
        args.seed + 30,
    )

    print(json.dumps({
        "status": "COMPLETE",
        "elapsed_seconds": time.perf_counter() - started,
        "steps": steps,
        "contiguous_passes_completed": passes,
        "final_train_loss": last_loss,
        **metrics,
        "retention_probe": retention,
    }, indent=2))


if __name__ == "__main__":
    main()

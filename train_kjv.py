#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from exact_state_carry import ExactStateCarry, exact_cosine_loss


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train/evaluate Exact State Carry on the raw 77-character KJV."
    )
    parser.add_argument("--data", type=Path, default=repo / "data" / "kjv.txt")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo / "checkpoints" / "kjv_exact_state_200s.pt",
    )
    parser.add_argument("--train-seconds", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-chunks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--prompt",
        default=(
            "And the LORD said unto Moses, saying, "
            "Speak unto the children of Israel, "
        ),
    )
    parser.add_argument("--generate-length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=6)
    return parser.parse_args()


class StreamBatcher:
    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        stream_length = len(data) // batch_size
        if stream_length <= sequence_length + 1:
            raise ValueError("Corpus is too short for these batch settings")

        usable = stream_length * batch_size
        self.tokens = data[:usable].view(batch_size, stream_length)
        self.sequence_length = sequence_length
        self.stream_length = stream_length
        self.steps = (stream_length - 1) // sequence_length

    def chunk_at(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = position + self.sequence_length
        return (
            self.tokens[:, position:end],
            self.tokens[:, position + 1:end + 1],
        )

    def chunks(self):
        for step in range(self.steps):
            yield self.chunk_at(step * self.sequence_length)


def load_raw_kjv(path: Path) -> tuple[str, str, dict[str, int], list[str], torch.Tensor]:
    text = path.read_text(encoding="utf-8-sig")
    alphabet = "".join(sorted(set(text)))

    if len(alphabet) != 77:
        raise ValueError(
            f"Expected exactly 77 raw KJV characters, found {len(alphabet)}"
        )

    stoi = {character: index for index, character in enumerate(alphabet)}
    itos = list(alphabet)
    encoded = torch.tensor(
        [stoi[character] for character in text],
        dtype=torch.long,
    )
    return text, alphabet, stoi, itos, encoded


def torch_load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.inference_mode()
def evaluate(
    model: ExactStateCarry,
    batcher: StreamBatcher,
    max_chunks: int,
) -> dict[str, float | int]:
    model.eval()
    hidden = None
    loss_sum = 0.0
    accuracy_sum = 0.0
    maximum_norm_error = 0.0
    chunks = 0

    for inputs, targets in batcher.chunks():
        states, hidden = model(inputs, hidden)
        hidden = hidden.detach()

        loss_sum += exact_cosine_loss(targets, states).item()
        predictions = states.argmax(dim=-1)
        accuracy_sum += (
            predictions == targets
        ).float().mean().item()

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
        "maximum_norm_error": maximum_norm_error,
        "validation_chunks": chunks,
    }


@torch.inference_mode()
def generate(
    model: ExactStateCarry,
    prompt: str,
    stoi: dict[str, int],
    itos: list[str],
    length: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(f"Prompt contains unknown characters: {unknown!r}")

    model.eval()
    generator = torch.Generator().manual_seed(seed)
    prompt_ids = torch.tensor(
        [[stoi[character] for character in prompt]],
        dtype=torch.long,
    )

    states, hidden = model(prompt_ids)
    scores = states[:, -1]
    output = list(prompt)
    state_size = scores.shape[-1]

    for _ in range(length):
        # Native unlearned readout used by the original experiment:
        # keep positive state coordinates and normalize them as sampling weights.
        weights = scores.clamp_min(0.0)

        if temperature != 1.0:
            weights = weights.pow(
                1.0 / max(temperature, 1.0e-6)
            )

        if 0 < top_k < state_size:
            values, indices = torch.topk(weights, k=top_k, dim=-1)
            if float(values.sum()) <= model.eps:
                probabilities = torch.full_like(values, 1.0 / top_k)
            else:
                probabilities = values / values.sum(dim=-1, keepdim=True)

            selection = torch.multinomial(
                probabilities,
                1,
                generator=generator,
            )
            token = indices.gather(dim=-1, index=selection)
        else:
            if float(weights.sum()) <= model.eps:
                probabilities = torch.full_like(
                    weights,
                    1.0 / state_size,
                )
            else:
                probabilities = weights / weights.sum(
                    dim=-1,
                    keepdim=True,
                )
            token = torch.multinomial(
                probabilities,
                1,
                generator=generator,
            )

        output.append(itos[int(token.item())])
        states, hidden = model(token, hidden)
        scores = states[:, -1]

    model.train()
    return "".join(output)


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    text, alphabet, stoi, itos, encoded = load_raw_kjv(args.data)
    state_size = len(alphabet)

    split = int(len(encoded) * 0.95)
    train_data = encoded[:split]
    validation_data = encoded[split:]

    train_batches = StreamBatcher(
        train_data,
        args.batch_size,
        args.seq_len,
    )
    validation_batches = StreamBatcher(
        validation_data,
        args.batch_size,
        args.seq_len,
    )

    model = ExactStateCarry(state_size)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    hidden = None
    position = 0
    passes = 0
    steps = 0
    prior_training_seconds = 0.0
    last_loss = None

    if args.checkpoint.exists() and not args.fresh:
        checkpoint = torch_load_checkpoint(args.checkpoint)

        if checkpoint.get("alphabet") != alphabet:
            raise ValueError("Checkpoint alphabet does not match the corpus")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        loaded_hidden = checkpoint.get("hidden")
        if (
            loaded_hidden is not None
            and loaded_hidden.shape == (args.batch_size, state_size)
        ):
            hidden = loaded_hidden
            position = int(checkpoint.get("pos", 0))
        else:
            hidden = None
            position = 0

        passes = int(checkpoint.get("passes", 0))
        steps = int(checkpoint.get("steps", 0))
        prior_training_seconds = float(
            checkpoint.get("training_seconds", 0.0)
        )
        last_loss = checkpoint.get("last_loss")

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print(json.dumps({
        "status": "STARTING",
        "torch_version": torch.__version__,
        "threads": args.threads,
        "state_size": state_size,
        "parameter_count": parameter_count,
        "raw_kjv_characters": len(encoded),
        "train_characters": len(train_data),
        "validation_characters": len(validation_data),
        "steps_per_contiguous_pass": train_batches.steps,
        "checkpoint_loaded": args.checkpoint.exists() and not args.fresh,
        "prior_training_seconds": prior_training_seconds,
        "additional_training_seconds_requested": args.train_seconds,
        "state_carried_between_chunks": True,
    }), flush=True)

    model.train()
    started = time.perf_counter()

    while time.perf_counter() - started < args.train_seconds:
        if position + args.seq_len + 1 >= train_batches.stream_length:
            position = 0
            passes += 1
            hidden = None

        inputs, targets = train_batches.chunk_at(position)
        states, hidden = model(inputs, hidden)
        loss = exact_cosine_loss(targets, states)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        optimizer.step()

        hidden = hidden.detach()
        position += args.seq_len
        steps += 1
        last_loss = float(loss.detach())

    added_seconds = time.perf_counter() - started
    total_training_seconds = prior_training_seconds + added_seconds

    if args.train_seconds > 0:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hidden": hidden,
            "pos": position,
            "passes": passes,
            "steps": steps,
            "training_seconds": total_training_seconds,
            "last_loss": last_loss,
            "alphabet": alphabet,
        }, args.checkpoint)

    metrics = evaluate(
        model,
        validation_batches,
        args.validation_chunks,
    )

    sample = generate(
        model=model,
        prompt=args.prompt,
        stoi=stoi,
        itos=itos,
        length=args.generate_length,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed + 40,
    )

    print(json.dumps({
        "status": "COMPLETE",
        "added_training_seconds_actual": added_seconds,
        "total_training_seconds": total_training_seconds,
        "steps": steps,
        "contiguous_passes_completed": passes,
        "last_train_loss": last_loss,
        **metrics,
        "generation_top_k": args.top_k,
        "generation_temperature": args.temperature,
    }, indent=2), flush=True)

    print("----- BEGIN SAMPLE -----")
    print(sample)
    print("----- END SAMPLE -----")


if __name__ == "__main__":
    main()

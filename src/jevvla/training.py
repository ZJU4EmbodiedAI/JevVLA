from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import TrainingData, check_disjoint
from .losses import (denoising_score_matching_loss, gradient_matching_loss,
                     masked_preference_cross_entropy, mixture_gradient_target,
                     pairwise_outcome_loss)
from .model import SCHEMA, UnifiedQuadraticEnergy
from .profiles import TrainingConfig


def native_score(model, context, action, config, *, create_graph):
    with torch.enable_grad():
        action = action.detach().requires_grad_(True)
        encoded = config.action_bound * torch.tanh(action * config.action_gain / config.action_bound)
        encoded = F.pad(encoded, (0, 32 - encoded.shape[1]))
        score = model.raw_energy(context, encoded[:, None]).sum()
        return torch.autograd.grad(score, action, create_graph=create_graph)[0]


def sampled_sigma(n, config, generator, device):
    step = torch.randint(1, config.noise_steps + 1, (n,), generator=generator, device=device)
    return config.noise_floor + config.noise_slope * step.float() / config.noise_steps


def denoising_term(model, batch, config, generator, *, training):
    if "native_action" in batch:
        centers = batch["native_action"]
        mass = batch.get("target", torch.ones(centers.shape[:2], device=centers.device))
    elif "centers" in batch:
        centers = batch["centers"]
        mass = batch["mixture_weights"]
    elif "action" in batch:
        centers = batch["action"][:, None]
        mass = torch.ones(centers.shape[:2], device=centers.device)
    else:
        raise ValueError("denoising requires native_action, centers, or action")
    branch = torch.multinomial(mass, 1, generator=generator).squeeze(1)
    clean = centers[torch.arange(len(centers), device=centers.device), branch]
    sigma = sampled_sigma(len(clean), config, generator, clean.device)
    noise = torch.randn(clean.shape, generator=generator, device=clean.device)
    noisy = clean + sigma[:, None] * noise
    predicted = native_score(model, batch["context"], noisy, config, create_graph=training)
    return denoising_score_matching_loss(predicted, noise, sigma, sample_weight=batch.get("weight"))


def objective(model, batch, config, generator, *, training):
    pieces = {}
    if config.preference_weight:
        if "candidate" not in batch:
            raise ValueError("preference training requires candidate features")
        logits = model.logits(batch["context"], batch["candidate"], batch["prior"],
                              batch["candidate_mask"], batch["direct_index"],
                              prior_scale=config.prior_scale)
        if config.objective == "ce":
            pieces["preference"] = masked_preference_cross_entropy(logits, batch["target"],
                                                                   batch["candidate_mask"])
        elif config.objective == "pairwise":
            pieces["preference"] = pairwise_outcome_loss(logits, batch["terminal_success"],
                                                        batch["candidate_mask"])
        else:
            raise ValueError("objective must be ce or pairwise")
    if config.dsm_weight:
        pieces["denoising"] = denoising_term(model, batch, config, generator, training=training)
    if config.gradient_weight:
        if "action" not in batch:
            raise ValueError("gradient matching requires native action states")
        if "gradient_target" in batch:
            target = batch["gradient_target"]
        elif "centers" in batch and "sigma" in batch:
            target, _ = mixture_gradient_target(batch["action"], batch["centers"],
                                                batch["mixture_weights"], batch["sigma"])
        else:
            raise ValueError("gradient matching requires gradient_target or centers and sigma")
        predicted = native_score(model, batch["context"], batch["action"], config,
                                 create_graph=training)
        coordinates = batch.get("coordinate_mask")
        if coordinates is None:
            coordinates = torch.arange(predicted.shape[1], device=predicted.device) >= config.supervised_start
        if coordinates.shape != (predicted.shape[1],) or not bool(coordinates.any()):
            raise ValueError("coordinate mask must select supervised action dimensions")
        pieces["gradient"] = gradient_matching_loss(predicted, target,
                                                    coordinate_mask=coordinates,
                                                    sample_weight=batch.get("weight")) / coordinates.sum()
    if not pieces:
        raise ValueError("at least one training objective must be enabled")
    total = sum(pieces.get(name, 0.0) * weight for name, weight in (
        ("preference", config.preference_weight), ("denoising", config.dsm_weight),
        ("gradient", config.gradient_weight)
    ))
    return total, pieces


def evaluate_loss(model, data, config, *, device):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.seed)
    total, rows = 0.0, 0
    for indices in torch.arange(len(data)).split(config.batch_size):
        loss, _ = objective(model, data.batch(indices, device), config, generator, training=False)
        total += float(loss.detach()) * len(indices)
        rows += len(indices)
    return total / rows


def selection_scores(model, data, config, *, device):
    model.eval()
    result = []
    with torch.no_grad():
        for indices in torch.arange(len(data)).split(config.batch_size):
            batch = data.batch(indices, device)
            logits = model.logits(batch["context"], batch["candidate"], batch["prior"],
                                  batch["candidate_mask"], batch["direct_index"],
                                  gain=config.inference_gain, prior_scale=config.prior_scale)
            result.append(logits.cpu().numpy())
    return np.concatenate(result)


def choice_metrics(outcome, choice, direct):
    row = np.arange(len(choice))
    original = np.asarray(outcome[row, direct], bool)
    selected = np.asarray(outcome[row, choice], bool)
    return {"episodes": len(choice), "direct_successes": int(original.sum()),
            "successes": int(selected.sum()), "rescues": int((~original & selected).sum()),
            "harms": int((original & ~selected).sum()),
            "interventions": int((choice != direct).sum())}


def select_global_margin(scores, outcome, direct, mask, *, fixed_margin=None):
    n, k = scores.shape
    if outcome.shape != scores.shape or mask.shape != scores.shape or direct.shape != (n,):
        raise ValueError("selection arrays have inconsistent shapes")
    rows = np.arange(n)
    if k == 1:
        return 0.0 if fixed_margin is None else float(fixed_margin), choice_metrics(outcome, direct, direct)
    challenger_scores = np.where(mask, scores, -np.inf).copy()
    challenger_scores[rows, direct] = -np.inf
    challenger = np.argmax(challenger_scores, axis=1)
    advantage = challenger_scores[rows, challenger] - scores[rows, direct]
    thresholds = np.array([fixed_margin]) if fixed_margin is not None else np.r_[
        0.0, np.unique(advantage[np.isfinite(advantage) & (advantage > 0)])
    ]
    best = None
    for margin in thresholds:
        choice = np.where(advantage > margin, challenger, direct)
        metrics = choice_metrics(outcome, choice, direct)
        key = (metrics["successes"], -metrics["harms"], -metrics["interventions"], -float(margin))
        if best is None or key > best[0]:
            best = key, float(margin), metrics
    return best[1], best[2]


def fit(train: TrainingData, validation: TrainingData | None, output: str | Path,
        config: TrainingConfig, *, device="cpu", warmstart=None,
        future: TrainingData | None = None, future_validation: TrainingData | None = None,
        select_best: bool = True):
    if (config.epochs < 1 or config.batch_size < 1 or config.future_batch_size < 1 or
            config.learning_rate <= 0 or config.action_gain <= 0 or config.action_bound <= 0 or
            config.noise_steps < 1 or config.noise_floor <= 0 or config.noise_slope < 0 or
            config.gradient_clip <= 0 or any(value < 0 for value in
            (config.preference_weight, config.gradient_weight, config.dsm_weight, config.future_weight))):
        raise ValueError("invalid training settings")
    if config.selection not in {"loss", "success"}:
        raise ValueError("selection must be loss or success")
    if select_best and validation is None:
        raise ValueError("checkpoint selection requires validation data")
    if config.future_weight and (future is None or (select_best and future_validation is None)):
        raise ValueError("future supervision requires separate training and validation data")
    if validation is not None:
        check_disjoint(train, validation)
    if future is not None and future_validation is not None:
        check_disjoint(future, future_validation)
    target_device = torch.device(device)
    torch.manual_seed(config.seed)
    generator = torch.Generator(device=target_device).manual_seed(config.seed)
    order_generator = torch.Generator().manual_seed(config.seed)
    model = (UnifiedQuadraticEnergy.from_checkpoint(warmstart, device=device) if warmstart
             else UnifiedQuadraticEnergy(config.width, config.rank).to(target_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_key = None
    future_config = TrainingConfig(**{**asdict(config), "preference_weight": 0.0,
                                    "gradient_weight": 0.0, "dsm_weight": 1.0})
    for epoch in range(1, config.epochs + 1):
        model.train()
        initial_order = torch.randperm(len(train), generator=order_generator)
        if config.future_weight:
            future_order = torch.randperm(len(future), generator=order_generator)
            batches = [
                (initial_order[torch.arange(batch * config.batch_size,
                                            (batch + 1) * config.batch_size) % len(train)], future_indices)
                for batch, future_indices in enumerate(future_order.split(config.future_batch_size))
            ]
        else:
            batches = [(indices, None) for indices in initial_order.split(config.batch_size)]
        for indices, future_indices in batches:
            loss, _ = objective(model, train.batch(indices, target_device), config,
                                generator, training=True)
            if config.future_weight:
                extra = denoising_term(model, future.batch(future_indices, target_device), config,
                                       generator, training=True)
                loss = loss + config.future_weight * extra
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("training loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        held_loss = evaluate_loss(model, validation, config, device=target_device) if validation is not None else None
        if config.future_weight and future_validation is not None:
            held_loss += config.future_weight * evaluate_loss(model, future_validation, future_config,
                                                               device=target_device)
        margin = config.direct_margin
        metrics = {"validation_loss": held_loss} if held_loss is not None else {}
        if select_best and config.selection == "success":
            scores = selection_scores(model, validation, config, device=target_device)
            values = validation.values
            margin, success_metrics = select_global_margin(
                scores, values["terminal_success"].numpy(), values["direct_index"].numpy(),
                values["candidate_mask"].numpy(),
                fixed_margin=None if config.select_margin else config.direct_margin
            )
            metrics.update(success_metrics)
            key = (metrics["successes"], -metrics["harms"], -metrics["interventions"], -epoch)
        else:
            key = (-held_loss, -epoch) if select_best else (epoch,)
        if not select_best or best_key is None or key > best_key:
            best_key = key
            torch.save({"schema": SCHEMA, "architecture": "UnifiedQuadraticEnergy",
                        "width": model.width, "rank": model.rank,
                        "model": {name: value.detach().cpu().clone()
                                  for name, value in model.state_dict().items()},
                        "epoch": epoch, "config": asdict(config),
                        "global_gain": config.inference_gain,
                        "global_direct_margin": margin, "selection": metrics}, path)
    return UnifiedQuadraticEnergy.from_checkpoint(path, device=device)

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import torch

from .data import TrainingData
from .profiles import TrainingConfig
from .training import choice_metrics, selection_scores


def evaluate_candidates(model, data: TrainingData, config: TrainingConfig, *, device="cpu", margin=None):
    scores = selection_scores(model, data, config, device=torch.device(device))
    direct = data.values["direct_index"].numpy()
    rows = np.arange(len(data))
    mask = data.values["candidate_mask"].numpy()
    challenger = np.where(mask, scores, -np.inf).copy()
    challenger[rows, direct] = -np.inf
    selected = np.argmax(challenger, axis=1)
    threshold = config.direct_margin if margin is None else margin
    choice = np.where(challenger[rows, selected] > scores[rows, direct] + threshold, selected, direct)
    metrics = choice_metrics(data.values["terminal_success"].numpy(), choice, direct)
    return {**metrics, "success_rate": metrics["successes"] / len(data)}


def evaluate_policy(environment, policy, seeds: Iterable[int], max_steps: int,
                    success: Callable, *, reset_options=None, action_transform=None):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    episodes = 0
    successes = 0
    for seed in seeds:
        reset = environment.reset(seed=int(seed), options=reset_options) if reset_options is not None else (
            environment.reset(seed=int(seed))
        )
        observation = reset[0] if isinstance(reset, tuple) and len(reset) == 2 else reset
        if callable(getattr(policy, "reset", None)):
            policy.reset()
        completed = False
        for step in range(max_steps):
            action = policy(observation) if callable(policy) else policy.infer(observation)
            if action_transform is not None:
                action = action_transform(action)
            result = environment.step(action)
            if len(result) == 5:
                observation, reward, terminated, truncated, info = result
                done = bool(terminated or truncated)
            elif len(result) == 4:
                observation, reward, done, info = result
            else:
                raise ValueError("environment.step must return four or five values")
            completed = bool(success(environment, observation, info))
            if completed or done:
                break
        episodes += 1
        successes += int(completed)
    if not episodes:
        raise ValueError("seeds must contain at least one episode")
    return {"episodes": episodes, "successes": successes, "success_rate": successes / episodes}

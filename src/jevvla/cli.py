from __future__ import annotations

import argparse
from dataclasses import fields, replace
import importlib
import json
from pathlib import Path
import tomllib

import numpy as np
import torch

from .data import TrainingData
from .evaluation import evaluate_candidates, evaluate_policy
from .model import UnifiedQuadraticEnergy
from .profiles import PROFILES, TrainingConfig, profile
from .training import fit


def training_data(path, benchmark, *, future=False, max_steps=200):
    if benchmark != "language_table":
        return TrainingData.load(path)
    from .adapters.language_table import prepare_future, prepare_initial

    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if future and "frozen_BC_feature" in arrays:
        return TrainingData(prepare_future(arrays, max_steps=max_steps))
    if not future and {"task", "seed", "native_action", "candidate"}.issubset(arrays):
        return TrainingData(prepare_initial(arrays))
    return TrainingData(arrays)


def configuration(args):
    values = {}
    if args.config is not None:
        with args.config.open("rb") as stream:
            values.update(tomllib.load(stream))
    unknown = set(values) - {item.name for item in fields(TrainingConfig)}
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    for name in ("epochs", "batch_size", "learning_rate", "seed", "objective"):
        if hasattr(args, name) and getattr(args, name) is not None:
            values[name] = getattr(args, name)
    return profile(args.benchmark, **values)


def train(args):
    config = configuration(args)
    data = training_data(args.data, args.benchmark)
    validation = training_data(args.validation, args.benchmark)
    future = training_data(args.future, args.benchmark, future=True, max_steps=args.max_steps) if args.future is not None else None
    future_validation = training_data(args.future_validation, args.benchmark, future=True,
                                      max_steps=args.max_steps) if args.future_validation is not None else None
    fit(data, validation, args.output, config, device=args.device, warmstart=args.warmstart,
        future=future, future_validation=future_validation)
    payload = torch.load(args.output, map_location="cpu", weights_only=True)
    if args.refit is not None:
        if config.future_weight and args.refit_future is None:
            raise ValueError("future supervision refit requires --refit-future")
        selected = int(payload["epoch"])
        selection = payload["selection"]
        selected_margin = payload["global_direct_margin"]
        fit(training_data(args.refit, args.benchmark), None, args.output, replace(config, epochs=selected),
            device=args.device, future=training_data(args.refit_future, args.benchmark, future=True,
                                                    max_steps=args.max_steps) if args.refit_future else None,
            select_best=False)
        payload = torch.load(args.output, map_location="cpu", weights_only=True)
        payload["selection"] = selection
        payload["global_direct_margin"] = selected_margin
        torch.save(payload, args.output)
    print(json.dumps({"checkpoint": str(args.output), "epoch": payload["epoch"],
                      "selection": payload["selection"]}, sort_keys=True))


def evaluate_records(args):
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = TrainingConfig(**payload["config"]) if "config" in payload else configuration(args)
    model = UnifiedQuadraticEnergy.from_checkpoint(args.checkpoint, device=args.device)
    data = TrainingData.load(args.data)
    result = evaluate_candidates(model, data, config, device=args.device,
                                 margin=payload.get("global_direct_margin", config.direct_margin))
    print(json.dumps(result, sort_keys=True))


def evaluate(args):
    module_name, factory_name = args.factory.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    settings = json.loads(args.factory_options)
    bundle = factory(checkpoint=args.checkpoint, device=args.device, **settings)
    environment = bundle["environment"]
    try:
        result = evaluate_policy(environment, bundle["policy"], args.seeds, args.max_steps,
                                 bundle["success"], reset_options=bundle.get("reset_options"),
                                 action_transform=bundle.get("action_transform"))
    finally:
        if callable(getattr(environment, "close", None)):
            environment.close()
    print(json.dumps(result, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jevvla")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--benchmark", choices=tuple(PROFILES), required=True)
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--validation", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--config", type=Path)
    training.add_argument("--device", default="cpu")
    training.add_argument("--epochs", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--seed", type=int)
    training.add_argument("--objective", choices=("ce", "pairwise"))
    training.add_argument("--warmstart", type=Path)
    training.add_argument("--future", type=Path)
    training.add_argument("--future-validation", type=Path)
    training.add_argument("--refit", type=Path)
    training.add_argument("--refit-future", type=Path)
    training.add_argument("--max-steps", type=int, default=200)
    training.set_defaults(run=train)
    records = commands.add_parser("evaluate-records")
    records.add_argument("--checkpoint", type=Path, required=True)
    records.add_argument("--data", type=Path, required=True)
    records.add_argument("--benchmark", choices=tuple(PROFILES), required=True)
    records.add_argument("--config", type=Path)
    records.add_argument("--device", default="cpu")
    records.set_defaults(run=evaluate_records)
    rollout = commands.add_parser("evaluate")
    rollout.add_argument("--factory", required=True)
    rollout.add_argument("--factory-options", default="{}")
    rollout.add_argument("--checkpoint", type=Path, required=True)
    rollout.add_argument("--device", default="cpu")
    rollout.add_argument("--seeds", type=int, nargs="+", required=True)
    rollout.add_argument("--max-steps", type=int, required=True)
    rollout.set_defaults(run=evaluate)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()

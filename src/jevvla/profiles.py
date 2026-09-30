from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 0
    width: int = 128
    rank: int = 8
    preference_weight: float = 1.0
    dsm_weight: float = 0.0
    gradient_weight: float = 0.0
    future_weight: float = 0.0
    future_batch_size: int = 128
    objective: str = "ce"
    action_gain: float = 1.0
    action_bound: float = 4.0
    noise_floor: float = 0.2
    noise_slope: float = 0.8
    noise_steps: int = 10
    prior_scale: float = 1.0
    inference_gain: float = 1.0
    direct_margin: float = 0.0
    select_margin: bool = False
    selection: str = "loss"
    supervised_start: int = 0


PROFILES = {
    "cliport": TrainingConfig(seed=5137, selection="success", select_margin=True),
    "vima": TrainingConfig(batch_size=64, seed=5137, prior_scale=20.0, inference_gain=4.0),
    "libero": TrainingConfig(seed=5137, inference_gain=0.25, direct_margin=0.005),
    "language_table": TrainingConfig(batch_size=16, seed=260930, dsm_weight=1.0,
                                   future_weight=0.2, action_gain=10.0, action_bound=1.0),
    "metaworld": TrainingConfig(epochs=8, learning_rate=2e-4,
                               preference_weight=0.0, gradient_weight=1.0, supervised_start=4),
    "metaworld_continuation": TrainingConfig(epochs=3, learning_rate=3e-5,
                                            preference_weight=0.0, gradient_weight=1.0, supervised_start=4),
}


def profile(name: str, **overrides):
    if name not in PROFILES:
        raise ValueError(f"unknown benchmark profile: {name}")
    return replace(PROFILES[name], **{key: value for key, value in overrides.items() if value is not None})

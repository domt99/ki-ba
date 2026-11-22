import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# same mapping as .githooks/prepare-commit-msg
TEAM_INITIALS = {
    "domenictheis@gmail.com": "dt",
    "maurice.myr@outlook.de": "mm",
}


def get_initials():
    email = subprocess.run(
        ["git", "config", "user.email"], capture_output=True, text=True
    ).stdout.strip().lower()
    if email not in TEAM_INITIALS:
        raise ValueError(
            f"git user.email '{email}' is not a team address, "
            f"allowed: {', '.join(TEAM_INITIALS)} (see README Setup)"
        )
    return TEAM_INITIALS[email]


def get_run_name(config):
    if config.template:
        return Path(config.template).stem
    topology = "-".join(str(n) for n in config.topology)
    return f"{config.data_set_type}-{topology}-{config.optimizer_type}"


def create_run_folder(config, initials, root="results"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    name = f"{timestamp}_{initials}_{get_run_name(config)}"
    folder = Path(root) / name
    # short runs (XOR) can finish within the same second
    counter = 2
    while folder.exists():
        folder = Path(root) / f"{name}_{counter}"
        counter += 1
    folder.mkdir(parents=True)
    return folder


def save_result(folder, framework, initials, duration_seconds, test_accuracy=None):
    result = {
        "run_by": initials,
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(framework.config),
        "parameter_count": None,
        "duration_seconds": round(duration_seconds, 1),
        "best_epoch": framework.best_epoch,
        "best_validation_accuracy": _to_float(framework.best_validation_accuracy)
        if framework.best_epoch is not None else None,
        "test_evaluated": test_accuracy is not None,
        "test_accuracy": _to_float(test_accuracy),
        "epoch_errors": [_to_float(value) for value in framework.epoch_errors],
        "train_accuracy": [_to_float(value) for value in framework.train_accuracy],
        "validation_accuracy": [_to_float(value) for value in framework.validation_accuracy],
    }
    path = Path(folder) / "result.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def _to_float(value):
    return None if value is None else float(value)

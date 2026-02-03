import os

from rich.table import Table
from rich.console import Console
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _read_run(log_dir):
    """Read metrics from a single TensorBoard run directory."""
    ea = EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])

    def last_value(tag):
        if tag in tags:
            events = ea.Scalars(tag)
            if events:
                return events[-1].value
        return None

    train_acc = last_value("train_acc_epoch") or last_value("train_acc")
    val_acc = last_value("val_acc")
    test_acc = last_value("test_acc")
    duration = last_value("training_duration")
    trainable_params = last_value("trainable_params")
    frozen_params = last_value("frozen_params")

    # Number of epochs: count distinct steps in an epoch-level metric
    n_epochs = None
    for tag in ("train_acc_epoch", "val_acc", "train_loss_epoch", "val_loss"):
        if tag in tags:
            n_epochs = len(ea.Scalars(tag))
            break

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "n_epochs": n_epochs,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "duration": duration,
    }


def _merge_event_files(version_dir):
    """Merge metrics from all event files in a version directory."""
    merged = {}
    for entry in sorted(os.listdir(version_dir)):
        if entry.startswith("events.out.tfevents"):
            result = _read_run(version_dir)
            for k, v in result.items():
                if v is not None:
                    merged[k] = v
            break  # EventAccumulator reads all event files in the dir
    return merged


def _fmt(val, fmt=".4f"):
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:{fmt}}"
    return str(val)


def _fmt_params(val):
    if val is None:
        return "-"
    n = int(val)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_duration(val):
    if val is None:
        return "-"
    secs = int(val)
    if secs >= 3600:
        return f"{secs // 3600}h {(secs % 3600) // 60}m {secs % 60}s"
    if secs >= 60:
        return f"{secs // 60}m {secs % 60}s"
    return f"{secs}s"


def summarize(save_dir="logs/", all_versions=False):
    """Print a summary table of training runs found in save_dir.

    If all_versions is False (default), only the latest version per model is shown.
    """
    table = Table(title="Training Summary")
    table.add_column("Model", style="bold")
    table.add_column("Version")
    table.add_column("Train Acc", justify="right")
    table.add_column("Val Acc", justify="right")
    table.add_column("Test Acc", justify="right")
    table.add_column("Trainable", justify="right")
    table.add_column("Frozen", justify="right")
    table.add_column("Epochs", justify="right")
    table.add_column("Duration", justify="right")

    if not os.path.isdir(save_dir):
        Console().print(f"[red]Log directory not found: {save_dir}[/red]")
        return

    for model_name in sorted(os.listdir(save_dir)):
        model_dir = os.path.join(save_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        versions = sorted(
            [v for v in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, v))]
        )
        if not all_versions:
            versions = versions[-1:]

        for version in versions:
            version_dir = os.path.join(model_dir, version)

            metrics = _merge_event_files(version_dir)
            if not metrics:
                continue

            table.add_row(
                model_name,
                version,
                _fmt(metrics.get("train_acc")),
                _fmt(metrics.get("val_acc")),
                _fmt(metrics.get("test_acc")),
                _fmt_params(metrics.get("trainable_params")),
                _fmt_params(metrics.get("frozen_params")),
                _fmt(metrics.get("n_epochs"), "d"),
                _fmt_duration(metrics.get("duration")),
            )

    console = Console()
    console.print(table)


if __name__ == "__main__":
    summarize()

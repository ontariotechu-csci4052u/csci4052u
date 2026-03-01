# csci4052u — Library Reference

A helper library for machine learning pipelines in CSCI 4052U. It provides three exported modules:

- [`csci4052u.data`](#data-module) — dataset loading and preparation
- [`csci4052u.training`](#training-module) — model training
- [`csci4052u.report`](#report-module) — results summarization

---

## Installation

```bash
pip install git+https://github.com/ontariotechu-csci4052u/csci4052u.git
```

---

## Data Module

```python
from csci4052u.data import FashionMNISTDataModule
```

### `FashionMNISTDataModule`

A [PyTorch Lightning `DataModule`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) wrapping the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Handles downloading, splitting, and batching automatically.

The dataset contains 70,000 28×28 grayscale images across 10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

```python
FashionMNISTDataModule(
    data_dir="./data",
    batch_size=64,
    num_workers=2,
    shape=None,
    fraction=1.0,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `str` | `"./data"` | Directory where the dataset is downloaded and cached. |
| `batch_size` | `int` | `64` | Number of samples per batch in all dataloaders. |
| `num_workers` | `int` | `2` | Number of subprocesses for data loading. Set to `0` to load in the main process. |
| `shape` | `tuple[int, int, int]` or `None` | `None` | Desired output tensor shape as `(channels, height, width)`. If `None`, images are returned as `(1, 28, 28)` normalized tensors. If provided, images are resized to `(height, width)` and, if `channels=3`, expanded from grayscale to 3-channel (RGB-like). Normalization is applied automatically. |
| `fraction` | `float` | `1.0` | Fraction of the dataset to use, in the range `(0, 1]`. Useful for quick experiments on a subset of data. Applied to both the train/val pool and the test set. |

#### Dataset splits

The full training pool (60,000 images, scaled by `fraction`) is split into:
- **Train**: 5/6 of the pool
- **Validation**: 1/6 of the pool

The test set (10,000 images, scaled by `fraction`) is kept separate.

#### Example — default usage

```python
from csci4052u.data import FashionMNISTDataModule

dm = FashionMNISTDataModule()
```

#### Example — resize for a pretrained CNN expecting 3-channel 224×224 input

```python
dm = FashionMNISTDataModule(
    batch_size=128,
    shape=(3, 224, 224),
)
```

#### Example — train on 10% of the data for a quick prototype

```python
dm = FashionMNISTDataModule(fraction=0.1)
```

---

## Training Module

```python
from csci4052u.training import train
```

### `train()`

A high-level training function that wraps PyTorch Lightning's `Trainer`. It automatically configures TensorBoard logging, early stopping, and model checkpointing, then runs training followed by a test evaluation.

Results (metrics, hyperparameters, parameter counts, duration) are saved to a TensorBoard log directory and can be reviewed with [`summarize()`](#summarize).

```python
train(
    model_name,
    model,
    datamodule,
    *,
    max_epochs,
    monitor_metric,
    save_top_k=1,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | *(required)* | Name used to identify this model in logs. Logs are saved to `logs/<model_name>/`. |
| `model` | `LightningModule` | *(required)* | The model to train. Must be a PyTorch Lightning `LightningModule`. |
| `datamodule` | `LightningDataModule` | *(required)* | The data module providing train, validation, and test dataloaders. |
| `max_epochs` | `int` | *(required, keyword-only)* | Maximum number of training epochs. Training may stop earlier if early stopping is triggered. |
| `monitor_metric` | `str` | *(required, keyword-only)* | Name of the metric to monitor for early stopping and checkpointing (e.g., `"val_loss"`). This metric must be logged by the model during validation. |
| `save_top_k` | `int` | `1` | Number of best model checkpoints to keep on disk. |

#### Automatic behaviours

- **TensorBoard logging**: logs are saved to `logs/<model_name>/version_N/`. Run `tensorboard --logdir logs/` to view them.
- **Early stopping**: training halts if `monitor_metric` does not improve for 3 consecutive epochs (minimization mode).
- **Checkpointing**: the best `save_top_k` checkpoints (by `monitor_metric`) are saved automatically.
- **Parameter logging**: trainable and frozen parameter counts are logged as hyperparameters.
- **Duration logging**: total training wall-clock time is recorded in the TensorBoard log.
- **Test evaluation**: `trainer.test()` is called automatically after training completes.

#### Example

```python
import lightning as L
import torch
import torch.nn as nn
from csci4052u.data import FashionMNISTDataModule
from csci4052u.training import train

class SimpleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

dm = FashionMNISTDataModule(batch_size=128)
model = SimpleModel()

train(
    "simple_mlp",
    model,
    dm,
    max_epochs=20,
    monitor_metric="val_loss",
)
```

---

## Report Module

```python
from csci4052u.report import summarize
```

### `summarize()`

Reads TensorBoard logs produced by [`train()`](#train) and prints a formatted summary table to the terminal. Useful for comparing multiple models and training runs side by side.

```python
summarize(
    save_dir="logs/",
    all_versions=False,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_dir` | `str` | `"logs/"` | Path to the root TensorBoard log directory. Should match the `save_dir` used during training (always `logs/` when using [`train()`](#train)). |
| `all_versions` | `bool` | `False` | If `False`, only the latest version of each model is shown. If `True`, every recorded version is shown, allowing you to track changes across re-runs. |

#### Displayed columns

| Column | Description |
|--------|-------------|
| Model | Model name passed to `train()` |
| Version | TensorBoard version directory (e.g., `version_0`) |
| Train Acc | Final training accuracy (`train_acc_epoch` or `train_acc`) |
| Val Acc | Final validation accuracy (`val_acc`) |
| Test Acc | Test accuracy (`test_acc`) |
| Trainable | Number of trainable parameters (formatted as K/M) |
| Frozen | Number of frozen (non-trainable) parameters (formatted as K/M) |
| Epochs | Number of epochs completed |
| Duration | Total training time (formatted as `Xh Ym Zs`) |

Missing metrics are displayed as `-`.

#### Example — show latest run per model

```python
from csci4052u.report import summarize

summarize()
```

#### Example — show all versions for full history

```python
summarize(all_versions=True)
```

#### Example output

```
                        Training Summary
┏━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Model      ┃ Version   ┃ Train Acc ┃  Val Acc ┃  Test Acc ┃ Trainable ┃ Frozen ┃ Epochs ┃ Duration ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ simple_mlp │ version_0 │    0.8821 │   0.8745 │    0.8712 │    101.8K │      0 │     12 │    1m 4s │
└────────────┴───────────┴───────────┴──────────┴───────────┴───────────┴────────┴────────┴──────────┘
```

---

## End-to-End Workflow

```python
from csci4052u.data import FashionMNISTDataModule
from csci4052u.training import train
from csci4052u.report import summarize

# 1. Prepare data
dm = FashionMNISTDataModule(batch_size=128, fraction=0.1)

# 2. Define and train your model
model = MyLightningModel()
train("my_model", model, dm, max_epochs=30, monitor_metric="val_loss")

# 3. Review results
summarize()
```

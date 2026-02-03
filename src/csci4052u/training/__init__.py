import time
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

def train(
    model_name: str,
    model: LightningModule,
    datamodule: LightningDataModule,
    *,
    max_epochs: int,
    monitor_metric: str,
    save_top_k:int = 1,
):
    
    logger = TensorBoardLogger(save_dir="logs/", name=model_name)
    callbacks = [
        EarlyStopping(monitor=monitor_metric, patience=3, mode="min"),
        ModelCheckpoint(monitor=monitor_metric, save_top_k=save_top_k, mode='min'),
    ]

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.log_hyperparams(model.hparams, {
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
    })

    start = time.time()
    trainer.fit(model, datamodule=datamodule)
    duration = time.time() - start
    logger.experiment.add_scalar("training_duration", duration, global_step=0)

    trainer.test(model, datamodule=datamodule)
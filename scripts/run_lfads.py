import os
import shutil
from datetime import datetime
from pathlib import Path

import logging
import os
import warnings
from glob import glob
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra.utils import call, instantiate
from omegaconf import OmegaConf, DictConfig

def main():
    # ---------- OPTIONS -----------
    PROJECT_STR = "lfads"
    DATASET_STR = "temp"
    RUN_DIR = Path("results") / PROJECT_STR / DATASET_STR
    OVERWRITE = False
    # ------------------------------

    # Overwrite the directory if necessary
    if RUN_DIR.exists() and OVERWRITE:
        shutil.rmtree(RUN_DIR)
    if not RUN_DIR.exists():
        RUN_DIR.mkdir(parents=True)
    # Copy this script into the run directory
    # Switch to the `RUN_DIR` and train the model
    config = OmegaConf.load('params.yaml')
    os.chdir(RUN_DIR)
    run_model(
        config=config.lfads,
        # checkpoint_dir='./lightning_checkpoints',
        do_train=True,
        do_posterior_sample=True,
    )

def run_model(
    checkpoint_dir: str = None,
    config: DictConfig = None,
    do_train: bool = True,
    do_posterior_sample: bool = True,
):
    """Adds overrides to the default config, instantiates all PyTorch Lightning
    objects from config, and runs the training pipeline.
    """

    # Avoid flooding the console with output during multi-model runs
    if config.ignore_warnings:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore")

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)

    # Instantiate `LightningDataModule` and `LightningModule`
    datamodule = instantiate(config.datamodule, _convert_="all")
    model = instantiate(config.model)

    # If `checkpoint_dir` is passed, find the most recent checkpoint in the directory
    if checkpoint_dir:
        ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
        ckpt_path = max(glob(ckpt_pattern), key=os.path.getctime)

    if do_train:
        # Instantiate the pytorch_lightning `Trainer` and its callbacks and loggers
        trainer = instantiate(
            config.trainer,
            callbacks=[instantiate(c) for c in config.callbacks.values()],
            logger=[instantiate(lg) for lg in config.logger.values()],
            # gpus=int(torch.cuda.is_available()),
        )
        # Temporary workaround for PTL step-resuming bug
        if checkpoint_dir:
            ckpt = torch.load(ckpt_path)
            trainer.fit_loop.epoch_loop._batches_that_stepped = ckpt["global_step"]
        # Train the model
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path if checkpoint_dir else None,
        )
        # Restore the best checkpoint if necessary - otherwise, use last checkpoint
        if config.posterior_sampling.use_best_ckpt:
            ckpt_path = trainer.checkpoint_callback.best_model_path
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    else:
        if checkpoint_dir:
            # If not training, restore model from the checkpoint
            model.load_state_dict(torch.load(ckpt_path)["state_dict"])

    # Run the posterior sampling function
    if do_posterior_sample:
        if torch.cuda.is_available():
            model = model.to("cuda")
        call(config.posterior_sampling.fn, model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()
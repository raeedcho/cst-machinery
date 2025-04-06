import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

import os
import warnings
from glob import glob
from pathlib import Path

import pytorch_lightning as pl
import torch
import hydra
from hydra.utils import call, instantiate
from lfads_torch.utils import flatten

def main(args):
    dataset_str = args.dataset

    run_dir = Path("results") / "lfads" / dataset_str
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.logdir) / 'train-lfads'
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / f'{dataset_str}.log',
        level=args.loglevel,
    )

    os.chdir(run_dir)
    model = train_model(
        config_path=args.config,
        # checkpoint_dir='./lightning_checkpoints',
    )

    # Save the model
    torch.save(model.state_dict(), "lfads_model.pt")
    logger.info(f'Saved model as lfads_model.pt')

def train_model(
    overrides: dict = {},
    config_path: str = None,
    checkpoint_dir: str = None,
) -> pl.LightningModule:
    # load config
    config_path = Path(config_path)
    overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]
    with hydra.initialize(
        config_path=config_path.parent,
        job_name="train_lfads",
        version_base="1.1",
    ):
        config = hydra.compose(config_name=config_path.name, overrides=overrides)

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

    if torch.cuda.is_available():
        model = model.to("cuda")
    call(config.posterior_sampling.fn, model=model, datamodule=datamodule)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and save trial frame from SMILE data')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to lfads config file',
        default='../conf/lfads/lfads-torch.yaml',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name',
        default='Prez_2022-07-21',
    )
    parser.add_argument(
        '--logdir',
        type=str,
        help='Logging directory',
        default='logs/',
    )
    parser.add_argument(
        '--loglevel',
        type=str,
        help='Logging level',
        default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    )

    args = parser.parse_args()
    main(args)
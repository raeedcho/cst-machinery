import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import smile_extract

def setup_logging(args, subfolder_name: str='default') -> None:
    log_dir = Path(args.log_dir) / subfolder_name
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True,parents=True)
    logging.basicConfig(
        filename=log_dir/ f'{args.dataset}.log',
        level=args.loglevel,
    )
    
def load_trial_frame(args) -> pd.DataFrame:
    trialframe_dir = Path(args.trialframe_dir)
    composition_config = OmegaConf.load(args.composition_config)
    tf = smile_extract.compose_from_frames(
        meta=pd.read_parquet(trialframe_dir / args.dataset / f'{args.dataset}_{composition_config.info}.parquet'),
        trialframe_dict={
            key: pd.read_parquet(trialframe_dir / args.dataset / f'{args.dataset}_{filepart}.parquet')
            for key, filepart in composition_config.composition.items()
        }
    )
    return tf

def setup_results_dir(args, subfolder_name: str='default') -> Path:
    results_dir: Path = Path(args.results_dir) / subfolder_name / args.dataset
    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir
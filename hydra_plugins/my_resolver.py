from omegaconf import OmegaConf
from pathlib import Path

OmegaConf.register_new_resolver(
    "relpath", lambda p: str(Path(__file__).parent / ".." / p)
)
OmegaConf.register_new_resolver("max", lambda *args: max(args))
OmegaConf.register_new_resolver("sum", lambda *args: sum(args))
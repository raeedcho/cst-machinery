from argparse import Namespace
from typing import Any, Dict
from dvclive.lightning import DVCLiveLogger

class DVCLiveLFADSLogger(DVCLiveLogger):
    """
    Custom logger for LFADS that uses DVCLive to log metrics and parameters.
    This logger basically inherits from DVCLiveLogger and overrides the
    `_sanitize_params` method to handle specific LFADS parameters that are
    objects (which cannot be written to YAML files without sanitizing).
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        # logging of arrays with dimension > 1 is not supported, sanitize as string
        params = {
            k: str(v) if hasattr(v, "ndim") and v.ndim > 1 else v
            for k, v in params.items()
        }

        # logging of argparse.Namespace is not supported, sanitize as string
        params = {
            k: str(v) if isinstance(v, Namespace) else v for k, v in params.items()
        }

        # logging of non-basic objects is not supported, sanitize as string
        params = {
            k: str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v
            for k, v in params.items()
        }

        return params  # noqa: RET504
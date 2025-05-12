import argparse
import functools
from pathlib import Path
from typing import Callable, Optional

def create_default_parser(description='Run the script with the standard arguments') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Dataset name, e.g. $(monkey)_$(session_date)',
        required=True,
    )
    parser.add_argument(
        '--trialframe_dir',
        type=str,
        help='Path to parent folder containing trial frame outputs',
        default='data/trialframe/',
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        help='Path to the results directory',
        default='results/',
    )
    parser.add_argument(
        '--log_dir',
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
    parser.add_argument(
        '--composition_config',
        type=str,
        help='Path to the trialframe composition config file',
        default='conf/trialframe.yaml',
    )
    return parser

def with_parsed_args(
        parser_creator: Callable = create_default_parser,
        description: Optional[str] = None,
    ) -> Callable:
    """
    Decorator that creates an argparse.ArgumentParser using a custom description if provided.
    It parses the command-line arguments and passes them as the first argument to the decorated main function.
    """
    def decorator(main_func):
        @functools.wraps(main_func)
        def wrapper(*args, **kwargs):
            if description is not None:
                parser = parser_creator(description=description)
            else:
                parser = parser_creator()
            parsed_args, _ = parser.parse_known_args()
            return main_func(parsed_args, *args, **kwargs)
        return wrapper
    return decorator
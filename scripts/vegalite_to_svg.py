"""Convert a Vega-Lite JSON spec to SVG using vl-convert.

This script is intentionally kept free of heavy dependencies (no torch, no
scipy, no cloudpickle) so that vl-convert's embedded V8 engine runs in a clean
process with no MPS/CUDA state that could cause crashes on Apple Silicon.

It exists as a separate DVC stage so that the SVG rendering is decoupled from
the neural-data pipeline stages that load PyTorch models.
"""
import argparse
import logging
from pathlib import Path

import vl_convert as vlc

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Convert Vega-Lite JSON spec to SVG.')
    parser.add_argument('--input', type=Path, required=True, help='Path to input .json Vega-Lite spec.')
    parser.add_argument('--output', type=Path, required=True, help='Path for output .svg file.')
    args = parser.parse_args()

    logger.info(f'Converting {args.input} -> {args.output}')
    spec = args.input.read_text()
    svg = vlc.vegalite_to_svg(spec)
    args.output.write_text(svg)
    logger.info(f'Saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()

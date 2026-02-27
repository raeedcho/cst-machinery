# CST-Machinery AI Coding Guidelines

## Project Overview
**cst-machinery**: Comparative analysis of motor control machinery between externally-cued (CST) and interactive reaching movements. This is a neuroscience pipeline combining neural recording preprocessing, data transformation, and latent dynamics modeling.

### Core Data Flow
1. **Extract**: Load SMILE format monkey neural/behavioral data → parquet trial frames
2. **Smooth**: Apply temporal filtering to spike rates
3. **Prep LFADS**: Chop continuous data into overlapping segments → HDF5 tensors
4. **Train LFADS**: Deep learning model (PyTorch Lightning) on neural dynamics
5. **Merge**: Reconstruct continuous predictions from overlapping segments
6. **Analyze**: Subspace analysis (dPCA), decoding, OFC modeling

## Architecture Patterns

### Multi-Index DataFrames (Critical Pattern)
Data uses multi-level indices with `trial_id` and `time` as primary levels, but often includes additional levels like `state`, `task`, etc.
- **See**: [src/states.py](../src/states.py#L1), [tests/conftest.py](../tests/conftest.py#L5)
- **Preserve all index levels**: Functions should maintain all existing index levels unless explicitly designed to manipulate them
- Use `trialframe` package utilities: `get_index_level()`, `state_list_to_transitions()`, `multivalue_xs()`
- When slicing or transforming, explicitly preserve or document any index level changes in docstrings

### Configuration System (OmegaConf + Hydra)
- `params.yaml`: Dataset definitions, pipeline hyperparameters, directory paths
- `conf/trialframe*.yaml`: Data composition specs (which parquet files combine into trial frames)
- `conf/lfads/`: Separate YAML per dataset + shared configs (model, callbacks, datamodule)
- **Pattern**: Use `OmegaConf.load()` to parse configs; pass as `DictConfig` to functions
- DVC uses `${var}` syntax for parameter interpolation across stages

### DVC Pipeline Orchestration
- [dvc.yaml](../dvc.yaml) defines 6+ stages with `foreach: ${datasets}` per-dataset execution
- Each stage has explicit `deps`, `outs`, `params` declarations
- Scripts use argparse + `cli.create_default_parser()` wrapper to consume config params
- **Key commands**: `dvc repro` (run pipeline), `dvc dag` (view DAG), `dvc params diff`

## Development Workflow

### From Exploration to Production
This project follows a notebook-driven workflow that transitions exploratory work into reproducible pipeline stages:

1. **Exploratory Analysis** (Jupyter notebooks in `scripts/`)
   - Start with exploratory notebooks to prototype analysis on individual session datasets
   - Test hypotheses, visualize intermediate results, iterate quickly
   - Examples: `behavior_plot.py`, `dynamic_consistency.ipynb`, `ofc_inference.ipynb`

2. **Generalize & Modularize** (Move to `src/`)
   - Once an analysis takes shape, extract reusable functions into appropriate modules in `src/`
   - Add to existing module (e.g., `subspace_tools.py`) or create new module for related functions
   - Include unit tests in `tests/` with fixtures from [conftest.py](../tests/conftest.py)

3. **Scriptify & Integrate** (DVC pipeline stage)
   - Convert notebook workflow into a reproducible script (e.g., `subspace_split.py`)
   - Add stage to [dvc.yaml](../dvc.yaml) with `foreach: ${datasets}` for per-dataset execution
   - Script reads params from `params.yaml`, calls `src/` functions, logs results
   - Now reproducible across all datasets with `dvc repro`

## File Organization

### `src/` - Reusable Modules
- **[io.py](../src/io.py)**: Setup logging, load trial frames with config-based composition
- **[chop_merge.py](../src/chop_merge.py)**: Low-level numpy operations for windowing/merging overlapping segments
- **[states.py](../src/states.py)**: Movement state classification using hand kinematics + go cue logic
- **[subspace_tools.py](../src/subspace_tools.py)**: PCA, dPCA wrapper, optional dekodec integration for potent/null spaces
- **[lfads_callbacks.py](../src/lfads_callbacks.py)**: PyTorch Lightning callbacks for custom plots during training
- **[ofc_model.py](../src/ofc_model.py)**: Optimal feedback control task simulation using ioc package

### `scripts/` - Entry Points
All scripts: read DVC config params → load data → save outputs with logging
- **extract_normalized_data.py**: SMILE → parquet (calls smile_extract.compose_from_frames)
- **prep_lfads_tensors.py**: Uses [chop_merge.frame_to_chops()](../src/chop_merge.py#L90) for windowing
- **train_lfads.py**: Launches PyTorch Lightning trainer on LFADS model
- **merge_lfads_outputs.py**: Reconstructs full recording from overlapping LFADS predictions

## Key Dependencies
- **smile-extraction** (local sibling): SMILE format I/O, low-level state extraction
- **trialframe** (local sibling): Multi-index DataFrame utilities, kinematic computations
- **lfads-torch** (local sibling): LFADS neural dynamics model
- **ioc** (local sibling): Optimal feedback control task examples
- **dynamax**: State-space models (JAX backend)
- **dpca**: Demixed PCA analysis

## Testing & Validation

### Test Structure
- **[tests/conftest.py](../tests/conftest.py)**: Fixtures create synthetic MultiIndex DataFrames (simple_timeseries_df, longer_timeseries_df, small_state_series)
- **Test patterns**: [test_chop_merge.py](../tests/test_chop_merge.py) validates roundtrip (chop → merge consistency)
- Run with pytest: `pytest` (covers src/ with --cov-report=term-missing)

### Command
```bash
# From workspace root (con env: feedback-machinery)
conda run --name feedback-machinery python -m pytest
```

## Developer Conventions

### Imports & Cross-Module Usage
- Always import from `src` module internally: `from src.states import get_movement_state_renamer`
- External packages: numpy (numerical), pandas (DataFrames), omegaconf (configs), logging (diagnostics)
- Optional: dekodec (potent/null decomp), lfads-specific Lightning classes

### Function Documentation
Use Google-style docstrings with Parameters, Returns, See Also sections. Include MultiIndex level names in docstring when relevant (e.g., [src/io.py](../src/io.py#L50) get_targets).

### Logging
Use `logging.getLogger(__name__)` per-module; configure via [src/io.py setup_logging()](../src/io.py#L15) + params.yaml loglevel.

### Argparse Wrappers
Use `@with_parsed_args()` decorator [src/cli.py](../src/cli.py#L59) to auto-parse args in scripts; fall back to `create_default_parser()` manually if needed.

## Common Patterns to Reuse
1. **Load trial frame**: `load_trial_frame(args)` → returns composed DataFrame from multiple parquets
2. **Window continuous data**: `frame_to_chops(df, window_len=60, overlap=20)` creates overlapping segments for LFADS
3. **Merge predictions**: `chops_to_frame(chops, orig_frame, overlap=20, smooth_pwr=2)` inverts windowing
4. **State transitions**: `state_list_to_transitions()` → DataFrame of (trial_id, time, old_state, new_state)

## Gotchas & Debugging Tips
- **MultiIndex alignment**: After groupby or reset_index, always verify index structure matches original
- **Overlap semantics**: `chop_data` overlap is point count; `merge_chops` has smooth_pwr≥1 to blend segment seams smoothly
- **Config interpolation**: If params.yaml value not found in script, check dvc.yaml stage `params:` list
- **LFADS dataset paths**: Must match HDF5 keys exactly; verify with [scripts/train_lfads.py](../scripts/train_lfads.py) dataset_info YAML

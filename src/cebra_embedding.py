"""
CEBRA embedding utilities for neural population data.

Provides functions to convert trialframe DataFrames to/from CEBRA-compatible
numpy arrays, fit CEBRA models, and save embeddings as parquet files that
integrate with the trialframe composition system.

CEBRA (Consistent EmBeddings of high-dimensional Recordings using Auxiliary
variables) is a nonlinear dimensionality reduction method based on contrastive
learning that can incorporate behavioral labels.

See Also:
    - https://cebra.ai for CEBRA documentation
    - src.io.load_trial_frame for loading composed trialframes
    - conf/trialframe_cebra.yaml for composition config using CEBRA embeddings
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cebra
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def trialframe_to_cebra_arrays(
    trialframe: pd.DataFrame,
    neural_signal: str = "motor cortex",
    continuous_behavior_signals: Optional[List[str]] = None,
    discrete_behavior_column: Optional[str] = None,
    pad_between_trials: int = 10,
) -> Dict[str, Any]:
    """
    Extract neural and behavioral data from a trialframe into CEBRA-ready arrays.

    Concatenates all trials into continuous numpy arrays, inserting zero-padding
    between trials so that CEBRA's temporal convolution does not bleed across
    trial boundaries. Returns a dict with the arrays and an index map for
    reconstructing the trialframe MultiIndex after embedding.

    Parameters:
        trialframe: DataFrame with MultiIndex levels including ``trial_id`` and
            ``time`` (and optionally ``block``, ``state``, ``task``, etc.).
            Columns should be hierarchical with signal groups at the top level.
        neural_signal: Top-level column name for neural data
            (e.g., ``'motor cortex'``).
        continuous_behavior_signals: List of top-level column names to extract
            as continuous behavioral labels (e.g., ``['hand position',
            'hand velocity']``). Columns within each signal are concatenated.
        discrete_behavior_column: Name of an index level to extract as a
            discrete label (e.g., ``'task'`` or ``'state'``). Will be
            integer-encoded.
        pad_between_trials: Number of zero-padding rows to insert between
            trials. Should be >= the model's receptive field to prevent
            temporal bleeding. Default 10 (matches offset10-model).

    Returns:
        dict with keys:
            - ``'neural'``: np.ndarray of shape ``(total_time, n_neurons)``
            - ``'continuous_labels'``: np.ndarray of shape
              ``(total_time, n_label_dims)`` or None
            - ``'discrete_labels'``: np.ndarray of shape ``(total_time,)``
              of int or None
            - ``'index_map'``: pd.MultiIndex — the original index entries
              (excluding padding rows) in concatenation order
            - ``'data_mask'``: np.ndarray of bool, shape ``(total_time,)``.
              True for real data rows, False for padding.
            - ``'trial_ids'``: list of trial_id values in concatenation order
            - ``'discrete_label_map'``: dict mapping int codes to original
              label strings (only if discrete_behavior_column is set)
            - ``'n_neurons'``: int, number of neural channels
    """
    # Determine which index levels contain trial_id
    trial_id_level = "trial_id"

    # Get unique trial IDs preserving order
    trial_ids = trialframe.index.get_level_values(trial_id_level).unique().tolist()

    neural_chunks = []
    continuous_chunks = []
    discrete_chunks = []
    index_entries = []
    mask_chunks = []

    # Build discrete label encoding if needed
    discrete_label_map = {}
    if discrete_behavior_column is not None:
        if discrete_behavior_column in trialframe.index.names:
            all_labels = trialframe.index.get_level_values(
                discrete_behavior_column
            ).unique()
        else:
            raise ValueError(
                f"Discrete behavior column '{discrete_behavior_column}' "
                f"not found in index levels: {trialframe.index.names}"
            )
        label_to_int = {label: i for i, label in enumerate(sorted(all_labels))}
        discrete_label_map = {v: k for k, v in label_to_int.items()}

    # Get neural columns
    if neural_signal in trialframe.columns.get_level_values(0):
        neural_cols = trialframe[neural_signal]
    else:
        raise ValueError(
            f"Neural signal '{neural_signal}' not found in columns. "
            f"Available: {trialframe.columns.get_level_values(0).unique().tolist()}"
        )

    n_neurons = neural_cols.shape[1]

    for i, trial_id in enumerate(trial_ids):
        # Extract this trial's data
        trial_data = trialframe.xs(trial_id, level=trial_id_level)

        # Neural data
        trial_neural = trial_data[neural_signal].values.astype(np.float32)
        neural_chunks.append(trial_neural)

        # Track original index for this trial
        # Reconstruct the full index including trial_id
        trial_index = trial_data.index
        # Add trial_id back to create proper MultiIndex entries
        if isinstance(trial_index, pd.MultiIndex):
            new_arrays = []
            new_names = []
            # Insert trial_id at the right position
            tid_pos = trialframe.index.names.index(trial_id_level)
            for pos in range(len(trialframe.index.names)):
                if pos == tid_pos:
                    new_arrays.append([trial_id] * len(trial_index))
                    new_names.append(trial_id_level)
                elif trialframe.index.names[pos] in trial_index.names:
                    new_arrays.append(
                        trial_index.get_level_values(trialframe.index.names[pos])
                    )
                    new_names.append(trialframe.index.names[pos])
                else:
                    # This level was consumed by xs
                    new_arrays.append([trial_id] * len(trial_index))
                    new_names.append(trialframe.index.names[pos])
            reconstructed_index = pd.MultiIndex.from_arrays(
                new_arrays, names=new_names
            )
        else:
            reconstructed_index = pd.MultiIndex.from_arrays(
                [[trial_id] * len(trial_index), trial_index],
                names=[trial_id_level, trial_index.name or "time"],
            )
        index_entries.append(reconstructed_index)

        n_rows = len(trial_neural)
        mask_chunks.append(np.ones(n_rows, dtype=bool))

        # Continuous labels
        if continuous_behavior_signals:
            cont_parts = []
            for sig in continuous_behavior_signals:
                if sig in trial_data.columns.get_level_values(0):
                    cont_parts.append(trial_data[sig].values.astype(np.float32))
                else:
                    logger.warning(
                        f"Continuous signal '{sig}' not found for trial {trial_id}, skipping."
                    )
            if cont_parts:
                continuous_chunks.append(np.hstack(cont_parts))

        # Discrete labels
        if discrete_behavior_column is not None:
            trial_discrete = trial_data.index.get_level_values(
                discrete_behavior_column
            )
            discrete_chunks.append(
                np.array([label_to_int[label] for label in trial_discrete])
            )

        # Insert padding between trials (not after the last one)
        if i < len(trial_ids) - 1 and pad_between_trials > 0:
            neural_chunks.append(np.zeros((pad_between_trials, n_neurons), dtype=np.float32))
            mask_chunks.append(np.zeros(pad_between_trials, dtype=bool))

            if continuous_behavior_signals and continuous_chunks:
                n_cont = continuous_chunks[-1].shape[1] if continuous_chunks[-1].ndim > 1 else 1
                continuous_chunks.append(
                    np.zeros((pad_between_trials, n_cont), dtype=np.float32)
                )

            if discrete_behavior_column is not None:
                # Use the last trial's last label for padding (arbitrary)
                discrete_chunks.append(
                    np.full(pad_between_trials, discrete_chunks[-1][-1], dtype=int)
                )

    result = {
        "neural": np.vstack(neural_chunks),
        "continuous_labels": (
            np.vstack(continuous_chunks) if continuous_chunks else None
        ),
        "discrete_labels": (
            np.concatenate(discrete_chunks) if discrete_chunks else None
        ),
        "index_map": index_entries[0].append(index_entries[1:])
        if len(index_entries) > 1
        else index_entries[0],
        "data_mask": np.concatenate(mask_chunks),
        "trial_ids": trial_ids,
        "discrete_label_map": discrete_label_map if discrete_label_map else None,
        "n_neurons": n_neurons,
    }

    logger.info(
        f"Prepared CEBRA arrays: {result['neural'].shape[0]} total timesteps "
        f"({sum(m.sum() for m in mask_chunks if m.dtype == bool)} data + padding), "
        f"{n_neurons} neurons, {len(trial_ids)} trials"
    )

    return result


def cebra_embedding_to_dataframe(
    embedding: np.ndarray,
    data_mask: np.ndarray,
    index_map: pd.MultiIndex,
    col_prefix: str = "cebra",
) -> pd.DataFrame:
    """
    Convert a CEBRA embedding array back into a DataFrame with original MultiIndex.

    Parameters:
        embedding: np.ndarray of shape ``(total_time, output_dim)`` from
            ``cebra_model.transform()``.
        data_mask: Boolean array from ``trialframe_to_cebra_arrays`` indicating
            which rows are real data (True) vs padding (False).
        index_map: pd.MultiIndex from ``trialframe_to_cebra_arrays`` containing
            the original index entries for data rows only.
        col_prefix: Prefix for column names. Columns will be named
            ``'{col_prefix}_0'``, ``'{col_prefix}_1'``, etc.

    Returns:
        pd.DataFrame with the original trialframe MultiIndex and columns
        ``cebra_0``, ``cebra_1``, ..., ``cebra_{output_dim-1}``.
    """
    # Extract only the data rows (not padding)
    data_embedding = embedding[data_mask]

    if len(data_embedding) != len(index_map):
        raise ValueError(
            f"Embedding data rows ({len(data_embedding)}) do not match "
            f"index_map length ({len(index_map)}). Check that the same "
            f"data_mask was used."
        )

    output_dim = data_embedding.shape[1]
    columns = [f"{col_prefix}_{i}" for i in range(output_dim)]

    df = pd.DataFrame(
        data_embedding,
        index=index_map,
        columns=pd.Index(columns, name="signal"),
    )

    return df


def fit_cebra_model(
    neural_data: np.ndarray,
    continuous_labels: Optional[np.ndarray] = None,
    discrete_labels: Optional[np.ndarray] = None,
    **cebra_kwargs: Any,
) -> cebra.CEBRA:
    """
    Fit a CEBRA model with sensible defaults for electrophysiology data.

    Wraps ``cebra.CEBRA`` with defaults tuned for motor cortex recordings
    (following the macaque somatosensory cortex settings from Schneider et al.
    2023).

    Parameters:
        neural_data: np.ndarray of shape ``(time, n_neurons)``.
        continuous_labels: Optional np.ndarray of shape ``(time, n_dims)``
            for supervised (CEBRA-Behavior) or hybrid mode.
        discrete_labels: Optional np.ndarray of shape ``(time,)`` of int
            for supervised mode with discrete labels.
        **cebra_kwargs: Override any CEBRA parameter. Common overrides:
            - ``model_architecture``: default ``'offset10-model'``
            - ``batch_size``: default ``512``
            - ``max_iterations``: default ``5000``
            - ``output_dimension``: default ``8``
            - ``time_offsets``: default ``10``
            - ``conditional``: set automatically based on labels provided
            - ``temperature``: default ``1.0``

    Returns:
        Fitted ``cebra.CEBRA`` model.

    See Also:
        trialframe_to_cebra_arrays: for preparing input arrays
        cebra_embedding_to_dataframe: for converting output back to DataFrame
    """
    defaults = dict(
        model_architecture="offset10-model",
        batch_size=512,
        max_iterations=5000,
        output_dimension=8,
        time_offsets=10,
        temperature=1.0,
        temperature_mode="constant",
        device="cuda_if_available",
        verbose=True,
    )

    # Set conditional mode based on provided labels
    has_continuous = continuous_labels is not None
    has_discrete = discrete_labels is not None

    if has_continuous or has_discrete:
        defaults["conditional"] = "time_delta"
    else:
        defaults["conditional"] = "time"

    # User kwargs override defaults
    defaults.update(cebra_kwargs)

    model = cebra.CEBRA(**defaults)

    # Build fit arguments
    fit_args = [neural_data]
    if has_continuous and has_discrete:
        fit_args.extend([continuous_labels, discrete_labels])
    elif has_continuous:
        fit_args.append(continuous_labels)
    elif has_discrete:
        fit_args.append(discrete_labels)

    logger.info(
        f"Fitting CEBRA model: {defaults['model_architecture']}, "
        f"output_dim={defaults['output_dimension']}, "
        f"iterations={defaults['max_iterations']}, "
        f"conditional={defaults['conditional']}, "
        f"hybrid={'hybrid' in defaults and defaults.get('hybrid', False)}"
    )

    model.fit(*fit_args)

    return model


def save_cebra_parquet(
    embedding_df: pd.DataFrame,
    trialframe_dir: Union[str, Path],
    dataset: str,
    suffix: str = "neural-cebra-embedding",
) -> Path:
    """
    Save a CEBRA embedding DataFrame as a parquet file.

    Follows the project naming convention:
    ``{trialframe_dir}/{dataset}/{dataset}_{suffix}.parquet``

    Parameters:
        embedding_df: DataFrame with trialframe MultiIndex and CEBRA embedding
            columns, as returned by ``cebra_embedding_to_dataframe``.
        trialframe_dir: Root directory for trialframe data.
        dataset: Dataset name (e.g., ``'Dwight_2025-01-07'``).
        suffix: File suffix for naming. Default ``'neural-cebra-embedding'``.

    Returns:
        Path to the saved parquet file.
    """
    trialframe_dir = Path(trialframe_dir)
    out_dir = trialframe_dir / dataset
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{dataset}_{suffix}.parquet"
    embedding_df.to_parquet(out_path)
    logger.info(f"Saved CEBRA embedding to {out_path}")

    return out_path

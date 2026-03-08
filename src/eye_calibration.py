"""Eye tracker calibration utilities.

Provides functions to:
- Extract paired (eye position, target position) observations from
  hold periods in calibration or task trials.
- Fit a 1-D affine calibration transform from raw eye-tracker
  voltages to the workspace X (horizontal) coordinate.
- Apply the calibration to raw eye data, producing calibrated
  horizontal gaze position alongside raw pupil diameter.

.. note::

    The analog eye tracker on the Batista-lab rig outputs multiple
    channels labeled "Left Eye X", "Left Eye Y", "Right Eye X", and
    "Right Eye Y".  However, empirical analysis shows that the three
    active channels (ch0, ch1, ch2) are >0.93 correlated with each
    other, meaning they all measure essentially the same 1-D signal.
    Channel 3 ("Right Eye Y") and channel 5 ("Right Pupil") are
    flat-lined.  Because the eye tracker provides only one effective
    dimension of gaze information, this module fits a 1-D affine
    mapping from the two raw eye channels (``eye_x``, ``eye_y``) to
    the horizontal (X) workspace coordinate.  This is sufficient for
    the CST, RTT, and DCO tasks, whose visual feedback varies only
    along the horizontal axis.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_eye_calibration(
    eye_positions: np.ndarray,
    target_positions: np.ndarray,
) -> np.ndarray:
    """Fit a 1-D affine transform mapping raw eye signals to target X coordinate.

    Solves the least-squares problem::

        t_x = [e_x, e_y, 1] @ w

    where *w* is a (3,) coefficient vector (two gains + offset).

    The two raw eye channels (``eye_x`` and ``eye_y``) are both used as
    inputs because, despite being highly correlated, they jointly provide
    a marginally better linear predictor of gaze position than either
    channel alone.

    Parameters:
        eye_positions: (N, 2) array of mean raw eye (x, y) during hold.
        target_positions: (N, 2) array of corresponding target (x, y).
            Only the first column (X) is used for fitting.

    Returns:
        (1, 3) affine calibration matrix mapping ``[eye_x, eye_y, 1]``
        to calibrated X.

    Raises:
        ValueError: If fewer than 3 calibration points are provided
            (under-determined system).
    """
    eye_positions = np.asarray(eye_positions, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)

    if eye_positions.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 calibration points, got {eye_positions.shape[0]}"
        )
    if eye_positions.ndim != 2 or eye_positions.shape[1] != 2:
        raise ValueError(
            f"eye_positions must be (N, 2); got shape {eye_positions.shape}"
        )
    if target_positions.ndim != 2 or target_positions.shape[1] != 2:
        raise ValueError(
            f"target_positions must be (N, 2); got shape {target_positions.shape}"
        )
    if eye_positions.shape[0] != target_positions.shape[0]:
        raise ValueError(
            "eye_positions and target_positions must have the same number of rows; "
            f"got {eye_positions.shape[0]} and {target_positions.shape[0]}"
        )

    # Target X only (1-D calibration)
    target_x = target_positions[:, 0]

    # Augment with ones column: [e_x, e_y, 1]
    ones = np.ones((eye_positions.shape[0], 1))
    A_aug = np.hstack([eye_positions, ones])  # (N, 3)

    # Solve A_aug @ w = target_x
    w, residuals, rank, _ = np.linalg.lstsq(A_aug, target_x, rcond=None)

    calibration_matrix = w.reshape(1, 3)  # (1, 3)

    # Log quality
    predicted = A_aug @ w
    rmse = np.sqrt(np.mean((predicted - target_x) ** 2))
    logger.info(
        "Eye calibration fit (1-D): %d points, rank=%d, RMSE=%.4f",
        eye_positions.shape[0], rank, rmse,
    )

    return calibration_matrix


def apply_eye_calibration(
    raw_xy: np.ndarray,
    calibration_matrix: np.ndarray,
) -> np.ndarray:
    """Apply an affine calibration to raw eye position data (1-D).

    Parameters:
        raw_xy: (N, 2) array of raw eye (x, y) positions.
        calibration_matrix: (1, 3) affine matrix from
            :func:`fit_eye_calibration`.

    Returns:
        (N, 1) array of calibrated X positions.
    """
    calibration_matrix = np.asarray(calibration_matrix, dtype=float)
    if calibration_matrix.shape != (1, 3):
        raise ValueError(f"calibration_matrix must be (1, 3), got {calibration_matrix.shape}")

    ones = np.ones((raw_xy.shape[0], 1))
    augmented = np.hstack([raw_xy, ones])  # (N, 3)

    calibrated_x = augmented @ calibration_matrix.T  # (N, 1)

    return calibrated_x

def extract_calibration_pairs(
    trialframe: pd.DataFrame,
    targets_df: pd.DataFrame,
    hold_state: str,
    eye_columns: tuple[str, str] = ('eye_x', 'eye_y'),
    time_slice: slice = slice(pd.to_timedelta('0ms'), pd.to_timedelta('200ms')),
) -> tuple[np.ndarray, np.ndarray]:
    """Extract paired (mean eye position, target location) from hold periods.

    Uses trialframe with 'state' index level to identify hold windows,
    computes mean eye position during those windows, and pairs with
    target locations.

    Parameters:
        trialframe: DataFrame with MultiIndex including 'state' and 'time'
            levels, and columns including eye position channels.
        targets_df: Target table with index including 'trial_id' or
            'target' level and columns including ``['x', 'y']``.
        hold_state: Name of the state during which fixation occurs
            (e.g., ``'Target Hold'`` or ``'Reach Target On'``).
        eye_columns: Column names for (eye_x, eye_y) in the trialframe.
        time_slice: Time window within the hold state to average over
            (to exclude transients).

    Returns:
        ``(eye_positions, target_positions)`` — each (N, 2) arrays of
        paired observations.

    Example:
        >>> eye_pos, tgt_pos = extract_calibration_pairs(
        ...     trialframe=trial_df,
        ...     targets_df=targets,
        ...     hold_state='Reach Target On',
        ...     time_slice=slice(pd.to_timedelta('50ms'), pd.to_timedelta('200ms')),
        ... )
    """
    try:
        from trialframe import get_epoch_data, get_index_level
    except ImportError:
        raise ImportError(
            "extract_calibration_pairs requires the trialframe package. "
            "Install it with: pip install trialframe"
        )

    # Check if hold_state exists in the data
    if 'state' not in trialframe.index.names:
        raise ValueError("trialframe must have 'state' as an index level")
    
    if trialframe.empty:
        logger.warning("Empty trialframe provided")
        return np.empty((0, 2)), np.empty((0, 2))
    
    state_values = get_index_level(trialframe, 'state').unique()
    if hold_state not in state_values:
        logger.warning("Hold state '%s' not found in data", hold_state)
        return np.empty((0, 2)), np.empty((0, 2))

    # Define epoch for the hold state
    epochs = {hold_state: (hold_state, time_slice)}

    # Extract epoch data and filter to matching state/phase
    epoch_data = get_epoch_data(trialframe, epochs=epochs)
    matching_data = epoch_data.loc[
        get_index_level(epoch_data, 'state').values == 
        get_index_level(epoch_data, 'phase').values
    ]

    if matching_data.empty:
        logger.warning("No data found for hold state '%s'", hold_state)
        return np.empty((0, 2)), np.empty((0, 2))

    # Average eye position per trial during hold
    eye_data = (
        matching_data[list(eye_columns)]
        .groupby('trial_id')
        .mean()
    )

    # Join with targets
    targets_aligned = targets_df.copy()
    if 'trial_id' not in targets_aligned.index.names:
        # Assume single target per trial or need to select one
        targets_aligned = targets_aligned.groupby('trial_id').first()

    # Inner join to get matched pairs
    paired = eye_data.join(targets_df[['x', 'y']], how='inner', rsuffix='_target')

    if paired.empty:
        logger.warning("No matching eye-target pairs found")
        return np.empty((0, 2)), np.empty((0, 2))

    eye_positions = paired[list(eye_columns)].values
    target_positions = paired[['x', 'y']].values

    logger.info(
        "Extracted %d calibration pairs from hold state '%s'",
        len(eye_positions), hold_state,
    )

    return eye_positions, target_positions

def extract_rtt_calibration_pairs(
    trialframe: pd.DataFrame,
    targets_df: pd.DataFrame,
    eye_columns: tuple[str, str] = ('eye_x', 'eye_y'),
    saccade_cutoff: pd.Timedelta = pd.to_timedelta("50ms"),
    num_targets: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract calibration pairs from RTT sequential-reach trials.

    In RTT trials the monkey reaches to a sequence of targets
    (``randomtarg1`` through ``randomtargN``).  During each "Reach to
    Target N" state, the eye saccades rapidly to the new target and then
    fixates.  By discarding the first *saccade_cutoff* of each state, the
    remaining samples approximate steady fixation.  The mean eye position
    in that window is paired with the corresponding target X coordinate.

    Parameters:
        trialframe: DataFrame with MultiIndex including 'state', 'time',
            and 'trial_id' levels, containing eye position data.
        targets_df: Target table with 'target' level in index and
            columns including ``['x', 'y']``.
        eye_columns: Column names for (eye_x, eye_y) in the trialframe.
        saccade_cutoff: Duration to discard at the start of each
            reach-to-target state, to remove the saccade transient.
        num_targets: Maximum number of sequential targets to look for
            (RTT default is 8).

    Returns:
        ``(eye_positions, target_positions)`` — each ``(N, 2)`` arrays
        of paired observations (one row per reach that had sufficient
        post-saccade data).

    Example:
        >>> eye_pos, tgt_pos = extract_rtt_calibration_pairs(
        ...     trialframe=rtt_df,
        ...     targets_df=targets,
        ...     saccade_cutoff=pd.to_timedelta('50ms'),
        ... )
    """
    try:
        from trialframe import get_epoch_data, get_index_level
    except ImportError:
        raise ImportError(
            "extract_rtt_calibration_pairs requires the trialframe package. "
            "Install it with: pip install trialframe"
        )

    # Check if state exists in trialframe
    if 'state' not in trialframe.index.names:
        raise ValueError("trialframe must have 'state' as an index level")
    
    if trialframe.empty:
        logger.warning("Empty trialframe provided")
        return np.empty((0, 2)), np.empty((0, 2))

    # Define epochs for each RTT reach state
    time_slice = slice(saccade_cutoff, pd.to_timedelta('200ms'))
    all_epochs = {
        f'Reach to Target {n}': (f'Reach to Target {n}', time_slice)
        for n in range(1, num_targets + 1)
    }

    # Filter epochs to only include states that exist in the data
    state_values = get_index_level(trialframe, 'state').unique()
    epochs = {
        name: spec for name, spec in all_epochs.items()
        if name in state_values
    }
    
    if not epochs:
        logger.warning("No RTT reach states found in data")
        return np.empty((0, 2)), np.empty((0, 2))

    # Extract epoch data and filter to matching state/phase
    epoch_data = get_epoch_data(trialframe, epochs=epochs)
    matching_data = epoch_data.loc[
        get_index_level(epoch_data, 'state').values == 
        get_index_level(epoch_data, 'phase').values
    ]

    if matching_data.empty:
        logger.warning("No RTT reach data found")
        return np.empty((0, 2)), np.empty((0, 2))

    # Average eye position per trial per phase
    eye_means = (
        matching_data[list(eye_columns)]
        .groupby(['trial_id', 'phase'])
        .mean()
        .rename(
            level='phase',
            index=lambda x: x.replace('Reach to Target ', 'randomtarg'),
        )
        .rename_axis(index={'phase': 'target'})
    )

    # Join with targets on trial_id and target
    paired = eye_means.join(
        targets_df[['x', 'y']],
        how='inner',
    )

    if paired.empty:
        logger.warning("No matching RTT eye-target pairs found")
        return np.empty((0, 2)), np.empty((0, 2))

    eye_positions = paired[list(eye_columns)].values
    target_positions = paired[['x', 'y']].values

    logger.info(
        "Extracted %d RTT calibration pairs",
        len(eye_positions),
    )

    return eye_positions, target_positions
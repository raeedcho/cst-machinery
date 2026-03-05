"""Target-related utility functions for behavioral data analysis."""

import logging
import numpy as np
import pandas as pd
from trialframe import hierarchical_assign, multivalue_xs


def get_target_direction(targets: pd.DataFrame) -> pd.Series:
    """
    Compute the direction (in degrees) from start target to outer target for each trial.

    Derives the angle of the vector from 'start' to 'outer' target positions
    using arctan2 on (y, x) components.

    Parameters
    ----------
    targets : pd.DataFrame
        Target data with MultiIndex levels including 'block', 'trial_id', 'target',
        and columns 'x', 'y'. Typically loaded via :func:`src.io.get_targets`.

    Returns
    -------
    pd.Series
        Integer-valued series of target directions in degrees, indexed by
        ('block', 'trial_id'), named 'target direction'.
    """
    target_labels = set(targets.index.get_level_values('target').astype(str).unique())

    if {'start', 'outer'}.issubset(target_labels):
        target_dir = (
            targets
            .pipe(multivalue_xs, level='target', keys=['start', 'outer'])
            [['x', 'y']]
            .groupby(['block', 'trial_id', 'target'])
            .first()
            .unstack(level='target')
            .swaplevel(axis=1)
            .dropna(how='any', axis=0)
            .pipe(hierarchical_assign, {
                'relative target': lambda df: df['outer'] - df['start']
            })
            ['relative target']
            .apply(lambda row: np.arctan2(row['y'], row['x']) * 180 / np.pi, axis=1)
            .astype(int)
            .rename('target direction')
        )
        return target_dir

    logging.getLogger(__name__).warning(
        "Could not find both 'start' and 'outer' targets; using fallback target direction=0 for all trials."
    )
    fallback_idx = (
        targets
        .groupby(['block', 'trial_id'])
        .size()
        .index
    )
    return pd.Series(0, index=fallback_idx, name='target direction', dtype=int)

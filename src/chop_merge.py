import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def chop_data(data, overlap, window):
    """Rearranges an array of continuous data into overlapping segments.

    This low-level function takes a 2-D array of features measured
    continuously through time and breaks it up into a 3-D array of
    partially overlapping time segments.

    Parameters
    ----------
    data : np.ndarray
        A TxN numpy array of N features measured across T time points.
    overlap : int
        The number of points to overlap between subsequent segments.
    window : int
        The number of time points in each segment.

    Returns
    -------
    np.ndarray
        An SxTxN numpy array of S overlapping segments spanning
        T time points with N features.

    See Also
    --------
    lfads_tf2.utils.merge_chops : Performs the opposite of this operation.

    """

    shape = (
        int((data.shape[0] - window) / (window - overlap)) + 1,
        window,
        data.shape[-1],
    )
    strides = (
        data.strides[0] * (window - overlap),
        data.strides[0],
        data.strides[1],
    )
    chopped = (
        np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        .copy()
        .astype("f")
    )
    return chopped

def merge_chops(data, overlap, orig_len=None, smooth_pwr=2):
    """Merges an array of overlapping segments back into continuous data.

    This low-level function takes a 3-D array of partially overlapping
    time segments and merges it back into a 2-D array of features measured
    continuously through time.

    Parameters
    ----------
    data : np.ndarray
        An SxTxN numpy array of S overlapping segments spanning
        T time points with N features.
    overlap : int
        The number of overlapping points between subsequent segments.
    orig_len : int, optional
        The original length of the continuous data, by default None
        will cause the length to depend on the input data.
    smooth_pwr : float, optional
        The power of smoothing. To keep only the ends of chops and
        discard the beginnings, use np.inf. To linearly blend the
        chops, use 1. Raising above 1 will increasingly prefer the
        ends of chops and lowering towards 0 will increasingly
        prefer the beginnings of chops (not recommended). To use
        only the beginnings of chops, use 0 (not recommended). By
        default, 2 slightly prefers the ends of segments.

    Returns
    -------
    np.ndarray
        A TxN numpy array of N features measured across T time points.

    See Also
    --------
    lfads_tf2.utils.chop_data : Performs the opposite of this operation.

    """

    if smooth_pwr < 1:
        logger.warning(
            "Using `smooth_pwr` < 1 for merging " "chops is not recommended."
        )

    merged = []
    full_weight_len = data.shape[1] - 2 * overlap
    # Create x-values for the ramp
    x = (
        np.linspace(1 / overlap, 1 - 1 / overlap, overlap)
        if overlap != 0
        else np.array([])
    )
    # Compute a power-function ramp to transition
    ramp = 1 - x ** smooth_pwr
    ramp = np.expand_dims(ramp, axis=-1)
    # Compute the indices to split up each chop
    split_ixs = np.cumsum([overlap, full_weight_len])
    for i in range(len(data)):
        # Split the chop into overlapping and non-overlapping
        first, middle, last = np.split(data[i], split_ixs)
        # Ramp each chop and combine it with the previous chop
        if i == 0:
            last = last * ramp
        elif i == len(data) - 1:
            first = first * (1 - ramp) + merged.pop(-1)
        else:
            first = first * (1 - ramp) + merged.pop(-1)
            last = last * ramp
        # Track all of the chops in a list
        merged.extend([first, middle, last])

    merged = np.concatenate(merged)
    # Indicate unmodeled data with NaNs
    if orig_len is not None and len(merged) < orig_len:
        nans = np.full((orig_len - len(merged), merged.shape[1]), np.nan)
        merged = np.concatenate([merged, nans])

    return merged

def frame_to_chops(neural_frame: pd.DataFrame, window_len: int, overlap: int) -> pd.Series:
    """Prepare neural tensors for LFADS training from trial frame-formatted data.

    Parameters
    ----------
    neural_frame : pd.DataFrame
        a dataframe of neural data to be chopped, indexed by trial_id and time into trial.
    window_len : int
        The length of the window to chop the data into.
    overlap : int
        The overlap between windows.

    Returns
    -------
    pd.Series
        The prepared neural tensors, indexed by trial_id and chop_id.
    """
    
    tensors = (
        neural_frame
        .groupby('trial_id')
        .apply(lambda df: chop_data(df.values, overlap=overlap, window=window_len)) # type: ignore
    )
    chops = pd.concat(
        [pd.Series(list(tensor)) for tensor in tensors],
        axis=0,
        keys=tensors.index,
        names=['trial_id','chop_id']
    )
    return chops

def chops_to_frame(chops: pd.Series, orig_frame: pd.DataFrame, overlap: int, smooth_pwr: float) -> pd.DataFrame:
    """Convert chops back to a DataFrame of neural data.

    Parameters
    ----------
    chops : pd.Series
        The chops to convert back to a DataFrame.
    window_len : int
        The length of the window used to chop the data.
    overlap : int
        The overlap between windows.
    orig_frame : pd.DataFrame
        The original DataFrame of neural data.

    Returns
    -------
    pd.DataFrame
        The converted DataFrame of neural data.
    """

    merged_chops = pd.concat(
        [
            pd.DataFrame(
                merge_chops(
                    np.stack(trial_chops.values), # type: ignore
                    overlap=overlap,
                    smooth_pwr=2,
                    orig_len=orig_frame.groupby('trial_id').get_group(trial_id).shape[0],
                ),
                columns=orig_frame.columns,
                index=orig_frame.groupby('trial_id').get_group(trial_id).index,
            )
            for trial_id,trial_chops in chops.groupby('trial_id')
        ],
        axis=0,
    )

    return merged_chops
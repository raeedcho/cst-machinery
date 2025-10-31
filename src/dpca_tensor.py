import pandas as pd
import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist
from typing import Optional

def make_dpca_tensor(trialframe: pd.DataFrame,conditions: list[str]) -> np.ndarray:
    """
    Convert the input data to a tensor format suitable for dPCA.
    """
    xr_obj = (
        trialframe
        .groupby(conditions)
        .mean()
        .stack()
        .reorder_levels(['channel'] + conditions)
        .to_xarray()
    )
    if isinstance(xr_obj, xr.DataArray):
        return xr_obj.to_numpy()
    # If it's a Dataset (e.g., due to top-level column names), collapse to DataArray
    return xr_obj.to_array().squeeze().to_numpy()

def make_dpca_tensor_simple(trialframe: pd.DataFrame,conditions: list[str]) -> np.ndarray:
    """
    Convert the input data to a tensor format suitable for dPCA.
    Note: This version for some reason does not output the same tensor as make_dpca_tensor.
    dPCA also works worse on it for some reason.
    """
    xr_obj = (
        trialframe
        .stack()
        .groupby(['channel'] + conditions).mean()
        .to_xarray()
    )
    if isinstance(xr_obj, xr.DataArray):
        return xr_obj.to_numpy()
    return xr_obj.to_array().squeeze().to_numpy()

def make_dpca_trial_tensor(trialframe: pd.DataFrame,conditions: list[str]) -> np.ndarray:
    """
    Convert the input data to a tensor format suitable for dPCA.
    This version outputs a tensor with trials as the first dimension,
    for dPCA auto-regularization.

    It produces bad results for some reason...
    """
    tensor = (
        trialframe
        .stack()
        .to_frame(name='activity') # type: ignore
        .assign(**{
            'trial num': lambda df: df.groupby(['channel']+conditions).cumcount(),
        })
        .set_index('trial num',append=True)
        .groupby(['trial num','channel'] + conditions).mean()
        .to_xarray()
        ['activity']
        .to_numpy()
    )

    return tensor

def hold_mahal_distance(points: pd.DataFrame, reference: pd.DataFrame, projmat: Optional[np.ndarray]=None) -> pd.DataFrame:
    if projmat is None:
        num_dims = points.shape[1]
        projmat = np.eye(num_dims)

    projected_ref = reference @ projmat
    projected_points = points @ projmat

    ref_mean = projected_ref.mean().values[np.newaxis,:]
    ref_cov = projected_ref.cov().values
    
    return pd.DataFrame(
        cdist(projected_points,ref_mean,metric='mahalanobis',VI=np.linalg.pinv(ref_cov)),
        index=points.index,
    )
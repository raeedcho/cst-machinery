import pandas as pd
import numpy as np

def get_index_level(df,level=None):
    if level is None:
        level = df.index.names
    return df.reset_index(level=level)[level]

def multivalue_xs(df: pd.DataFrame,keys: list,level,**kwargs) -> pd.DataFrame:
    possible_keys = df.groupby(level=level).groups.keys()
    return pd.concat([df.xs(key=key,level=level,drop_level=False,**kwargs) for key in keys if key in possible_keys])

def hierarchical_assign(df,assign_dict):
    '''
    Extends pandas.DataFrame.assign to work with hierarchical columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to assign to
    assign_dict : dict of pandas.DataFrame or callable
        dictionary of dataframes to assign to df
    '''
    return (
        df
        .join(
            pd.concat(
                [val(df) if callable(val) else val for val in assign_dict.values()], # type: ignore
                axis=1,
                keys=assign_dict.keys(),
                names=['signal','channel'],
            ) # type: ignore
        )
    )

def make_dpca_tensor(trialframe: pd.DataFrame,conditions: list[str]) -> np.ndarray:
    """
    Convert the input data to a tensor format suitable for dPCA.
    """
    tensor = (
        trialframe
        .groupby(conditions)
        .mean()
        .stack()
        .reorder_levels(['channel'] + conditions)
        .to_xarray()
        .to_numpy()
    )

    return tensor

def make_dpca_tensor_simple(trialframe: pd.DataFrame,conditions: list[str]) -> np.ndarray:
    """
    Convert the input data to a tensor format suitable for dPCA.
    Note: This version for some reason does not output the same tensor as make_dpca_tensor.
    dPCA also works worse on it for some reason.
    """
    tensor = (
        trialframe
        .stack()
        .groupby(['channel'] + conditions).mean()
        .to_xarray()
        .to_numpy()
    )

    return tensor

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
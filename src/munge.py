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

def make_dpca_tensor(self,X: pd.DataFrame) -> np.ndarray:
    """
    Convert the input data to a tensor format suitable for dPCA.
    """
    index_to_drop = list(set(X.index.names)-{'trial_id','time'})
    trials = (
        X
        .reset_index(level=index_to_drop, drop=True)
        .stack()
        .reorder_levels(['trial_id','channel','time'])
        .to_xarray()
    )

    return trials.mean(dim='trial_id').to_numpy()

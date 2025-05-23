import pandas as pd
import numpy as np

def get_index_level(df,level=None):
    if level is None:
        level = df.index.names
    return df.reset_index(level=level)[level]

def multivalue_xs(df: pd.DataFrame,keys: list,level,**kwargs) -> pd.DataFrame:
    return pd.concat([df.xs(key=key,level=level,drop_level=False,**kwargs) for key in keys])

def group_average(td,keys=[]):
    return (
        td
        .stack()
        .groupby(keys+['channel'],observed=True)
        .agg('mean')
        .unstack()
        .dropna(axis=1,how='all')
    )

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
                [val(df) if callable(val) else val for val in assign_dict.values()],
                axis=1,
                keys=assign_dict.keys(),
                names=['signal','channel'],
            )
        )
    )
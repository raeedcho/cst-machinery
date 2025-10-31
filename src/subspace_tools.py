# a set of tools to manipulate and analyze geometry of high dimensional data

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
import scipy.linalg as la
try:
    import dekodec
except Exception:  # pragma: no cover - fallback for environments without dekodec
    class _DekodecPlaceholder:
        def get_potent_null(self, *args, **kwargs):
            raise ImportError("dekodec is not installed; get_potent_null is unavailable")
    dekodec = _DekodecPlaceholder()

def frac_var_explained_by_subspace(X,subspace):
    '''
    Calculate the fraction of variance explained by a subspace

    Arguments:
        X - (numpy array) data to project
        subspace - (numpy array) basis set to project onto

    Returns:
        (float) fraction of variance explained by subspace
    '''
    X_var = np.var(X,axis=0).sum()
    X_cov = np.cov(X,rowvar=False)
    return np.trace(subspace.T @ X_cov @ subspace)/X_var

def subspace_overlap_index(X,Y,var_cutoff=0.99):
    '''
    Calculate the subspace overlap index (from Elsayed et al. 2016)
    between two data matrices (X and Y), given a number of dimensions

    Arguments:
        X,Y - (numpy arrays) matrix containing data (e.g. firing rates)
            with features along columns and observations along rows
        num_dims - (int) number of dimensions to use in the subspace
            overlap calculation

    Returns:
        (float) subspace overlap index
    '''
    assert X.shape[1] == Y.shape[1], 'X and Y must have same number of features'
    assert X.ndim == 2, 'X must be a 2D array'
    assert Y.ndim == 2, 'Y must be a 2D array'

    Y_potent, Y_null = dekodec.get_potent_null(Y,num_dims=10)

    return frac_var_explained_by_subspace(X,Y_potent)

def bootstrap_subspace_overlap(signal_grouped,num_bootstraps=100,var_cutoff=0.99):
    '''
    Compute subspace overlap for each pair of tasks and epochs,
    with bootstrapping to get distributions

    Arguments:
        signal_grouped: (pandas.Series.GroupBy object) trial data signal grouped by some key (e.g. task, epoch)
        num_bootstraps: (int) number of bootstraps to perform

    Returns:
        pandas.DataFrame: dataframe with rows corresponding to each bootstrap
            of subspace overlap computed for pairs of group keys
    '''
    signal_boots = []
    for boot_id in range(num_bootstraps):
        data_td = signal_grouped.agg(
            lambda rates : np.row_stack(rates.sample(frac=1,replace=True))
        ).rename('signal')
        proj_td = signal_grouped.agg(
            lambda rates : np.row_stack(rates.sample(frac=1,replace=True))
        ).rename('signal')
        signal_pairs = (
            data_td
            .reset_index()
            .join(
                proj_td.reset_index(),
                how='cross',
                lsuffix='_data',
                rsuffix='_proj',
            )
        )

        signal_pairs['boot_id'] = boot_id
        signal_boots.append(signal_pairs)
    
    signal_boots = (
        pd.concat(signal_boots).reset_index(drop=True)
        .assign(**{
            'subspace_overlap' : lambda df : df.apply(
                lambda row : subspace_overlap_index(row['signal_data'],row['signal_proj'],var_cutoff=var_cutoff),
                axis=1
            ),
            'subspace_overlap_rand' : lambda df : df.apply(
                lambda row : subspace_overlap_index(np.random.standard_normal(row['signal_data'].shape), row['signal_proj'], var_cutoff=var_cutoff),
                axis=1
            )
        })
    )

    return signal_boots

def calculate_fraction_variance(arr):
    return np.var(arr,axis=0)/(np.var(arr,axis=0).sum())

def calc_projected_variance(X,proj_matrix):
    '''
    Calculate the variance of the data projected onto the basis set
    defined by proj_matrix
    
    Arguments:
        X - (numpy array) data to project
        proj_matrix - (numpy array) basis set to project onto
        
    Returns:
        (float) projected variance
    '''
    # Validate shapes
    assert X.ndim == 2, 'X must be a 2D array'
    assert proj_matrix.ndim == 2, 'proj_matrix must be a 2D array'
    assert X.shape[1] == proj_matrix.shape[0], 'proj_matrix must have same number of rows as X has features'

    X_cov = np.cov(X, rowvar=False)
    # Total variance in the projected subspace equals trace(P^T Cov(X) P)
    return float(np.trace(proj_matrix.T @ X_cov @ proj_matrix))

def find_potent_null_space(X,Y):
    '''
    Runs a linear regression from X to Y to find the potent and null spaces
    of the transformation that takes X to Y. For example, X could be neural
    activity and Y could be behavioral data, and potent space would be the
    neural space that correlates best with behavior.

    Arguments:
        X (numpy array) - ndarray of input data with features along columns
            and observations along rows
        Y (numpy array) - ndarray of output data with features along columns
            and observations along rows

    Returns:
        (tuple) - tuple containing:
            potent_space - (numpy array) potent space of transformation
                shape is (X.shape[1],<potent dimensionality>)
            null_space - (numpy array) null space of transformation
                shape is (X.shape[1],<null dimensionality>)
    '''
    model = LinearRegression(fit_intercept=False)
    model.fit(X,Y)

    # model.coef_ is the coefficients of the linear regression
    # (shape: (n_targets, n_features))
    
    # null space is the set of vectors such that model.coef_ @ null_vector = 0
    # null basis is of shape (n_features, null_dim)
    null_space = la.null_space(model.coef_)
    
    # potent space is the orthogonal complement of the null space
    # which is equivalent to the row space of the model.coef_ matrix
    potent_space = la.orth(model.coef_.T)

    return potent_space,null_space
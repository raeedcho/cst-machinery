import numpy as np
from .time_slice import reindex_trial_from_event
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import TruncatedSVD,PCA

class JointSubspace(BaseEstimator,TransformerMixin):
    '''
    Model to find a joint subspace given multiple datasets in the same full-D space 
    and a number of dimensions to use for each dataset.

    This model will first reduce the dimensionality of the datasets to num_dims using TruncatedSVD,
    then concatenate the resulting PCs to form the joint subspace. Lastly, it will orthogonalize
    the joint subspace using SVD. The result will be a projection matrix from full-D space to
    the joint subspace (n_features x (num_dims*num_conditions)).

    Note: This class will not remove the mean of the datasets before performing TruncatedSVD by default.

    Arguments:
        df - (pd.DataFrame) DataFrame containing data (e.g. firing rates) and condition (e.g. task).
            Data will be grouped by the provided condition column to form multiple datasets.
            Each element of df[signal] is a numpy array with features along columns
            and optionally observations along rows. These arrays will be stacked via
            np.row_stack() to form a single data matrix for each dataset.
        signal - (str) name of column in df containing data
        condition - (str) name of column in df containing condition labels
        num_dims - (int) number of dimensions to use for each data matrix to compose
            the joint subspace.
        orthogonalize - (bool) whether or not to orthogonalize the joint subspace
            using SVD. Default is True.
        remove_latent_offsets - (bool) whether or not to remove the mean of each
            dataset before projecting into the joint subspace. If True, the mean of
            each resultant latent dimension will be 0. If False, the offsets in the
            original signals (signal means) will be passed through the transformation.
            Default is True, as in normal PCA.

    Returns:
        (numpy array) projection matrix from full-D space to joint subspace
            (n_features x (num_dims*num_conditions))
    '''
    def __init__(self,n_comps_per_cond=2,orthogonalize=True,condition=None,remove_latent_offsets=True):
        '''
        Initiates JointSubspace model.
        '''
        assert condition is not None, "Must provide condition column name"

        self.n_comps_per_cond = n_comps_per_cond
        self.orthogonalize = orthogonalize
        self.condition = condition
        self.remove_latent_offsets = remove_latent_offsets

    def fit(self,X,y=None):
        '''
        Fits the joint subspace model.

        Arguments:
            X - (pd.DataFrame) DataFrame containing data (e.g. firing rates) and condition (e.g. task
                Data will be grouped by the provided condition column to form multiple datasets.
            y - unused

        Returns:
            self - the fitted transformer object
        '''

        # group data by condition
        self.conditions_ = X.groupby(self.condition).groups.keys()
        self.n_conditions_ = len(self.conditions_)
        self.n_components_ = self.n_comps_per_cond*self.n_conditions_
        self.full_mean_ = X.mean()
        self.cond_means_ = (
            X
            .groupby(self.condition)
            .agg('mean')
            - self.full_mean_
        )

        dim_red_models = (
            X
            .groupby(self.condition)
            .apply(lambda x: PCA(n_components=self.n_comps_per_cond).fit(x))
        )

        proj_mat = np.row_stack([model.components_ for model in dim_red_models])
        if not self.orthogonalize:
            self.P_ = proj_mat.T
        else:
            _,_,Vt = np.linalg.svd(proj_mat,full_matrices=False)
            self.P_ = dekodec.max_var_rotate(Vt.T,X.values)

        return self

    def transform(self,X):
        '''
        Projects data into joint subspace.

        Arguments:
            X - (pd.DataFrame)
                DataFrame containing data (e.g. firing rates) and condition (e.g. task)

        Returns:
            (pd.DataFrame) New DataFrame with an additional column containing the
                projected data (column name is f'{self.signal}_joint_pca')
        '''
        assert hasattr(self,'P_'), "Model not yet fitted"

        if self.remove_latent_offsets:
            return (
                X
                -self.full_mean_
                -self.cond_means_
            ) @ self.P_
        else:
            return X @ self.P_

class SoftnormScaler(BaseEstimator,TransformerMixin):
    def __init__(self, norm_const=5):
        self.norm_const = norm_const

    def fit(self,X,y=None):
        def get_range(arr,axis=None):
            return np.nanmax(arr,axis=axis)-np.nanmin(arr,axis=axis)
        self.activity_range_ = get_range(X,axis=0)
        return self

    def transform(self,X):
        return X / (self.activity_range_ + self.norm_const)

class BaselineShifter(BaseEstimator,TransformerMixin):
    def __init__(self, ref_event: str, ref_slice: slice, timecol: str='time'):
        self.ref_event = ref_event
        self.ref_slice = ref_slice
        self.timecol = timecol

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        baseline = (
            X
            .groupby('trial_id',group_keys=False)
            .apply(reindex_trial_from_event,event=self.ref_event,timecol=self.timecol)
            .loc[(slice(None),self.ref_slice),:]
            .groupby('trial_id')
            .agg(lambda s: np.nanmean(s,axis=0))
        )
        return X - baseline

class VarimaxTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, tol=1e-6, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,X,y=None):
        """
        Return rotation matrix that implements varimax (or quartimax) rotation.

        Adapted from _ortho_rotation from scikit-learn _factor_analysis.py module.
        """
        nrow, ncol = X.shape
        rotation_matrix = np.eye(ncol)
        var = 0

        for _ in range(self.max_iter):
            comp_rot = np.dot(X, rotation_matrix)
            tmp = comp_rot * np.transpose((comp_rot**2).sum(axis=0) / nrow)
            u, s, v = np.linalg.svd(np.dot(X.T, comp_rot**3 - tmp))
            rotation_matrix = np.dot(u, v)
            var_new = np.sum(s)
            if var != 0 and var_new < var * (1 + self.tol):
                break
            var = var_new

        # return np.dot(X, rotation_matrix).T
        self.rotation_matrix_ = rotation_matrix
        return self

    def transform(self,X):
        return X @ self.rotation_matrix_
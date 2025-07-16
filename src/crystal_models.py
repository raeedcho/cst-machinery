import numpy as np
from .timeseries import remove_baseline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import TruncatedSVD,PCA
from dPCA.dPCA import dPCA
import xarray
import pandas as pd
from typing import Union,Optional
from functools import partial
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from .time_slice import get_epoch_data
from .munge import make_dpca_tensor
import dekodec
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
        check_is_fitted(self,'P_')

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
        check_is_fitted(self, 'activity_range_')
        return X / (self.activity_range_ + self.norm_const)

def BaselineShifter(ref_event: str, ref_slice: slice, timecol: str = 'time'):
    return FunctionTransformer(
        remove_baseline,
        validate=False,
        check_inverse=False,
        accept_sparse=False,
        kw_args={'ref_event': ref_event, 'ref_slice': ref_slice, 'timecol': timecol},
    )

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

class DataFrameTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self,X,y=None):
        self.transformer.fit(X,y)
        return self

    def transform(self, X):
        output = self.transformer.transform(X)
        return pd.DataFrame(
            output,
            index=X.index,
            columns=range(output.shape[1]),
        )

class CISFinder(DataFrameTransformer):
    """
    model for trialframe data to find CIS.
    """
    def __init__(
            self,
            reference_task: str='DCO',
            move_epoch: dict[str,tuple[str,slice]] = {'move': ('Move', slice(pd.to_timedelta('-0.4s'), pd.to_timedelta('0.6s')))}
    ) -> None:
        self.reference_task = reference_task
        self.move_epoch = move_epoch
        transformer = PCA()
        super().__init__(transformer)

    def fit(self,X,y=None,**fit_kwargs):
        tensor = (
            X
            .xs(level='task',key=self.reference_task)
            .pipe(get_epoch_data,epochs=self.move_epoch)
            .groupby(['phase','time'])
            .mean()
            .values
        )
        self.transformer.fit(X=tensor)
        return self

class PhaseConcatDPCA(DataFrameTransformer):
    """
    dPCA model for trialframe data that concatenates phases of trials along the time dimension.
    Note: I originally wrote this because I thought dPCA would be better than using PCA
    on marginalized data to find CIS and direction-specific components, but it turns out
    marginalized PCA works better, for some inexplicable reason.
    """
    def __init__(self,conditions: list[str], protect: list[str]=['t'], **dpca_kwargs):
        assert conditions[-1] == 'time', "Last condition must be 'time' for PhaseConcatDPCA"
        assert 'labels' in dpca_kwargs, "Must provide 'labels' in dpca_kwargs"
        assert dpca_kwargs['labels'][-1] == 't', "Last label must be 't' for PhaseConcatDPCA"

        self.conditions = conditions
        self.protect = protect
        transformer = dPCA(**dpca_kwargs)
        transformer.protect = self.protect  # Protect the time condition
        super().__init__(transformer=transformer)

    def fit(self, X, y=None):
        tensor = (
            X
            .groupby('phase')
            .apply(
                lambda df: make_dpca_tensor(df,conditions=self.conditions)
            )
            .pipe(np.concatenate,axis=-1)  # Concatenate along the time dimension
        )
        self.transformer.fit(X=tensor)
        self.is_fitted_ = True  # Mark as fitted
        return self

    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, 'transformer')

        out = self.transformer.transform(X.values.T)
        return pd.concat(
            {
                marg: pd.DataFrame(
                    marg_proj.T,
                    index=X.index,
                )
                for marg,marg_proj in out.items()
            },
            axis=1,
        )

class MarginalizedDeconstructor(BaseEstimator,TransformerMixin):
    """
    Model to deconstruct delay center out data using marginalizations and dekodec.
    With marginalizations, decompose into direction-specific and condition-invariant components.
    With dekodec, decompose each component set into prep- and move-unique components
    and components shared across prep and move phases.
    """
    def __init__(
            self,
            n_comps_per_phase: int = 10,
            training_epochs: dict[str,tuple[str,slice]] = {
                'prep': ('Target On', slice(pd.to_timedelta('100ms'), pd.to_timedelta('400ms'))),
                'move': ('Move', slice(pd.to_timedelta('-100ms'), pd.to_timedelta('200ms'))),
            },
            dekodec_var_cutoff: float = 0.99,
        ) -> None:
        self.n_comps_per_phase = n_comps_per_phase
        self.training_epochs = training_epochs
        self.dekodec_var_cutoff = dekodec_var_cutoff

        prep_move_pipeline = Pipeline([
            ('reduction', JointSubspace(
                n_comps_per_cond=self.n_comps_per_phase,
                condition='phase',
                remove_latent_offsets=False,
            )),
            ('dekodec', dekodec.DekODec(
                var_cutoff=self.dekodec_var_cutoff,
                condition='phase',
                split_transform=True,
            )),
        ])
        self.prep_move_cis = clone(prep_move_pipeline)
        self.prep_move_dir = clone(prep_move_pipeline)

    def fit(self, X, y=None):
        dco_data = (
            X
            .xs(level='task',key='DCO')
            .pipe(get_epoch_data,epochs=self.training_epochs)
        )

        cis_marginal = (
            dco_data
            .groupby(['phase','time']).mean()
        )
        dir_marginal = (
            (dco_data-cis_marginal)
            .groupby(['target direction','phase','time']).mean()
        )
        
        self.prep_move_cis.fit(cis_marginal)
        self.prep_move_dir.fit(dir_marginal)
        self.is_fitted_ = True  # Mark as fitted

    def transform(self, X, marg: Optional[str] = None):
        
        if marg == 'cis':
            check_is_fitted(self, 'prep_move_cis')
            out = self.prep_move_cis.transform(X)
        elif marg == 'dir':
            check_is_fitted(self, 'prep_move_dir')
            out = self.prep_move_dir.transform(X)
        else:
            check_is_fitted(self, 'prep_move_cis')
            check_is_fitted(self, 'prep_move_dir')
            out = pd.concat(
                {
                    'cis': self.prep_move_cis.transform(X),
                    'dir': self.prep_move_dir.transform(X),
                },
                axis=1,
                names=['marg', 'space', 'component'],
            )

        return out

class CISDirPartition(BaseEstimator,TransformerMixin):
    """
    Model to partition data into condition-invariant and direction-specific components.
    Uses a pipeline to deconstruct the data into condition-invariant and direction-specific components.
    Pipeline consists of:
        - Reduction (JointSubspace): to reduce the dimensionality of the data, balancing by variance in each condition.
        - CIS/Dir decomposition (CISFinder): to find condition invariant signals
        - Prep/Move decomposition (DekODec): to decompose the condition invariant signals into prep- and move-unique components
    """
    def __init__(
            self,
            n_comps_per_cond: int=10,
            reference_task: str='DCO',
            training_epochs: dict[str,tuple[str,slice]] = {
                'prep': ('Target On', slice(pd.to_timedelta('100ms'), pd.to_timedelta('400ms'))),
                'move': ('Move', slice(pd.to_timedelta('-100ms'), pd.to_timedelta('200ms'))),
            },
            var_cutoff: float = 0.99,
            split_transform: bool = True,
    ):
        self.n_comps_per_cond = n_comps_per_cond
        self.reference_task = reference_task
        self.training_epochs = training_epochs
        self.var_cutoff = var_cutoff
        self.split_transform = split_transform

    def fit(self, X, y=None):
        training_data = (
            X
            .xs(level='task', key=self.reference_task)
            .pipe(get_epoch_data, epochs=self.training_epochs)
        )

        reduction_model = JointSubspace(
            n_comps_per_cond=self.n_comps_per_cond,
            condition='phase',
            remove_latent_offsets=False,
        )
        prep_move_model = dekodec.DekODec(
            var_cutoff=self.var_cutoff,
            condition='phase',
            split_transform=self.split_transform,
        )

        reduction_model.fit(training_data)
        reduced_data =reduction_model.transform(training_data)

        cis_potent,cis_null = dekodec.get_potent_null(
            (
                reduced_data
                .groupby(['phase','time']).mean()
                .values
            ),
            var_cutoff=self.var_cutoff,
        )

        cis_null_data = reduced_data @ cis_null
        prep_move_model.fit(cis_null_data)

        self.unsplit_projmat_ = reduction_model.P_
        self.subspaces_ = {
            'cis': reduction_model.P_ @ cis_potent,
            **{
                space: reduction_model.P_ @ cis_null @ projmat
                for space, projmat in prep_move_model.subspaces_.items()
            }
        }

    def transform(self,X):
        check_is_fitted(self, 'subspaces_')

        if self.split_transform:
            return self.transform_split(X)
        else:
            return self.transform_full(X)

    def transform_full(self, X):
        return X @ np.column_stack(tuple(self.subspaces_.values()))

    def transform_split(self, X):
        return pd.concat(
            {
                space: X @ proj_mat
                for space,proj_mat in self.subspaces_.items()
            },
            axis=1,
            names=['space','component'],
        )


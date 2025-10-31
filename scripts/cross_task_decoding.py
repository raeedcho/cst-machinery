import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import trialframe as tfr
from src import crystal_models
from src.cli import with_parsed_args, create_default_parser
from src.io import generic_preproc
import smile_extract
from pathlib import Path
from omegaconf import OmegaConf,DictConfig,ListConfig
from typing import Union,Optional
import logging
logger = logging.getLogger(__name__)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    cross_val_score,
    cross_val_predict,
    cross_validate,
)
from sklearn.metrics import explained_variance_score, r2_score, make_scorer
from sklearn.base import RegressorMixin,BaseEstimator
from sklearn.pipeline import make_pipeline

@with_parsed_args(
    parser_creator=create_default_parser,
    description='Cross-task decoding analysis--train on one task and test on the other.',
)
def main(args):
    dataset = args.dataset

    log_dir = Path(args.log_dir) / 'cross-task-decoding'
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True,parents=True)
    logging.basicConfig(
        filename=log_dir/ f'{dataset}.log',
        level=args.loglevel,
    )

    tf = generic_preproc(args)
    neural_data, hand_data = precondition_data(tf)

    trial_predictions = decode_single_trials(neural_data,hand_data['x'])
    trial_scores = score_single_trials(hand_data['x'],trial_predictions)
    task_scores = score_tasks(hand_data['x'],trial_predictions)
    
    task_score_heatmap = sns.heatmap(
        data = (
            task_scores
            .unstack(level='model')
        ),
        vmin=0,
        vmax=1,
        annot=True,
        annot_kws={'fontsize': 21},
        cmap='gray',
    )

    scores_plot = sns.jointplot(
        data=(
            trial_scores
            .unstack(level='model')
        ),
        x='CST-trained',
        y='RTT-trained',
        hue='task',
        hue_order=['CST','RTT'],
        palette=['C0','C1'],
        xlim=(-1,1),
        ylim=(-1,1),
        marginal_ticks=False
    )
    scores_plot.plot_marginals(sns.rugplot,height=0.1,palette=['C0','C1'])
    scores_plot.refline(x=0,y=0)

    output_folder = Path(args.results_dir) / 'cross-task-decoding' / dataset
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    heatmap_fig = task_score_heatmap.get_figure()
    if heatmap_fig is not None:
        heatmap_fig.savefig(output_folder / f'{dataset}_decoder-task-score-heatmap.svg')
    scores_plot.figure.savefig(output_folder / f'{dataset}_decoder-trial-scores-scatter.svg')

    if not (output_folder / 'trial-predictions').exists():
        (output_folder / 'trial-predictions').mkdir()
    for trial_id,true_data in hand_data.groupby('trial_id'):
        assert isinstance(true_data['x'],pd.Series), f"Only one value in trial{trial_id}"
        trial_fig = plot_trial_predictions(
            true_data=true_data['x'],
            pred_data=trial_predictions.groupby('trial_id').get_group(trial_id),
        )
        trial_fig.savefig(output_folder / 'trial-predictions' / f'{trial_id}.svg')
        plt.close(trial_fig)

def precondition_data(tf: pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    preproc = (
        tf
        .groupby('trial_id')
        .filter(lambda df: np.any(tfr.get_index_level(df,'state') == 'Go Cue'))
        # .pipe(drop_hand_drop_trials) # type: ignore
    )

    trim_pipeline = lambda df: (
        df
        .groupby('trial_id', group_keys=False)
        .apply(lambda df: tfr.reindex_trial_from_event(df, event='Go Cue'))
        .pipe(
            tfr.slice_by_time,
            time_slice=slice(pd.to_timedelta('-0.5s'),pd.to_timedelta('3s')),
            timecol='time'
        )
    )

    scale_PCA_pipeline = make_pipeline(
        crystal_models.SoftnormScaler(),
        # crystal_models.BaselineShifter(
        #     ref_event='Hold Center (Ambiguous Cue)',
        #     ref_slice=slice('-0.5 sec','0 sec'),
        #     timecol='time'
        # ),
        PCA(n_components=15),
    )
    neural_data = (
        preproc
        ['motor cortex']
        .pipe(lambda df: pd.DataFrame(
            scale_PCA_pipeline.fit_transform(df),
            index=df.index,
        ))
        .pipe(trim_pipeline)
    )

    hand_data = (
        preproc
        ['hand position']
        .groupby('trial_id',group_keys=False)
        .apply(tfr.estimate_kinematic_derivative, deriv=1, cutoff=30)
        .pipe(trim_pipeline)
    )

    return neural_data, hand_data

def drop_hand_drop_trials(trialframe: pd.DataFrame)-> pd.DataFrame:
    hand_pos: pd.DataFrame = trialframe['hand position'] # type: ignore
    scaled_hand_pos = StandardScaler().fit_transform(hand_pos)

    vert_hand_out_of_bounds = (
        (scaled_hand_pos[:,1] < -3) | (scaled_hand_pos[:,1] > 3)
    )

    # Drop any trials where vert_hand_out_of_bounds is True
    trial_ids_to_drop = hand_pos.index[vert_hand_out_of_bounds].get_level_values('trial_id').unique()
    dropped_trials = (
        tfr.multivalue_xs(
            trialframe,
            keys=trial_ids_to_drop,
            level='trial_id',
        )
        ['trial datetime']
        .groupby(['task','trial_id'])
        .count()
    )
    trialframe = trialframe.drop(index=trial_ids_to_drop, level='trial_id')

    num_trials = len(trial_ids_to_drop)
    num_total_trials = len(hand_pos.index.get_level_values('trial_id').unique())
    logger.info(f'Dropped {num_trials} of {num_total_trials} trials with out-of-bounds hand position')
    logger.debug(f'Dropped trials with timepoint count: {dropped_trials}')

    return trialframe

def decode_single_trials(predictor_data: pd.DataFrame, target_data: pd.Series)->pd.DataFrame:
    available_tasks = tfr.get_index_level(predictor_data,'task').unique()
    single_task_models ={
        f'{task}-trained': TaskTrainedDecoder(task=task)
        for task in available_tasks
    }

    from itertools import combinations
    if len(available_tasks) >= 2:
        dual_task_models = {
            f'{tasks[0]}-{tasks[1]}-trained': TaskTrainedDecoder(task=list(tasks))
            for tasks in combinations(available_tasks, 2)
        }
    else:
        dual_task_models = {}

    if len(available_tasks) >= 3:
        tri_task_models = {
            f'{tasks[0]}-{tasks[1]}-{tasks[2]}-trained': TaskTrainedDecoder(task=list(tasks))
            for tasks in combinations(available_tasks, 3)
        }
    else:
        tri_task_models = {}

    # Combine all models into a single dictionary
    models = {
        **single_task_models,
        **dual_task_models,
        **tri_task_models,
    }

    trial_predictions = pd.DataFrame(
        {
            name: cross_val_predict(
                model,
                predictor_data,
                target_data,
                cv=LeaveOneGroupOut(),
                groups=target_data.index.get_level_values('trial_id'),
            )
            for name, model in models.items()
        },
        columns=pd.Index(list(models.keys()),name='model'),
        index=target_data.index,
    )

    return trial_predictions

def score_single_trials(true_data: pd.Series,pred_data: pd.DataFrame)->pd.Series:
    return score_groups(true_data,pred_data,['trial_id','task','model'])

def score_tasks(true_data: pd.Series,pred_data: pd.DataFrame)->pd.Series:
    return score_groups(true_data,pred_data,['task','model'])

def score_groups(true_data: pd.Series,pred_data: pd.DataFrame, grouper: Union[str,list[str]])->pd.Series:
    return (
        pred_data
        .stack(level='model')
        .to_frame('predicted') # type: ignore
        .assign(true=true_data)
        .groupby(grouper)
        .apply(lambda group_set: r2_score(group_set['true'],group_set['predicted'])) # type: ignore
    )

def plot_trial_predictions(true_data: pd.Series,pred_data: pd.DataFrame)-> Figure:
    data = (
        pred_data
        .assign(true=true_data)
        .stack(level='model')
        .to_frame('hand velocity')
    ) # type: ignore
    task = tfr.get_index_level(data,'task').unique()
    assert len(task)==1, 'Data contains multiple trials'

    fig,ax = plt.subplots(figsize=(6,3))
    sns.lineplot(
        data=data,
        x='time',
        y='hand velocity',
        hue='model',
        hue_order=['true','Dual','CST-trained','RTT-trained'],
        palette=['k','0.5','C0','C1'],
        ax=ax,
    )
    ax.set(
        xlabel='time (s)',
        ylabel='hand velocity (cm/s)',
        title=tfr.get_index_level(data,'task').unique(),
    )
    ax.axvline(0, color='k', linestyle='--')
    sns.despine(ax=ax,trim=True)
    
    return fig

class TaskTrainedDecoder(BaseEstimator,RegressorMixin):
    """
    A linear regression model trained on data from only a single task.
    """
    def __init__(self,task: Optional[Union[str,list[str]]]=None,**kwargs):
        assert isinstance(task,(str,list)), 'task must be a string or list of strings'
        if isinstance(task,str):
            task = [task]
        assert all(isinstance(t,str) for t in task), 'task must be a string or list of strings'

        self.task = task
        self.model = LinearRegression(**kwargs)
    
    def fit(self,X,y,**kwargs):
        if isinstance(y,pd.Series):
            y = y.to_frame()

        self.output_columns = y.columns

        if self.task is not None:
            X_task = tfr.multivalue_xs(X,level='task',keys=self.task)
            y_task = tfr.multivalue_xs(y,level='task',keys=self.task)
        else:
            X_task = X
            y_task = y

        self.model.fit(X_task,y_task)
        return self

    def predict(self,X,**kwargs):
        return (
            pd.DataFrame(self.model.predict(X),index=X.index,columns=self.output_columns)
            .squeeze()
        )
    
if __name__=='__main__':
    main() # type: ignore
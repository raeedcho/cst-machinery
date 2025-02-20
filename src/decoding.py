import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from . import munge

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import explained_variance_score, r2_score

from sklearn.base import RegressorMixin

def get_test_labels(df,test_size=0.25):
    gss = GroupShuffleSplit(n_splits=1,test_size=test_size)
    _,test = next(gss.split(df, groups=munge.get_index_level(df,'trial_id')))
    return np.isin(np.arange(df.shape[0]),test)

class TaskTrainedDecoder(RegressorMixin):
    def __init__(self,task,**kwargs):
        self.task = task
        self.model = LinearRegression(**kwargs)
    
    def fit(self,X,y):
        X_task = X.xs(level='task',key=self.task)
        y_task = y.xs(level='task',key=self.task)
        self.model.fit(X_task,y_task)
        return self

    def predict(self,X):
        return self.model.predict(X)

def fit_models(train_df: pd.DataFrame,signal_name,target_name,sample_size=30000):
    subsampled_training_df = (
        train_df
        .groupby('task')
        .sample(n=sample_size)
    )
    # individual models
    models = {}
    for task,task_df in subsampled_training_df.groupby('task'):
        models[task] = LinearRegression().fit(
            task_df[signal_name],
            task_df[target_name],
        )

    # Dual models
    models['Dual'] = LinearRegression().fit(
        subsampled_training_df[signal_name],
        subsampled_training_df[target_name],
    )

    return models

def model_predict(df,signal,models):
    ret_df = df.copy()
    for model_name,model in models.items():
        ret_df = ret_df.assign(**{
            f'{model_name} calibrated': model.predict(np.row_stack(ret_df[signal].values))
        })
    return ret_df

def score_models(test_df,models,signal_name,target_name):
    score_func = r2_score

    scores = pd.Series(index=pd.MultiIndex.from_product(
        [test_df.groupby('task').groups.keys(),models.keys()],
        names=['Test data','Train data']
    ))
    for task,task_df in test_df.groupby('task'):
        for model_name, model in models.items():
            scores[(task,model_name)] = model.score(task_df[signal_name],task_df[target_name])
    
    return scores

def score_trials(df,models,signal_name,target_name):
    trial_scores = (
        df
        .groupby(['task','trial_id'])
        .apply(lambda trial: pd.Series({
            f'{model_name} score': model.score(trial[signal_name],trial[target_name])
            for model_name,model in models.items()
        }))
    )
    return trial_scores
    
def run_decoder_analysis(td,signal,hand_or_cursor='Hand',pos_or_vel='velocity',trace_component=0):
    models = fit_models(td_train_test,signal,target_name=f'{hand_or_cursor} {pos_or_vel}')
    scores = score_models(td_train_test,signal,models,target_name=f'{hand_or_cursor} {pos_or_vel}')
    trial_scores = score_trials(td_train_test.loc[td_train_test['Test set']],signal,models,target_name=f'{hand_or_cursor} {pos_or_vel}')
    td_pred = (
        td_train_test
        .pipe(model_predict,signal,models)
        .melt(
            id_vars=['trial_id','Time from go cue (s)','task'],
            value_vars=[f'{hand_or_cursor} {pos_or_vel}','CST calibrated','RTT calibrated','Dual calibrated'],
            var_name='Model',
            value_name=f'{hand_or_cursor} {pos_or_vel} (cm/s)',
        )
    )
    
    #trials_to_plot=[292,145]
    #trials_to_plot=[246,169]
    #trials_to_plot=[11,169]
    trials_to_plot=[119,169]
    #trials_to_plot=td_pred.groupby('task').sample(n=1)['trial_id']
    g=sns.relplot(
        data=td_pred.loc[np.isin(td_pred['trial_id'],trials_to_plot)],
        x='Time from go cue (s)',
        y=f'{hand_or_cursor} {pos_or_vel} (cm/s)',
        hue='Model',
        # hue_order=[f'{hand_or_cursor} {pos_or_vel}','CST calibrated','RTT calibrated','Dual calibrated'],
        # palette=['k','C0','C1','0.5'],
        hue_order=[f'{hand_or_cursor} {pos_or_vel}','CST calibrated','RTT calibrated'],
        palette=['k','C0','C1'],
        kind='line',
        row='trial_id',
        row_order=trials_to_plot,
        height=4,
        aspect=2,
    )
    g.axes[0,0].set_yticks([-200,0,200])
    g.axes[0,0].set_xticks([0,2,4,6])
    sns.despine(fig=g.fig,trim=True)
    
    # fig_name = src.util.format_outfile_name(td,postfix='cst71_rtt52_vel_pred')
    # g.fig.savefig(os.path.join('../results/2022_sfn_poster/',fig_name+'.pdf'))
    
    heatmap_fig,ax = plt.subplots(1,1)
    sns.heatmap(
        ax=ax,
        data=scores.unstack()[['CST','RTT']],
        vmin=0,
        vmax=1,
        annot=True,
        annot_kws={'fontsize': 21},
        cmap='gray',
    )

    single_trial_scatter = sns.jointplot(
        data=trial_scores.reset_index(),
        y='RTT score',
        x='CST score',
        hue='task',
        hue_order=['CST','RTT'],
        palette=['C0','C1'],
        xlim=(-1,1),
        ylim=(-1,1),
        marginal_ticks=False,
    )
    single_trial_scatter.plot_marginals(sns.rugplot,height=0.1,palette=['C0','C1'])
    single_trial_scatter.refline(x=0,y=0)
    # single_trial_scatter.set(xlim=(-1,1),ylim=(-1,1))
    # single_trial_scatter.plot_joint([-1,1],[-1,1],linestyle='--',color='.5')

    return g.fig, heatmap_fig

def trial_dropout_analysis(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    signal: str,
    target_name: str,
    drop_style: str = 'max',
):
    """
    Perform trial dropout analysis by iteratively finding which training trial
    impacts the model performance the most. Iterates by removing the most impactful
    trial and re-running the analysis until there are no trials left to remove.

    Produces a numpy array of model performance as a function of the number of
    trials removed.

    Parameters:
    - df_train: Training data
    - df_test: Test data
    - signal: Signal column to include in the filtered trial data
    - target_name: Target column to predict

    Returns:
    - list of scores as trials are removed
    - Ordered list of removed trials (by trial_id)
    """

    df_test = (
        df_test
        .assign(**{'Validation': lambda df: get_test_labels(df,test_size=0.5)})
    )
    validation_df = df_test.loc[df_test['Validation']]
    df_test = df_test.loc[~df_test['Validation']]

    def drop_trial(df,trial_id):
        return df.loc[df['trial_id']!=trial_id]
    
    def train_score(train_set: pd.DataFrame,test_set: pd.DataFrame) -> float:
        return (
            LinearRegression()
            .fit(
                np.row_stack(train_set[signal]),
                train_set[target_name],
            )
            .score(
                np.row_stack(test_set[signal]),
                test_set[target_name],
            )
        )

    dropped_scores = np.zeros(len(df_train['trial_id'].unique())-1)
    dropped_trials = np.zeros(len(df_train['trial_id'].unique())-1,dtype=int)
    for dropped_trial_num in range(len(df_train['trial_id'].unique())-1):
        dropped_scores[dropped_trial_num] = train_score(df_train,df_test)

        if drop_style == 'random':
            dropped_trials[dropped_trial_num] = np.random.choice(df_train['trial_id'].unique())
        else:
            score_without_trial: dict(float) = {}
            for trial_id in df_train['trial_id'].unique():
                df_train_dropped = drop_trial(df_train,trial_id)
                score_without_trial[trial_id] = train_score(df_train_dropped,validation_df)
            
            if drop_style == 'max':
                dropped_trials[dropped_trial_num] = max(score_without_trial,key=score_without_trial.get)
            elif drop_style == 'min':
                dropped_trials[dropped_trial_num] = min(score_without_trial,key=score_without_trial.get)
            else:
                raise ValueError(f"Invalid drop_style: {drop_style}. Must be 'max', 'min', or 'random'.")

        df_train = drop_trial(df_train,dropped_trials[dropped_trial_num])

    return dropped_scores, dropped_trials

def get_trial_importance(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    signal: str,
    target_name: str,
)->pd.DataFrame:
    """
    Get the importance of each trial in the training set by training a decoder
    on individual trials of the training set and scoring with the test set.
    """
    def train_score(train_set: pd.DataFrame,test_set: pd.DataFrame) -> float:
        return (
            LinearRegression()
            .fit(
                np.row_stack(train_set[signal]),
                train_set[target_name],
            )
            .score(
                np.row_stack(test_set[signal]),
                test_set[target_name],
            )
        )

    trial_importance = (
        df_train
        .groupby('trial_id')
        .apply(lambda trial: train_score(trial,df_test))
    )

    return trial_importance
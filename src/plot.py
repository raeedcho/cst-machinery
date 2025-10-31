import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib as mpl
from matplotlib import animation
import pandas as pd

def plot_hand_trace(trial: pd.DataFrame, targets, ax=None,timecol='time',trace_component='x'):
    if ax is None:
        ax = plt.gca()

    # zero line
    ax.plot([trial[timecol][0],trial[timecol][-1]],[0,0],'-k')

    # targets
    targ_size = 10
    if not np.isnan(trial['idx_ctHoldTime']) and not np.isnan(trial['idx_pretaskHoldTime']):
        ax.add_patch(Rectangle(
            (trial[timecol][trial['idx_ctHoldTime']],-targ_size/2),
            trial[timecol][trial['idx_pretaskHoldTime']]-trial[timecol][trial['idx_ctHoldTime']],
            targ_size,
            color='0.5',
        ))

    if trial['task']=='RTT':
        if not np.isnan(trial['idx_pretaskHoldTime']) and not np.isnan(trial['idx_goCueTime']):
            ax.add_patch(Rectangle(
                (trial[timesig][trial['idx_pretaskHoldTime']],-targ_size/2),
                trial[timesig][trial['idx_goCueTime']]-trial[timesig][trial['idx_pretaskHoldTime']],
                targ_size,
                color='C1',
            ))
        for idx_targ_start,idx_targ_end,targ_loc in zip(
            trial['idx_rtgoCueTimes'].astype(int),
            trial['idx_rtHoldTimes'].astype(int),
            trial['rt_locations'][:,trace_component]-trial['ct_location'][trace_component],
        ):
            if not np.isnan(idx_targ_start) and not np.isnan(idx_targ_end):
                ax.add_patch(Rectangle(
                    (trial[timesig][idx_targ_start],targ_loc-targ_size/2),
                    trial[timesig][idx_targ_end]-trial[timesig][idx_targ_start],
                    targ_size,
                    color='C1',
                ))
    elif trial['task']=='CST':
        if not np.isnan(trial['idx_pretaskHoldTime']) and not np.isnan(trial['idx_goCueTime']):
            ax.add_patch(Rectangle(
                (trial[timesig][trial['idx_pretaskHoldTime']],-targ_size/2),
                trial[timesig][trial['idx_goCueTime']]-trial[timesig][trial['idx_pretaskHoldTime']],
                targ_size,
                color='C0',
            ))

    # cursor
    ax.plot(
        trial[timesig],
        trial['rel_cursor_pos'][:,trace_component],
        c='b',
        alpha=0.5,
    )
    
    # hand
    ax.plot(
        trial[timesig],
        trial['rel_hand_pos'][:,trace_component],
        c='k',
    )
    ax.set_ylim(-60,60)
    ax.set_ylabel('Hand position (cm)')
    ax.set_xlabel(timesig)
    sns.despine(ax=ax,trim=True)


task_colors = {'CST':'C0','RTT':'C1'}
def animate_trial_monitor_no_raster(trial):
    fig = plt.figure(figsize=(10,3.5))
    gs = mpl.gridspec.GridSpec(1,2,figure=fig,width_ratios=[1,2.5])
    monitor_ax = fig.add_subplot(gs[0,0])
    beh_ax = fig.add_subplot(gs[0,1])

    trial['trialtime'] = trial['Time from go cue (s)']

    src.plot.plot_hand_trace(trial,ax=beh_ax)
    beh_blocker = beh_ax.add_patch(Rectangle((0,-100),10,200,color='w',zorder=100))
    beh_ax.set_xlim([-1,6])
    beh_ax.set_ylim([-60,60])
    beh_ax.set_yticks([-50,50])
    beh_ax.set_xticks([0,6])
    beh_ax.set_xlabel('Time from go cue (s)')
    beh_ax.set_ylabel('Hand position (cm)')
    sns.despine(ax=beh_ax,trim=True)

    hand = monitor_ax.add_patch(Circle(trial['rel_hand_pos'][0,:2][::-1],5,color='k',fill=False))
    cursor = monitor_ax.add_patch(Circle(trial['rel_cursor_pos'][0,:2][::-1],5,zorder=100,color='y'))
    center_target=monitor_ax.add_patch(Rectangle((-5,-5),10,10,color='0.25'))
    if trial['task'] == 'RTT':
        targets = [
            monitor_ax.add_patch(Rectangle(
                targ_loc[:2][::-1]-trial['ct_location'][:2][::-1]-[5,5],
                10,
                10,
                color='C1',
                visible=False,
            )) for targ_loc in trial['rt_locations']
        ]

    monitor_ax.set_xlim([-60,60])
    monitor_ax.set_ylim([-60,60])
    monitor_ax.set_xticks([])
    monitor_ax.set_yticks([])
    sns.despine(ax=monitor_ax,left=True,bottom=True)

    plt.tight_layout()

    def init_plot():
        beh_blocker.set(x=0)
        return [beh_blocker]

    def animate(frame_time):
        beh_blocker.set(x=frame_time)

        frame_idx = int(frame_time/trial['bin_size']) + trial['idx_goCueTime']
        hand.set(center=trial['rel_hand_pos'][frame_idx,:2][::-1])
        cursor.set(center=trial['rel_cursor_pos'][frame_idx,:2][::-1])

        if frame_idx<trial['idx_pretaskHoldTime']:
            center_target.set(color='0.25')
        elif (trial['idx_pretaskHoldTime']<=frame_idx) and (frame_idx<trial['idx_goCueTime']):
            center_target.set(color=task_colors[trial['task']])
        elif frame_idx>=trial['idx_goCueTime']:
            center_target.set(visible=False)

        if trial['task']=='RTT':
            idx_targ_start = trial['idx_rtgoCueTimes']
            idx_targ_end = trial['idx_rtHoldTimes']
            on_targs = (idx_targ_start<frame_idx) & (frame_idx<idx_targ_end)
            for target,on_indicator in zip(targets,on_targs):
                target.set(visible=on_indicator)

        if trial['task']=='CST':
            if (trial['idx_goCueTime']<=frame_idx) and (frame_idx<trial['idx_cstEndTime']):
                center_target.set(visible=True)
                center_target.set(color='0.25')
                cursor.set(color='C0')
            elif frame_idx>=trial['idx_cstEndTime']:
                center_target.set(visible=False)
                cursor.set(color='y')
    
        return [beh_blocker]

    frame_interval = 30 #ms
    frames = np.arange(trial['trialtime'][0],trial['trialtime'][-1],frame_interval*1e-3)
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init_plot,
        frames = frames,
        interval = frame_interval,
        blit = True,
    )

    return anim

for trial_to_plot in [52,71]:
    anim = animate_trial_monitor_no_raster(td.loc[td['trial_id']==trial_to_plot].squeeze())
    anim_name = src.util.format_outfile_name(td,postfix=f'trial_{trial_to_plot}_monitor_anim')
    anim.save(os.path.join('../results/2022_sfn_poster/',anim_name+'.mp4'),writer='ffmpeg',fps=30,dpi=400)
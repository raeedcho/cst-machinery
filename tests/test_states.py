import numpy as np
import pandas as pd

from src.states import reassign_state, get_movement_state_renamer


def make_hand_pos_with_states():
    # Build minimal DataFrame: trial_id,time,state in index; x,y columns
    trial_id = [1] * 6
    times = pd.to_timedelta(np.arange(6) * 0.1, unit="s")
    states = ["Fixation", "Fixation", "Go Cue", "Go Cue", "Go Cue", "Go Cue"]
    idx = pd.MultiIndex.from_arrays([trial_id, times, states], names=["trial_id", "time", "state"])

    # Position: start near (0,0), after Go Cue move outward beyond radius
    pos = np.array([
        [0.0, 0.0],
        [0.5, 0.2],  # still inside
        [0.8, 0.3],  # go cue starts here
        [2.5, 0.0],  # out of start radius
        [3.0, 0.0],
        [3.0, 0.1],
    ])
    hand_pos = pd.DataFrame(pos, index=idx, columns=["x", "y"])
    return hand_pos


def test_reassign_state_simple():
    # Minimal reassign: change states to all "New"
    hand_pos = make_hand_pos_with_states()
    new_states = pd.Series(["New"] * len(hand_pos), index=hand_pos.index.droplevel("state"))
    out = reassign_state(hand_pos, new_state=new_states)
    assert (out.index.get_level_values("state") == "New").all()


def test_get_movement_state_renamer_simple_case():
    hand_pos = make_hand_pos_with_states()

    # Start target info: same trial, center at (0,0), radius small
    sti_idx = pd.MultiIndex.from_product([[1], ["start"]], names=["trial_id", "target"])  # conforms with get_targets
    start_target_info = pd.DataFrame({"x": [0.0], "y": [0.0], "radius": [0.5]}, index=sti_idx)

    new_states = get_movement_state_renamer(hand_pos, start_target_info, go_state="Go Cue")
    # After Go Cue, once distance > radius+cursor (0.5+2), we expect 'Move'
    # Our sample becomes >2.5 at the 4th time row.
    assert new_states.loc[(1, pd.to_timedelta(0.3, unit="s"))] == "Go Cue"
    # The first time after exceeding threshold should be labeled 'Move'
    later_states = new_states.loc[(1, pd.to_timedelta(0.4, unit="s")) :]
    assert (later_states == "Move").any()
"""Tests for eye calibration utilities."""

import numpy as np
import pandas as pd
import pytest

from src.eye_calibration import (
    apply_eye_calibration,
    extract_calibration_pairs,
    extract_rtt_calibration_pairs,
    fit_eye_calibration,
)


class TestFitEyeCalibration:
    def test_identity_x_transform(self):
        """If eye_x equals target_x, the matrix should recover target X."""
        eye = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        tgt = np.array([[0, 99], [1, 99], [0, 99], [1, 99]], dtype=float)
        M = fit_eye_calibration(eye, tgt)
        assert M.shape == (1, 3)
        # Should map [eye_x, eye_y, 1] → target_x
        predicted = np.hstack([eye, np.ones((4, 1))]) @ M.T
        np.testing.assert_allclose(predicted.ravel(), tgt[:, 0], atol=1e-10)

    def test_known_affine(self):
        """Recover a known affine transform (1-D)."""
        rng = np.random.default_rng(123)
        # Ground truth: target_x = 2*eye_x + 3*eye_y + 10
        true_w = np.array([[2.0, 3.0, 10.0]])

        eye_pts = rng.uniform(-5, 5, (20, 2))
        ones = np.ones((20, 1))
        target_x = (np.hstack([eye_pts, ones]) @ true_w.T).ravel()
        target_pts = np.column_stack([target_x, np.zeros(20)])  # y unused

        recovered_M = fit_eye_calibration(eye_pts, target_pts)
        np.testing.assert_allclose(recovered_M, true_w, atol=1e-10)

    def test_scale_and_offset(self):
        """Recover a scale + offset mapping."""
        # target_x = 5 * eye_x + 100
        eye = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        tgt = np.array([[100, 0], [105, 0], [110, 0], [115, 0]], dtype=float)
        M = fit_eye_calibration(eye, tgt)
        predicted = np.hstack([eye, np.ones((4, 1))]) @ M.T
        np.testing.assert_allclose(predicted.ravel(), tgt[:, 0], atol=1e-10)

    def test_too_few_points_raises(self):
        """Need at least 3 points."""
        pts = np.array([[0, 0], [1, 1]], dtype=float)
        with pytest.raises(ValueError, match="at least 3"):
            fit_eye_calibration(pts, pts)

    def test_mismatched_row_count_raises(self):
        eye = np.array([[0, 0], [1, 1], [2, 2]])
        target = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="same number of rows"):
            fit_eye_calibration(eye, target)


class TestApplyEyeCalibration:
    def test_identity_x(self):
        """eye_x passthrough when matrix is [1, 0, 0]."""
        M = np.array([[1, 0, 0]], dtype=float)
        raw_xy = np.array([[1.0, 3.0], [2.0, 4.0]])  # (n_samples, 2)
        result = apply_eye_calibration(raw_xy, M)
        # Result should be (N, 1): [1*eye_x + 0*eye_y + 0]
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result[:, 0], [1.0, 2.0])

    def test_scale_and_offset(self):
        """Test scaling + translation (1-D)."""
        M = np.array([[2, 3, 10]], dtype=float)  # 2*eye_x + 3*eye_y + 10
        raw_xy = np.array([[1.0, 2.0]])  # (1, 2)
        result = apply_eye_calibration(raw_xy, M)
        # 2*1 + 3*2 + 10 = 18
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(18.0)

    def test_nan_propagation(self):
        """NaN eye values should produce NaN calibrated values."""
        M = np.array([[1, 0, 0]], dtype=float)
        raw_xy = np.array([[np.nan, np.nan]])  # (1, 2)
        result = apply_eye_calibration(raw_xy, M)
        assert result.shape == (1, 1)
        assert np.isnan(result).all()

    def test_bad_matrix_shape_raises(self):
        M = np.eye(3)  # Wrong shape (3,3) instead of (1,3)
        raw_xy = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match=r"\(1, 3\)"):
            apply_eye_calibration(raw_xy, M)


class TestExtractCalibrationPairs:
    @pytest.fixture
    def calib_trialframe(self):
        """Build synthetic trialframe with eye data and 'state' index."""
        n_per_trial = 50
        trial_ids = [1, 2, 3]

        # Build trialframe with (trial_id, state, time) index
        frames = []
        for tid in trial_ids:
            times = pd.timedelta_range('0ms', periods=n_per_trial, freq='10ms', name='time')
            # First half 'Move', second half 'Target Hold'
            states = ['Move'] * (n_per_trial // 2) + ['Target Hold'] * (n_per_trial - n_per_trial // 2)
            
            idx = pd.MultiIndex.from_arrays(
                [[tid] * n_per_trial, states, times],
                names=['trial_id', 'state', 'time'],
            )
            
            rng = np.random.default_rng(tid)
            frames.append(pd.DataFrame(
                {'eye_x': rng.normal(tid, 0.01, n_per_trial),
                 'eye_y': rng.normal(tid * 2, 0.01, n_per_trial),
                 'pupil': np.ones(n_per_trial) * 5.0},
                index=idx,
            ))
        
        trialframe = pd.concat(frames)
        
        # Targets: one reach target per trial
        targets_frames = []
        for tid in trial_ids:
            targets_frames.append(pd.DataFrame(
                {'x': [10.0 * tid], 'y': [20.0 * tid],
                 'radius': [3.0], 'height': [3.0]},
                index=pd.Index([tid], name='trial_id'),
            ))
        targets_df = pd.concat(targets_frames)

        return trialframe, targets_df

    def test_basic_extraction(self, calib_trialframe):
        trialframe, targets_df = calib_trialframe
        eye_pos, tgt_pos = extract_calibration_pairs(
            trialframe, targets_df,
            hold_state='Target Hold',
        )
        # Should get 3 pairs (all 3 trials)
        assert eye_pos.shape == (3, 2)
        assert tgt_pos.shape == (3, 2)

    def test_no_matching_state(self, calib_trialframe):
        trialframe, targets_df = calib_trialframe
        eye_pos, tgt_pos = extract_calibration_pairs(
            trialframe, targets_df,
            hold_state='Nonexistent State',
        )
        assert eye_pos.shape == (0, 2)

    def test_custom_eye_columns(self, calib_trialframe):
        trialframe, targets_df = calib_trialframe
        # Rename eye columns
        trialframe = trialframe.rename(columns={'eye_x': 'gaze_x', 'eye_y': 'gaze_y'})
        eye_pos, tgt_pos = extract_calibration_pairs(
            trialframe, targets_df,
            hold_state='Target Hold',
            eye_columns=('gaze_x', 'gaze_y'),
        )
        assert eye_pos.shape == (3, 2)


class TestExtractRttCalibrationPairs:
    """Tests for extract_rtt_calibration_pairs."""

    @pytest.fixture
    def rtt_data(self):
        """Synthetic RTT trialframe with 2 RTT trials, 3 reaches each."""
        n_per_state = 20  # 200 ms states at 10ms bins
        saccade_bins = 5  # 50 ms
        num_reaches = 3
        trial_ids = [1, 2, 3]  # 1 and 2 are RTT, 3 is CST

        # Target X positions per reach (same for both trials, for simplicity)
        target_xs = [30.0, 60.0, 90.0]

        # Meta - needed to verify task filters still work
        meta_df = pd.DataFrame(
            {'task': ['RTT', 'RTT', 'CST']},
            index=pd.Index(trial_ids, name='trial_id'),
        )

        eye_frames = []
        target_frames = []

        for tid in trial_ids:
            states_list = []
            times_list = []
            eye_x_list = []
            eye_y_list = []

            t_offset = 0
            # Build states: "Reach to Target 1", "Reach to Target 2", ...
            for n in range(1, num_reaches + 1):
                for i in range(n_per_state):
                    t_ms = t_offset + i * 10
                    times_list.append(pd.to_timedelta(t_ms, unit='ms'))
                    states_list.append(f'Reach to Target {n}')
                    # Eye position: saccade for first 5 bins, then steady fixation
                    if i < saccade_bins:
                        eye_x_list.append(0.0)  # saccade noise
                        eye_y_list.append(0.0)
                    else:
                        # Steady fixation: map target_x through inverse of a known affine
                        # target_x = 2*eye_x + 3*eye_y + 10
                        # so eye_x = (target_x - 10) / 2, eye_y = 0
                        eye_x_list.append((target_xs[n - 1] - 10) / 2.0)
                        eye_y_list.append(0.0)
                t_offset += n_per_state * 10

            n_total = len(times_list)
            # Build MultiIndex with state in index
            idx = pd.MultiIndex.from_arrays(
                [[tid] * n_total, states_list, times_list],
                names=['trial_id', 'state', 'time'],
            )
            eye_frames.append(pd.DataFrame(
                {'eye_x': eye_x_list, 'eye_y': eye_y_list,
                 'pupil': [5.0] * n_total},
                index=idx,
            ))

            # Targets for this trial
            target_rows = []
            target_idx = []
            for n in range(1, num_reaches + 1):
                target_rows.append({'x': target_xs[n - 1], 'y': 895.0,
                                     'radius': 7.0, 'height': 50.0})
                target_idx.append((tid, f'randomtarg{n}'))
            target_frames.append(pd.DataFrame(
                target_rows,
                index=pd.MultiIndex.from_tuples(target_idx,
                                                 names=['trial_id', 'target']),
            ))

        trialframe = pd.concat(eye_frames)
        targets_df = pd.concat(target_frames)

        return trialframe, targets_df, meta_df

    def test_basic_extraction(self, rtt_data):
        trialframe, targets_df, meta_df = rtt_data
        # Filter to RTT trials manually since function no longer takes meta_df
        rtt_trial_ids = meta_df[meta_df['task'] == 'RTT'].index
        trialframe_rtt = trialframe.loc[rtt_trial_ids]
        
        eye_pos, tgt_pos = extract_rtt_calibration_pairs(
            trialframe_rtt, targets_df,
            saccade_cutoff=pd.to_timedelta('50ms'),
            num_targets=3,
        )
        # 2 RTT trials × 3 reaches = 6 pairs
        assert eye_pos.shape == (6, 2)
        assert tgt_pos.shape == (6, 2)

    def test_saccade_cutoff_removes_noise(self, rtt_data):
        """Post-saccade means should reflect fixation, not saccade noise."""
        trialframe, targets_df, meta_df = rtt_data
        rtt_trial_ids = meta_df[meta_df['task'] == 'RTT'].index
        trialframe_rtt = trialframe.loc[rtt_trial_ids]
        
        eye_pos, tgt_pos = extract_rtt_calibration_pairs(
            trialframe_rtt, targets_df,
            saccade_cutoff=pd.to_timedelta('50ms'),
            num_targets=3,
        )
        # Post-saccade eye_x should be steady at (target_x - 10) / 2
        expected_eye_x = [(30 - 10) / 2, (60 - 10) / 2, (90 - 10) / 2]
        for i in range(3):  # first trial
            assert eye_pos[i, 0] == pytest.approx(expected_eye_x[i])

    def test_target_positions_correct(self, rtt_data):
        trialframe, targets_df, meta_df = rtt_data
        rtt_trial_ids = meta_df[meta_df['task'] == 'RTT'].index
        trialframe_rtt = trialframe.loc[rtt_trial_ids]
        
        eye_pos, tgt_pos = extract_rtt_calibration_pairs(
            trialframe_rtt, targets_df,
            num_targets=3,
        )
        # Targets for trial 1: 30, 60, 90
        np.testing.assert_allclose(tgt_pos[:3, 0], [30.0, 60.0, 90.0])
        np.testing.assert_allclose(tgt_pos[:3, 1], [895.0, 895.0, 895.0])

    def test_no_rtt_trials(self, rtt_data):
        trialframe, targets_df, meta_df = rtt_data
        # Get only CST trial
        cst_trial_ids = meta_df[meta_df['task'] == 'CST'].index
        trialframe_cst = trialframe.loc[cst_trial_ids]
        
        eye_pos, tgt_pos = extract_rtt_calibration_pairs(
            trialframe_cst, targets_df,
        )
        # CST trial in fixture still has "Reach to Target" states, so we get 3 pairs
        # (Function is task-agnostic, only looks at states present)
        assert eye_pos.shape == (3, 2)

    def test_missing_target_skipped(self, rtt_data):
        trialframe, targets_df, meta_df = rtt_data
        rtt_trial_ids = meta_df[meta_df['task'] == 'RTT'].index
        trialframe_rtt = trialframe.loc[rtt_trial_ids]
        
        # Ask for 5 targets but only 3 exist → should get 6 pairs (3 per trial)
        eye_pos, tgt_pos = extract_rtt_calibration_pairs(
            trialframe_rtt, targets_df,
            num_targets=5,
        )
        assert eye_pos.shape == (6, 2)

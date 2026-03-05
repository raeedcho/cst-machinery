"""Tests for src.cebra_embedding — data wrangling only, not CEBRA model fitting."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.cebra_embedding import (
    trialframe_to_cebra_arrays,
    cebra_embedding_to_dataframe,
    save_cebra_parquet,
)


@pytest.fixture
def multi_trial_trialframe():
    """Create a synthetic trialframe with hierarchical columns mimicking real data.

    Structure:
        Index: (trial_id, time) — 2 trials, 20 timesteps each at 10ms
        Columns: MultiIndex with signal groups:
            ('motor cortex', 'ch1'), ('motor cortex', 'ch2'), ('motor cortex', 'ch3')
            ('hand position', 'x'), ('hand position', 'y')
    """
    n_per_trial = 20
    n_trials = 3
    n_neurons = 3

    index_arrays = []
    for tid in range(1, n_trials + 1):
        times = pd.to_timedelta(np.arange(n_per_trial) * 0.01, unit="s")
        index_arrays.append(
            pd.MultiIndex.from_arrays(
                [[tid] * n_per_trial, times],
                names=["trial_id", "time"],
            )
        )
    index = index_arrays[0].append(index_arrays[1:])

    rng = np.random.default_rng(42)
    n_total = n_per_trial * n_trials
    data = {
        ("motor cortex", "ch1"): rng.standard_normal(n_total).astype(np.float32),
        ("motor cortex", "ch2"): rng.standard_normal(n_total).astype(np.float32),
        ("motor cortex", "ch3"): rng.standard_normal(n_total).astype(np.float32),
        ("hand position", "x"): rng.standard_normal(n_total).astype(np.float32),
        ("hand position", "y"): rng.standard_normal(n_total).astype(np.float32),
    }
    columns = pd.MultiIndex.from_tuples(
        list(data.keys()), names=["signal", "channel"]
    )
    df = pd.DataFrame(
        np.column_stack(list(data.values())),
        index=index,
        columns=columns,
    )
    return df


@pytest.fixture
def trialframe_with_state(multi_trial_trialframe):
    """Trialframe with an additional 'state' index level."""
    df = multi_trial_trialframe.copy()
    n_per_trial = 20
    n_trials = 3

    # Add state index: first half 'Hold', second half 'Move' within each trial
    states = []
    for _ in range(n_trials):
        states.extend(["Hold"] * (n_per_trial // 2) + ["Move"] * (n_per_trial // 2))

    df["state"] = states
    df = df.set_index("state", append=True)
    return df


@pytest.fixture
def trialframe_with_task(multi_trial_trialframe):
    """Trialframe with 'task' and 'state' index levels (closer to real data)."""
    df = multi_trial_trialframe.copy()
    n_per_trial = 20

    tasks = ["CST"] * n_per_trial + ["RTT"] * n_per_trial + ["DCO"] * n_per_trial
    states = []
    for _ in range(3):
        states.extend(["Hold"] * (n_per_trial // 2) + ["Move"] * (n_per_trial // 2))

    df["task"] = tasks
    df["state"] = states
    df = df.set_index(["task", "state"], append=True)
    return df


class TestTrialframeToCebraArrays:
    """Tests for trialframe_to_cebra_arrays."""

    def test_basic_neural_extraction(self, multi_trial_trialframe):
        """Neural data is extracted with correct shape."""
        result = trialframe_to_cebra_arrays(
            multi_trial_trialframe, pad_between_trials=0
        )
        assert result["neural"].shape == (60, 3)  # 3 trials * 20 timesteps, 3 neurons
        assert result["n_neurons"] == 3
        assert result["continuous_labels"] is None
        assert result["discrete_labels"] is None

    def test_padding_inserted(self, multi_trial_trialframe):
        """Zero padding is inserted between trials."""
        pad = 5
        result = trialframe_to_cebra_arrays(
            multi_trial_trialframe, pad_between_trials=pad
        )
        # 3 trials * 20 + 2 gaps * 5 padding = 70
        expected_rows = 60 + 2 * pad
        assert result["neural"].shape[0] == expected_rows
        assert result["data_mask"].sum() == 60  # only real data
        assert (~result["data_mask"]).sum() == 2 * pad

    def test_index_map_length(self, multi_trial_trialframe):
        """Index map contains entries only for real data rows."""
        result = trialframe_to_cebra_arrays(
            multi_trial_trialframe, pad_between_trials=5
        )
        assert len(result["index_map"]) == 60

    def test_trial_ids_preserved(self, multi_trial_trialframe):
        """All trial IDs are present in output."""
        result = trialframe_to_cebra_arrays(multi_trial_trialframe)
        assert result["trial_ids"] == [1, 2, 3]

    def test_continuous_labels(self, multi_trial_trialframe):
        """Continuous behavioral labels are extracted correctly."""
        result = trialframe_to_cebra_arrays(
            multi_trial_trialframe,
            continuous_behavior_signals=["hand position"],
            pad_between_trials=0,
        )
        assert result["continuous_labels"] is not None
        assert result["continuous_labels"].shape == (60, 2)  # x, y

    def test_discrete_labels(self, trialframe_with_state):
        """Discrete labels from index level are extracted correctly."""
        result = trialframe_to_cebra_arrays(
            trialframe_with_state,
            discrete_behavior_column="state",
            pad_between_trials=0,
        )
        assert result["discrete_labels"] is not None
        assert result["discrete_labels"].shape == (60,)
        assert result["discrete_label_map"] is not None
        # Should have 2 unique labels: Hold, Move
        assert len(result["discrete_label_map"]) == 2

    def test_discrete_labels_integer_encoded(self, trialframe_with_task):
        """Discrete labels are integer-encoded."""
        result = trialframe_to_cebra_arrays(
            trialframe_with_task,
            discrete_behavior_column="task",
            pad_between_trials=0,
        )
        assert result["discrete_labels"].dtype in [np.int64, np.int32, int]
        unique_vals = set(result["discrete_labels"])
        assert len(unique_vals) == 3  # CST, DCO, RTT

    def test_missing_neural_signal_raises(self, multi_trial_trialframe):
        """Raises ValueError if neural signal not found."""
        with pytest.raises(ValueError, match="not found in columns"):
            trialframe_to_cebra_arrays(
                multi_trial_trialframe, neural_signal="nonexistent"
            )

    def test_missing_discrete_column_raises(self, multi_trial_trialframe):
        """Raises ValueError if discrete column not in index."""
        with pytest.raises(ValueError, match="not found in index levels"):
            trialframe_to_cebra_arrays(
                multi_trial_trialframe, discrete_behavior_column="nonexistent"
            )


class TestCebraEmbeddingToDataframe:
    """Tests for cebra_embedding_to_dataframe."""

    def test_basic_roundtrip(self, multi_trial_trialframe):
        """Embedding converts back to DataFrame with correct index."""
        result = trialframe_to_cebra_arrays(
            multi_trial_trialframe, pad_between_trials=5
        )
        n_total = result["neural"].shape[0]
        output_dim = 4
        fake_embedding = np.random.randn(n_total, output_dim).astype(np.float32)

        df = cebra_embedding_to_dataframe(
            fake_embedding,
            result["data_mask"],
            result["index_map"],
            col_prefix="cebra",
        )

        assert df.shape == (60, output_dim)
        assert list(df.columns) == ["cebra_0", "cebra_1", "cebra_2", "cebra_3"]
        assert df.index.names == multi_trial_trialframe.index.names

    def test_index_matches_original(self, multi_trial_trialframe):
        """Output DataFrame index matches original trialframe index."""
        result = trialframe_to_cebra_arrays(
            multi_trial_trialframe, pad_between_trials=5
        )
        n_total = result["neural"].shape[0]
        fake_embedding = np.random.randn(n_total, 3).astype(np.float32)

        df = cebra_embedding_to_dataframe(
            fake_embedding,
            result["data_mask"],
            result["index_map"],
        )

        # Check that index entries match original
        pd.testing.assert_index_equal(df.index, multi_trial_trialframe.index)

    def test_mismatched_length_raises(self):
        """Raises ValueError if embedding length doesn't match index map."""
        embedding = np.random.randn(10, 3)
        mask = np.ones(10, dtype=bool)
        index = pd.MultiIndex.from_arrays(
            [[1] * 5, pd.to_timedelta(range(5), unit="s")],
            names=["trial_id", "time"],
        )

        with pytest.raises(ValueError, match="do not match"):
            cebra_embedding_to_dataframe(embedding, mask, index)


class TestSaveCebraParquet:
    """Tests for save_cebra_parquet."""

    def test_saves_correctly(self, tmp_path, multi_trial_trialframe):
        """Parquet file is saved with correct naming convention."""
        result = trialframe_to_cebra_arrays(
            multi_trial_trialframe, pad_between_trials=0
        )
        fake_embedding = np.random.randn(60, 4).astype(np.float32)
        df = cebra_embedding_to_dataframe(
            fake_embedding,
            result["data_mask"],
            result["index_map"],
        )

        dataset = "Test_2025-01-01"
        out_path = save_cebra_parquet(df, tmp_path, dataset)

        assert out_path.exists()
        assert out_path.name == f"{dataset}_neural-cebra-embedding.parquet"

        # Verify it can be read back
        loaded = pd.read_parquet(out_path)
        assert loaded.shape == df.shape

    def test_custom_suffix(self, tmp_path, multi_trial_trialframe):
        """Custom suffix is used in filename."""
        result = trialframe_to_cebra_arrays(
            multi_trial_trialframe, pad_between_trials=0
        )
        fake_embedding = np.random.randn(60, 4).astype(np.float32)
        df = cebra_embedding_to_dataframe(
            fake_embedding, result["data_mask"], result["index_map"]
        )

        out_path = save_cebra_parquet(
            df, tmp_path, "Test_2025-01-01", suffix="neural-cebra-time"
        )
        assert out_path.name == "Test_2025-01-01_neural-cebra-time.parquet"

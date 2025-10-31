import types
import numpy as np
import pytest

from src.subspace_tools import (
    calc_projected_variance,
    calculate_fraction_variance,
    frac_var_explained_by_subspace,
    find_potent_null_space,
    subspace_overlap_index,
)


def test_calculate_fraction_variance_simple():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4)) * np.array([1.0, 2.0, 0.5, 3.0])
    frac = calculate_fraction_variance(X)
    var = np.var(X, axis=0)
    expected = var / var.sum()
    assert np.isclose(frac.sum(), 1.0, atol=1e-7)
    assert np.allclose(frac, expected, rtol=1e-6, atol=1e-8)


def test_calc_projected_variance_matches_pca():
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 5))
    pca = PCA(n_components=5, svd_solver="full").fit(X)

    for k in [1, 2, 4, 5]:
        # PCA components_ has shape (n_components, n_features), rows are basis vectors
        P = pca.components_[:k].T  # columns are basis vectors
        proj_var = calc_projected_variance(X, P)
        expected = float(np.sum(pca.explained_variance_[:k]))
        assert np.isclose(proj_var, expected, rtol=1e-5, atol=1e-7)


def test_frac_var_explained_by_subspace_matches_ratio_to_total_var():
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(2)
    X = rng.standard_normal((400, 6))
    pca = PCA(n_components=6, svd_solver="full").fit(X)

    for k in [1, 3, 6]:
        P = pca.components_[:k].T
        # Our implementation uses sample covariance (ddof=1) in numerator and population variance (ddof=0) in denominator
        num = float(np.sum(pca.explained_variance_[:k]))
        denom = float(np.var(X, axis=0).sum())
        expected_ratio = num / denom
        got = float(frac_var_explained_by_subspace(X, P))
        assert np.isclose(got, expected_ratio, rtol=1e-5, atol=1e-7)


def test_find_potent_null_space_shapes_and_properties():
    rng = np.random.default_rng(3)
    n_samples = 500
    n_features = 7
    n_targets = 3
    X = rng.standard_normal((n_samples, n_features))
    W = rng.standard_normal((n_targets, n_features))
    Y = X @ W.T  # exact linear map, no noise

    potent, null = find_potent_null_space(X, Y)

    # Dimensions
    rank = np.linalg.matrix_rank(W)
    assert potent.shape[0] == n_features
    assert null.shape[0] == n_features
    assert potent.shape[1] == rank
    assert null.shape[1] == n_features - rank

    # Orthogonality between potent and null spaces
    ortho = potent.T @ null
    assert np.allclose(ortho, 0, atol=1e-8)

    # Null space vectors lie in the null of W (rows of W span potent space)
    assert np.allclose(W @ null, 0, atol=1e-8)


def test_subspace_overlap_index_uses_mocked_dekodec(monkeypatch):
    rng = np.random.default_rng(4)
    X = rng.standard_normal((250, 5))
    Y = rng.standard_normal((250, 5))

    # Create a simple fixed basis (first 2 standard basis vectors)
    basis = np.eye(5)[:, :2]

    # Monkeypatch dekodec.get_potent_null used inside subspace_tools
    def fake_get_potent_null(Y_in, num_dims=10):
        assert Y_in.shape[1] == 5
        return basis, np.eye(5)[:, 2:]

    import src.subspace_tools as st

    monkeypatch.setattr(st.dekodec, "get_potent_null", fake_get_potent_null)

    expected = frac_var_explained_by_subspace(X, basis)
    got = subspace_overlap_index(X, Y)
    assert np.isclose(got, expected, rtol=1e-6, atol=1e-8)

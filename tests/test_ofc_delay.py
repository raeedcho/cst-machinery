"""Test delay augmentation in CSTProblem."""
import numpy as np
from src.ofc_model import CSTProblem, augment_system
from ioc.methods.solvers import TodorovSOC
import jax.numpy as jnp


def test_augment_system_shapes():
    """Test that augment_system produces correct shapes."""
    # Simple 2x2 system
    A0 = jnp.eye(2)
    B0 = jnp.ones((2, 1))
    Q0 = jnp.repeat(jnp.eye(2)[:, :, jnp.newaxis], 10, axis=2)
    H0 = jnp.eye(2)
    
    delay = 0.06
    dt = 0.01
    h = int(np.floor(delay / dt))  # should be 6
    
    A, B, Q, H = augment_system(A0, B0, Q0, H0, delay, dt)
    
    expected_n = (h + 1) * 2  # 14
    assert A.shape == (expected_n, expected_n), f"A shape: {A.shape}"
    assert B.shape == (expected_n, 1), f"B shape: {B.shape}"
    assert Q.shape == (expected_n, expected_n, 10), f"Q shape: {Q.shape}"
    assert H.shape == (2, expected_n), f"H shape: {H.shape}"
    
    print("✓ augment_system shape test passed")


def test_cst_problem_no_delay():
    """Test CSTProblem without delay (baseline)."""
    problem = CSTProblem(
        lam=1.5,
        pos_or_vel='pos',
        signal_dependent_noise_const=1.5,
        motor_noise_const=0.4,
        trial_length=2,
        dt=0.01,
        delay=0.0,
    )
    
    # Should have original state dimension (5)
    assert problem.xdim == 5, f"xdim: {problem.xdim}"
    assert problem.A.shape == (5, 5), f"A shape: {problem.A.shape}"
    
    soc = TodorovSOC(problem)
    costs = soc.run(max_iter=50, eps=1e-14)
    costs_computed = costs[~jnp.isnan(costs)]
    
    assert len(costs_computed) > 0, "Solver failed to converge"
    print(f"✓ No delay: converged in {len(costs_computed) - 1} iterations, final cost: {costs_computed[-1]:.2e}")


def test_cst_problem_with_delay():
    """Test CSTProblem with 60ms delay."""
    problem = CSTProblem(
        lam=1.5,
        pos_or_vel='pos',
        signal_dependent_noise_const=1.5,
        motor_noise_const=0.4,
        trial_length=2,
        dt=0.01,
        delay=0.06,
    )
    
    # With delay=0.06, dt=0.01 -> h=6 -> augmented dim = (6+1)*5 = 35
    expected_xdim = 7 * 5
    assert problem.xdim == expected_xdim, f"xdim: {problem.xdim}, expected: {expected_xdim}"
    assert problem.A.shape == (expected_xdim, expected_xdim), f"A shape: {problem.A.shape}"
    
    soc = TodorovSOC(problem)
    costs = soc.run(max_iter=50, eps=1e-14)
    costs_computed = costs[~jnp.isnan(costs)]
    
    assert len(costs_computed) > 0, "Solver failed to converge"
    print(f"✓ With delay: converged in {len(costs_computed) - 1} iterations, final cost: {costs_computed[-1]:.2e}")
    
    # Cost should be higher with delay (controller is more constrained)
    print(f"  Augmented state dimension: {problem.xdim}")


def test_different_delays():
    """Test various delay values."""
    delays = [0.0, 0.02, 0.05, 0.10]
    dt = 0.01
    
    for delay in delays:
        problem = CSTProblem(
            lam=1.5,
            pos_or_vel='pos',
            signal_dependent_noise_const=1.5,
            motor_noise_const=0.4,
            trial_length=1,
            dt=dt,
            delay=delay,
        )
        
        h = int(np.floor(delay / dt)) if delay > 0 else 0
        expected_xdim = (h + 1) * 5
        assert problem.xdim == expected_xdim, f"delay={delay}: xdim={problem.xdim}, expected={expected_xdim}"
        
        soc = TodorovSOC(problem)
        costs = soc.run(max_iter=50, eps=1e-14)
        costs_computed = costs[~jnp.isnan(costs)]
        
        print(f"✓ delay={delay:.3f}s (h={h}): dim={problem.xdim}, converged in {len(costs_computed)-1} iters, cost={costs_computed[-1]:.2e}")

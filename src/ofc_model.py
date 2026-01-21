import numpy as np
import jax.numpy as jnp
from ioc.examples.problem import ControlTask


def augment_system(A0, B0, Q0, H0, delay, dt):
    """Augment system matrices to model a hard feedback delay.

    Implements the MATLAB AugRobustControl algorithm:
    - Builds an augmented state of size ((h+1) * n) where h = floor(delay/dt)
    - Shifts delayed state blocks down the augmented state vector
    - Applies control and observation to the most recent state block
    - Expands Q across delayed blocks using a time-shifted, averaged scheme

    Parameters
    - A0: (n, n) discrete-time state transition matrix
    - B0: (n, m) discrete-time input matrix
    - Q0: (n, n, T) state cost matrices per time step
    - H0: (p, n) observation matrix
    - delay: delay duration in the same time units as dt (seconds)
    - dt: discretization step (seconds)

    Returns
    - A: ((h+1)*n, (h+1)*n)
    - B: ((h+1)*n, m)
    - Q: (((h+1)*n), ((h+1)*n), T)
    - H: (p, (h+1)*n)
    """
    n = A0.shape[0]
    m = B0.shape[1]
    T = Q0.shape[2]
    p = H0.shape[0]

    h = int(np.floor(delay / dt))
    aug_n = (h + 1) * n

    # Augmented A: top-left A0, with block shift for delayed states
    A = jnp.zeros((aug_n, aug_n))
    A = A.at[0:n, 0:n].set(A0)
    if h > 0:
        A = A.at[n:aug_n, 0:(aug_n - n)].set(jnp.eye(h * n))

    # Augmented B: control affects the most recent state block only
    B = jnp.zeros((aug_n, m))
    B = B.at[0:n, :].set(B0)

    # Augmented H: observe the most recent state block
    H = jnp.zeros((p, aug_n))
    H = H.at[:, (aug_n - n):aug_n].set(H0)

    # Build Qaug by prepending h copies of the first Q slice
    Qaug = jnp.zeros((n, n, T + h))
    if h > 0:
        Qaug = Qaug.at[:, :, 0:h].set(jnp.repeat(Q0[:, :, 0][:, :, jnp.newaxis], h, axis=2))
    Qaug = Qaug.at[:, :, h:h + T].set(Q0)

    # Fill block-diagonal Q across delayed blocks, time-shifted and averaged
    Q = jnp.zeros((aug_n, aug_n, T))
    if h == 0:
        Q = Q.at[:, :, :].set(Q0)
    else:
        for time in range(T):
            for ii in range(h + 1):
                blk = slice(ii * n, (ii + 1) * n)
                Q = Q.at[blk, blk, time].set(Qaug[:, :, time + h - ii] / (h + 1))

    return A, B, Q, H


class CSTProblem(ControlTask):
    def __init__(
        self,
        lam=5,
        signal_dependent_noise_const=1.5,
        motor_noise_const=0.4,
        T = 800, # total time steps
        dt=0.01, # time step (seconds)
        delay=0.06,
        control_cost_exp=2,
        pos_cost_exp=-15,
        vel_cost_exp=10,
    ):
        # setup problem
        m = 1  # mass(kg)
        tau = 0.06  # time constant(sec)

        # compute system dynamics and cost matrices
        base_state_dim = 5
        # continuous dynamics
        A = np.zeros((base_state_dim, base_state_dim))
        A[0, 1] = 1
        A[1, 0] = lam ** 2
        A[1, 2] = lam ** 2
        A[1, 3] = lam
        A[2, 3] = 1
        A[3, 4] = 1/m
        A[4, 4] = -1/tau
        # discretize dynamics
        A = np.eye(base_state_dim) + A * dt
        A = jnp.array(A)

        # continuous input matrix
        B = np.zeros((base_state_dim, 1))
        B[4, 0] = 1/tau
        # discretize input matrix
        B = B * dt
        B = jnp.array(B)

        # action dependent noise transform in dynamics
        C = signal_dependent_noise_const
        
        # feedback matrix (all states observed)
        H = np.eye(base_state_dim)
        H = jnp.array(H)

        # state dependent noise transform in feedback
        D = 0.
        
        # control cost
        R = 10 ** control_cost_exp

        # state costs (same for all time steps)
        q = jnp.diag(jnp.array([10**pos_cost_exp, 10**vel_cost_exp, 0., 0., 0.]))
        Q = jnp.repeat(q[:, :, jnp.newaxis], T, axis=2)

        # Optional delay augmentation
        if delay and delay > 0:
            A, B, Q, H = augment_system(A, B, Q, H, delay, dt)
        
        # Set state_dim based on (possibly augmented) A
        state_dim = A.shape[0]
        obs_dim = H.shape[0]

        # initial state and uncertainty
        x0 = np.zeros((state_dim, 1))
        S0 = jnp.eye(state_dim) * 0.

        # Initial belief about the state
        B0 = jnp.array(x0)
        V0 = jnp.eye(state_dim) * 1e-1

        # Noise covariances (cholesky factors)
        C0 = motor_noise_const * B @ B.T
        D0 = jnp.eye(obs_dim) * 1e-3
        E0 = jnp.eye(state_dim) * 0.

        super().__init__(A, B, C, C0, H, D, D0, E0, Q, R, x0, S0, B0=B0, V0=V0)

def get_cst_pos_control(**kwargs):
    return CSTProblem(pos_cost_exp=5, vel_cost_exp=-15, **kwargs)

def get_cst_vel_control(**kwargs):
    return CSTProblem(pos_cost_exp=-15, vel_cost_exp=10, **kwargs)
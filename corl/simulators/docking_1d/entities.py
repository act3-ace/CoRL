"""
This module defines backend entities that maintain the state of a platform and defines the dynamics model used to
calculate state transitions between time steps, given applied controls. In this example, the dynamics model is
a 1D Double Integrator.
"""

import typing

import numpy as np
import scipy.integrate
import scipy.spatial
from pydantic import BaseModel, ConfigDict


class Deputy1dValidator(BaseModel):
    """
    This module validates that Deputy1D configs contain a name and initial state values.
    """

    name: str
    x: typing.Any = None
    xdot: typing.Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Deputy1D:  # noqa: PLW1641
    """
    1D point mass spacecraft with a +/- thruster and Double Integrator dynamics

    States
        x
        x_dot

    Controls
        thrust
            range = [-1, 1] Newtons

    Parameters
    ----------
    m: float
        Mass of spacecraft in kilograms, by default 12
    integration_method: str
        Numerical integration method passed to dynamics model.
    kwargs:
        Additional keyword arguments passed to Deputy1dValidator
    """

    def __init__(self, m=12, integration_method="RK45", **kwargs):
        self.config = self._get_config_validator()(**kwargs)
        self.name = self.config.name
        self.dynamics = Docking1dDynamics(m=m, integration_method=integration_method)

        self.control_default = np.zeros((1,))
        self.control_min = -1.0
        self.control_max = 1.0

        self._state = self._build_state()
        self.state_dot = np.zeros_like(self._state)

    def __eq__(self, other):
        if isinstance(other, Deputy1D):
            eq = (self.velocity == other.velocity).all()
            return eq and (self.position == other.position).all()
        return False

    @classmethod
    def _get_config_validator(cls):
        return Deputy1dValidator

    def step(self, step_size, action=None):
        """
        Executes a state transition simulation step for the entity

        Parameters
        ----------
        step_size : float
            duration of simulation step in seconds
        action : np.ndarray, optional
            Control action taken by entity, by default None resulting in a control of control_default.

        Raises
        ------
        KeyError
            Raised when action dict key not found in control map
        ValueError
            Raised when action is not one of the required types
        """

        if action is None:
            control = self.control_default.copy()
        elif isinstance(action, np.ndarray):
            control = action.copy()
        else:
            raise ValueError("action must be type np.ndarray")

        # enforce control bounds
        control = np.clip(control, self.control_min, self.control_max)

        # compute new state if dynamics were applied
        self.state, self.state_dot = self.dynamics.step(step_size, self.state, control)

    def _build_state(self):
        # builds initial state?
        return np.array([self.config.x.m, self.config.xdot.m], dtype=np.float32)

    @property
    def state(self) -> np.ndarray:
        """
        Returns copy of entity's state vector

        Returns
        -------
        np.ndarray
            copy of state vector
        """
        return self._state.copy()

    @state.setter
    def state(self, value: np.ndarray):
        self._state = value.copy()

    @property
    def position(self):
        """
        get 1d position vector
        """
        position = np.zeros(1)
        position[0] = self._state[0].copy()
        return position

    @property
    def velocity(self):
        """
        get 1d velocity vector
        """
        velocity = np.zeros(1)
        velocity[0] = self._state[1].copy()
        return velocity


class Docking1dDynamics:
    """
    State transition implementation for generic Linear Ordinary Differential Equation dynamics models of the form
    dx/dt = Ax+Bu. Computes next state through numerical integration of differential equation.

    Parameters
    ----------
    state_min : float or np.ndarray
        Minimum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
    state_max : float or np.ndarray
        Maximum allowable value for the next state. State values that exceed this are clipped.
        When a float, represents single limit applied to entire state vector.
        When an ndarray, each element represents the limit to the corresponding state vector element.
    angle_wrap_centers: np.ndarray
        Enables circular wrapping of angles. Defines the center of circular wrap such that angles are within [center+pi, center-pi].
        When None, no angle wrapping applied.
        When ndarray, each element defines the angle wrap center of the corresponding state element.
        Wrapping not applied when element is NaN.
    integration_method : string
        Numerical integration method used by dynamics solver. One of ['RK45', 'Euler'].
        'RK45' is slow but very accurate.
        'Euler' is fast but very inaccurate.
    """

    def __init__(
        self,
        state_min: float | np.ndarray = -np.inf,
        state_max: float | np.ndarray = np.inf,
        angle_wrap_centers: np.ndarray | None = None,
        m: float = 12.0,
        integration_method: str = "RK45",
    ):
        self.state_min = state_min
        self.state_max = state_max
        self.angle_wrap_centers = angle_wrap_centers
        self.m = m
        self.A, self.B = self._gen_dynamics_matrices()
        self.integration_method = integration_method

    def step(self, step_size: float, state: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the dynamics state transition from the current state and control input

        Parameters
        ----------
        step_size : float
            Duration of the simation step in seconds.
        state : np.ndarray
            Current state of the system at the beginning of the simulation step.
        control : np.ndarray
            Control vector of the dynamics model.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            tuple of the systems's next state and the state's instantaneous time derivative at the end of the step
        """

        if self.integration_method == "RK45":
            sol = scipy.integrate.solve_ivp(self.compute_state_dot, (0, step_size), state, args=(control,))

            next_state = sol.y[:, -1]  # save last timestep of integration solution
            state_dot = self.compute_state_dot(step_size, next_state, control)
        else:
            raise ValueError(f"invalid integration method '{self.integration_method}'")

        next_state = np.clip(next_state, self.state_min, self.state_max)
        next_state = self._wrap_angles(next_state)
        return next_state, state_dot

    def _wrap_angles(self, state):
        wrapped_state = state.copy()
        if self.angle_wrap_centers is not None:
            wrap_idxs = np.logical_not(np.isnan(self.angle_wrap_centers))

            wrapped_state[wrap_idxs] = ((wrapped_state[wrap_idxs] + np.pi) % (2 * np.pi)) - np.pi + self.angle_wrap_centers[wrap_idxs]

        return wrapped_state

    def compute_state_dot(self, t: float, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Computes the instataneous time derivative of the state vector

        Parameters
        ----------
        t : float
            Time in seconds since the beginning of the simulation step.
            Note, this is NOT the total simulation time but the time within the individual step.
        state : np.ndarray
            Current state vector at time t.
        control : np.ndarray
            Control vector.

        Returns
        -------
        np.ndarray
            Instantaneous time derivative of the state vector.
        """
        state_dot = np.matmul(self.A, state) + np.matmul(self.B, control)

        # clip state_dot by state limits
        lower_bounded_states = state <= self.state_min
        upper_bounded_state = state >= self.state_max

        state_dot[lower_bounded_states] = np.clip(state_dot[lower_bounded_states], 0, np.inf)
        state_dot[upper_bounded_state] = np.clip(state_dot[upper_bounded_state], -np.inf, 0)

        return state_dot

    def _gen_dynamics_matrices(self):
        """
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The A and B matrix defining 1D Double Integrator dynamics
        """

        m = self.m

        A = np.array(
            [[0, 1], [0, 0]],
            dtype=np.float32,
        )

        B = np.array(
            [[0], [1 / m]],
            dtype=np.float32,
        )

        return A, B

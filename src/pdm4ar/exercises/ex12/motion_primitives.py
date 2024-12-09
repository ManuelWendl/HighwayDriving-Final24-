from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import List, Callable, Set

import numpy as np

from dg_commons import logger, Timestamp
from dg_commons.planning.trajectory import Trajectory
from dg_commons.planning.trajectory_generator_abc import TrajGenerator
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.time import time_function


@dataclass
class MPGParam:
    dt: Decimal
    n_steps: int
    velocity: int
    steering: int

    def __post_init__(self):
        assert isinstance(self.dt, Decimal)

    @classmethod
    def from_vehicle_parameters(cls, dt: Decimal, n_steps: int, n_vel: int, n_steer: int) -> "MPGParam":
        """
        :param dt:
        :param n_steps:
        :param n_vel:
        :param n_steer:
        :param vp:
        :return:
        """
        return MPGParam(dt=dt, n_steps=n_steps, velocity=n_vel, steering=n_steer)


class MotionPrimitivesGenerator(TrajGenerator):
    """Generator of motion primitives sampling the state space"""

    def __init__(
        self,
        param: MPGParam,
        vehicle_dynamics: Callable[[VehicleState, VehicleCommands, Timestamp], VehicleState],
        vehicle_param: VehicleParameters,
    ):
        super().__init__(vehicle_dynamics=vehicle_dynamics, vehicle_param=vehicle_param)
        self.param = param

    @time_function
    def generate(
        self, x0: VehicleState, max_steering_angle: float, verbose: bool = False
    ) -> tuple[List[Trajectory], List[VehicleCommands]]:
        """
        :param x0: optionally if one wants to generate motion primitives only from a specific state
        :return:
        """
        v_samples, steer_samples = self.generate_samples(x0=x0, max_steering_angle=max_steering_angle)
        motion_primitives: List[Trajectory] = []
        commands: List[VehicleCommands] = []

        v_start = x0.vx
        sa_start = x0.delta

        n = len(v_samples) * len(steer_samples)
        if verbose:
            logger.debug(f"Attempting to generate {n} motion primitives")
        for v_end, sa_end in product(v_samples, steer_samples):
            is_valid, input_a, input_sa_rate = self.check_input_constraints(v_start, v_end, sa_start, sa_end)
            if not is_valid:
                continue
            init_state = VehicleState(x=0, y=0, psi=0, vx=v_start, delta=sa_start) if x0 is None else x0
            timestamps = [
                Decimal(0),
            ]
            states = [
                init_state,
            ]
            next_state = init_state
            cmds = VehicleCommands(acc=input_a, ddelta=input_sa_rate)
            for n_step in range(1, self.param.n_steps + 1):
                next_state = self.vehicle_dynamics(next_state, cmds, float(self.param.dt / self.param.n_steps))
                timestamps.append(n_step * self.param.dt / self.param.n_steps)
                states.append(next_state)
            motion_primitives.append(Trajectory(timestamps=timestamps, values=states))
            commands.append(cmds)
        if verbose:
            logger.info(f"{type(self).__name__}:Found {len(motion_primitives)} feasible motion primitives")

        return motion_primitives, commands

    def generate_samples(self, x0, max_steering_angle: float) -> tuple[List, List]:
        if self.param.velocity == 1:
            v_samples = [x0.vx]
        else:
            v_samples = np.linspace(
                max(x0.vx + self.vehicle_param.acc_limits[0] * 2 / 3 * float(self.param.dt), 5),
                min(
                    x0.vx + self.vehicle_param.acc_limits[1] * 2 / 3 * float(self.param.dt),
                    self.vehicle_param.vx_limits[1],
                ),
                self.param.velocity,
            )
        if self.param.steering == 1:
            steer_samples = [x0.delta]
        else:
            steer_samples = np.linspace(
                max(
                    x0.delta - self.vehicle_param.ddelta_max * float(self.param.dt),
                    x0.delta - max_steering_angle,
                ),
                min(
                    x0.delta + self.vehicle_param.ddelta_max * float(self.param.dt),
                    x0.delta + max_steering_angle,
                ),
                self.param.steering,
            )
        return v_samples, steer_samples

    def check_input_constraints(self, v_start, v_end, sa_start, sa_end) -> tuple[bool, float, float]:
        """
        :param v_start: initial velocity
        :param v_end: ending velocity
        :param sa_start: initial steering angle
        :param sa_end: ending steering angle
        :return: [is_valid,acc,steer_rate]
        """
        horizon = float(self.param.dt)
        acc = (v_end - v_start) / horizon
        sa_rate = (sa_end - sa_start) / horizon
        if not (-self.vehicle_param.ddelta_max <= sa_rate <= self.vehicle_param.ddelta_max) or (
            not self.vehicle_param.acc_limits[0] <= acc <= self.vehicle_param.acc_limits[1]
        ):
            return False, 0, 0
        else:
            return True, acc, sa_rate

import array
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import List, Callable, Set

from matplotlib.pylab import f
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
        self, x0: VehicleState, max_steering_angle_change: float, depth: int, goal_velocity: float
    ) -> tuple[List[VehicleState], List[VehicleCommands]]:
        """
        :param x0: optionally if one wants to generate motion primitives only from a specific state
        :return:
        """
        v_samples, steer_samples = self.generate_samples(
            x0=x0, max_steering_angle_change=max_steering_angle_change, depth=depth, goal_velocity=goal_velocity
        )
        motion_primitives: List[VehicleState] = []
        commands: List[VehicleCommands] = []

        v_start = x0.vx
        sa_start = x0.delta

        n = len(v_samples) * len(steer_samples)
        for v_end, sa_end in product(v_samples, steer_samples):
            is_valid, input_a, input_sa_rate = self.check_input_constraints(v_start, v_end, sa_start, sa_end)
            if not is_valid:
                continue
            init_state = VehicleState(x=0, y=0, psi=0, vx=v_start, delta=sa_start) if x0 is None else x0
            cmds = VehicleCommands(acc=input_a, ddelta=input_sa_rate)
            next_state = self.integrate_dynamics(init_state, cmds)
            motion_primitives.append(next_state)
            commands.append(cmds)

        return motion_primitives, commands

    def integrate_dynamics(self, x0: VehicleState, cmds: VehicleCommands) -> VehicleState:
        """
        :param x0: initial state
        :param cmds: vehicle commands
        :return: next state
        """
        next_state = x0
        for _ in range(self.param.n_steps):
            next_state = self.vehicle_dynamics(next_state, cmds, float(self.param.dt / self.param.n_steps))
        return next_state

    def generate_samples(self, x0, max_steering_angle_change: float, depth, goal_velocity: float) -> tuple[List, List]:
        end_velocity = min(
            max(
                goal_velocity,
                x0.vx + self.vehicle_param.acc_limits[0] * 2 / 3 * float(self.param.dt),
                self.vehicle_param.vx_limits[0],
            ),
            self.vehicle_param.vx_limits[1],
            x0.vx + self.vehicle_param.acc_limits[1] * 2 / 3 * float(self.param.dt),
        )

        if depth == 0:
            if self.param.velocity == 1:
                v_samples = [end_velocity]
            else:
                n_symmetric = self.param.velocity // 2
                v_lower = np.array(
                    [
                        end_velocity + i * self.vehicle_param.acc_limits[0] * 1 / 3 * float(self.param.dt)
                        for i in range(1, n_symmetric + 1)
                    ]
                )
                v_upper = np.array(
                    [
                        end_velocity + i * self.vehicle_param.acc_limits[1] * 1 / 3 * float(self.param.dt)
                        for i in range(1, n_symmetric + 1)
                    ]
                )
                v_samples = np.concatenate((v_lower, np.array([end_velocity]), v_upper))
                v_samples = np.clip(v_samples, self.vehicle_param.vx_limits[0], self.vehicle_param.vx_limits[1])
        else:
            v_samples = [end_velocity]

        if self.param.steering == 1:
            steer_samples = [x0.delta]
        else:
            steer_samples = np.linspace(
                max(
                    x0.delta - self.vehicle_param.ddelta_max * float(self.param.dt),
                    x0.delta - max_steering_angle_change * float(self.param.dt),
                ),
                min(
                    x0.delta + self.vehicle_param.ddelta_max * float(self.param.dt),
                    x0.delta + max_steering_angle_change * float(self.param.dt),
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

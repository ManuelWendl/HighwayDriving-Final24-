import numpy as np
import shapely

from dg_commons.sim.models.model_structures import TModelGeometry
from dg_commons.dynamics.bicycle_dynamic import VehicleState
from dg_commons.sim import SimObservations, PlayerObservations

from typing import Optional


def get_vehicle_shapely(
    sg: Optional[TModelGeometry],
    state: VehicleState,
    length_dilation_for_collision: float = 0.0,
    width_dilation_for_collision: float = 0.0,
    own_vehicle: bool = True,
    sim_obs: PlayerObservations = None,
):
    if own_vehicle:
        lf = sg.lf
        w_half = sg.w_half
        lr = sg.lr
    else:
        # Get the other vehicle's dimensions
        length = max(
            np.abs(sim_obs.occupancy.bounds[2] - sim_obs.occupancy.bounds[0]),
            np.abs(sim_obs.occupancy.bounds[3] - sim_obs.occupancy.bounds[1]),
        )
        w_half = (
            min(
                np.abs(sim_obs.occupancy.bounds[2] - sim_obs.occupancy.bounds[0]),
                np.abs(sim_obs.occupancy.bounds[3] - sim_obs.occupancy.bounds[1]),
            )
            / 2
        )
        lf = lr = length / 2

    cog = np.array([state.x, state.y])
    R = np.array([[np.cos(state.psi), -np.sin(state.psi)], [np.sin(state.psi), np.cos(state.psi)]])
    front_left = (
        cog + R @ np.array([lf * (1 + length_dilation_for_collision), w_half * (1 + width_dilation_for_collision)]).T
    )
    front_right = (
        cog + R @ np.array([lf * (1 + length_dilation_for_collision), -w_half * (1 + width_dilation_for_collision)]).T
    )
    back_left = (
        cog + R @ np.array([-lr * (1 + length_dilation_for_collision), w_half * (1 + width_dilation_for_collision)]).T
    )
    back_right = (
        cog + R @ np.array([-lr * (1 + length_dilation_for_collision), -w_half * (1 + width_dilation_for_collision)]).T
    )

    vehicle_shapely = shapely.Polygon((tuple(front_left), tuple(front_right), tuple(back_right), tuple(back_left)))

    return vehicle_shapely

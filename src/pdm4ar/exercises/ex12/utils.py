import numpy as np
import shapely

from dg_commons.sim.models.model_structures import TModelGeometry
from dg_commons.dynamics.bicycle_dynamic import VehicleState
from typing import Optional


def get_vehicle_shapely(sg: Optional[TModelGeometry], state: VehicleState, length_dilation_for_collision: float = 0.0):
    cog = np.array([state.x, state.y])
    R = np.array([[np.cos(state.psi), -np.sin(state.psi)], [np.sin(state.psi), np.cos(state.psi)]])
    front_left = cog + R @ np.array([sg.lf * length_dilation_for_collision, sg.w_half]).T
    front_right = cog + R @ np.array([sg.lf * length_dilation_for_collision, -sg.w_half]).T
    back_left = cog + R @ np.array([sg.lr * length_dilation_for_collision, sg.w_half]).T
    back_right = cog + R @ np.array([sg.lr * length_dilation_for_collision, -sg.w_half]).T

    vehicle_shapely = shapely.Polygon((tuple(front_left), tuple(front_right), tuple(back_left), tuple(back_right)))

    return vehicle_shapely

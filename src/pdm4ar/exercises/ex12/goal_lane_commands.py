import random
from dataclasses import dataclass
from typing import Sequence

from dg_commons import DgSampledSequence, PlayerName, X, U
from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal, RefLaneGoal
from dg_commons.sim import SimObservations, InitSimObservations, PlayerObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.maps import DgLanelet, DgLanePose
from dg_commons.sim import extract_pose_from_state


from dg_commons.dynamics.bicycle_dynamic import BicycleDynamics, VehicleState
from decimal import Decimal
from matplotlib import pyplot as plt
import numpy as np


def goal_lane_commands(self, sim_obs: SimObservations) -> VehicleCommands:
    """This method is called by the simulator as soon as we're on the goal lane.

    Notes:
    - L170-172 ex12/perf_metrics.py: Time to collision is ignored after the goal lane
      has been reached.
    - Lane heading has to be exact, discomfort must be minimized, do not collide.
    -> We correct the lane heading and decelerate if any collision is imminent at the
        cost of increased discomfort.
    """
    # Check lane heading
    # Taken from L188-209 ex12/perf_metrics.py
    ego_state: VehicleState = sim_obs.players[self.name].state  # type: ignore
    lanelet = self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id)
    dg_lanelet = DgLanelet.from_commonroad_lanelet(lanelet)
    pose = extract_pose_from_state(ego_state)
    dg_pose: DgLanePose = dg_lanelet.lane_pose_from_SE2_generic(pose)
    heading = dg_pose.relative_heading
    if ego_state.vx < 0:
        # L202 ex12/perf_metrics.py
        print("WARNING: Negative velocity gives a high heading penalty")

    ddelta = 0
    if np.abs(heading) > 1e-4:
        # Correct heading
        # next state
        # psi = psi + dt * dx * math.tan(x0.delta) / self.vg.wheelbase
        # heading = - dt * dx * math.tan(x0.delta) / self.vg.wheelbase
        ddelta = np.arctan(-heading * self.dyn.vg.wheelbase / (0.1 * ego_state.vx))
    # Decelerate if collision is imminent
    acc = 0
    time_to_collision = np.inf
    for player_name, player_obs in sim_obs.players.items():
        if player_name == self.name:
            continue
        other_state: VehicleState = player_obs.state  # type: ignore
        other_lanelet_ids = self.lanelet_network.find_lanelet_by_position([np.array([other_state.x, other_state.y])])

        # Check if the other player is on the same lanelet
        if self.goal_lanelet_id in other_lanelet_ids:
            other_shapely = self.get_vehicle_shapely(other_state)
            vehicle_shapely = self.get_vehicle_shapely(ego_state)
            # compute distance between shapelies
            distance = vehicle_shapely.distance(other_shapely)
            time_to_collision = min(time_to_collision, distance / ego_state.vx)

    if time_to_collision < 1:
        # Decelerate, TODO: more smoothly
        acc = self.sp.acc_limits[0]

    return VehicleCommands(acc=acc, ddelta=ddelta)

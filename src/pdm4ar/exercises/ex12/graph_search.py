from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq

from numpy import shape

from pdm4ar.exercises_def.ex09 import goal

from .weighted_graph import WeightedGraph, tree_node
from typing import Optional, List, Tuple
from .utils import get_vehicle_shapely
from dg_commons.dynamics.bicycle_dynamic import VehicleState
from dg_commons.sim.models.model_structures import TModelGeometry
import shapely
from dg_commons.sim import SimObservations
from dg_commons.maps.lanes import DgLanelet
from dg_commons import SE2Transform

Path = Optional[List[tree_node]]


@dataclass(frozen=False)
class GraphParams:
    collision_rejection_threshold = 0.05  # in percent
    collision_cost_weight = 1e9  # weight for collision cost
    collision_buffer = 0.2  # buffer for collision
    use_heuristic = True  # whether to use heuristic or not


@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph
    sg: Optional[TModelGeometry]
    # params: GraphParams

    @abstractmethod
    def path(self, start: tree_node, depth_dicts: list[dict]) -> Path:
        # Abstract function. Nothing to do here.
        pass


@dataclass
class Astar(InformedGraphSearch):
    params = GraphParams()
    goal_lanelet: Optional[DgLanelet] = None

    def check_other_vehicle_collision(
        self,
        current_state: VehicleState,
        other_vehicle_depth_dicts: list[dict],
        current_timestep: int,
        sim_obs: SimObservations,
    ):
        collision_probability = 0.0
        current_vehicle_shapely = sim_obs.players["Ego"].occupancy
        rotated_current_vehicle_shapely = shapely.affinity.rotate(
            current_vehicle_shapely, current_state.psi - sim_obs.players["Ego"].state.psi, origin="centroid"
        )
        translated_current_vehicle_shapely = shapely.affinity.translate(
            rotated_current_vehicle_shapely,
            xoff=current_state.x - sim_obs.players["Ego"].state.x,
            yoff=current_state.y - sim_obs.players["Ego"].state.y,
        )
        for other_vehicle_name, other_vehicle_depth_dict in other_vehicle_depth_dicts.items():
            vehicle_probability = 0.0
            if current_timestep not in other_vehicle_depth_dict:
                # print(f"Current timestep {current_timestep} not in other vehicle depth dict")
                break
            for other_vehicle_node in other_vehicle_depth_dict[current_timestep]:
                x_offset = other_vehicle_node.state.x - sim_obs.players[other_vehicle_name].state.x
                y_offset = other_vehicle_node.state.y - sim_obs.players[other_vehicle_name].state.y
                translated_other_vehicle_shapely = shapely.affinity.translate(
                    sim_obs.players[other_vehicle_name].occupancy, xoff=x_offset, yoff=y_offset
                )
                if shapely.intersects(
                    translated_current_vehicle_shapely,
                    translated_other_vehicle_shapely.buffer(self.params.collision_buffer),
                ):
                    vehicle_probability += other_vehicle_node.data

            collision_probability = max(collision_probability, vehicle_probability)
        return collision_probability

    def heuristic(self, u: tree_node, sim_obs: SimObservations) -> float:
        # Increment this counter every time the heuristic is called, to judge the performance
        # of the algorithm
        # TODO: Define heuristic as distance from goal lane
        # Caclulate the distance from the goal lane

        # In case no goal lanelet is provided, return 0
        if self.goal_lanelet is None or self.params.use_heuristic is False:
            return 0

        # Get the lane pose of the current state
        state_se2transform = SE2Transform(
            [sim_obs.players["Ego"].state.x, sim_obs.players["Ego"].state.y], sim_obs.players["Ego"].state.psi
        )
        lane_pose = self.goal_lanelet.lane_pose_from_SE2Transform(state_se2transform)

        return 5 * lane_pose.distance_from_center

    def path(self, start: tree_node, depth_dicts: list[dict], sim_obs: SimObservations, safe_depth=None) -> Path:
        # todo
        Q = [(float(0), start)]
        P = {start: None}
        C = {start: 0}
        H = {start: self.heuristic(start, sim_obs)}

        while Q:
            _, s = heapq.heappop(Q)

            if s.data == 1 or (safe_depth is not None and s.depth >= safe_depth):
                path = []
                current = s
                while current is not None:
                    path.insert(0, current)
                    current = P[current]
                return path

            for snext in s.successors:
                we = self.graph.get_weight(s, snext)
                if we:
                    wn = C[s] + we
                else:
                    wn = C[s]

                # Check for collision and don't push the node if it collides
                cp = self.check_other_vehicle_collision(snext.state, depth_dicts, snext.depth, sim_obs)
                if cp < self.params.collision_rejection_threshold:
                    if snext not in H:
                        H.update({snext: self.heuristic(snext, sim_obs)})

                    if snext not in C or wn < C[snext]:
                        P.update({snext: s})
                        C.update({snext: wn + self.params.collision_cost_weight * cp})
                        heapq.heappush(Q, (C[snext] + H[snext], snext))

        return []

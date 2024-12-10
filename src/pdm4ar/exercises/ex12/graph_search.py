from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq

from .weighted_graph import WeightedGraph, tree_node
from typing import Optional, List, Tuple
from .utils import get_vehicle_shapely
from dg_commons.dynamics.bicycle_dynamic import VehicleState
from dg_commons.sim.models.model_structures import TModelGeometry
import shapely

Path = Optional[List[tree_node]]


@dataclass(frozen=True)
class GraphParams:
    length_dilation_for_collision = 0.05  # in percent
    collision_rejection_threshold = 0.3


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

    def check_other_vehicle_collision(
        self, current_state: VehicleState, other_vehicle_depth_dicts: list[dict], current_timestep: int
    ):
        collision_probability = 0.0
        current_vehicle_shapely = get_vehicle_shapely(self.sg, current_state)
        for other_vehicle_depth_dict in other_vehicle_depth_dicts:
            for other_vehicle_node in other_vehicle_depth_dict[current_timestep]:
                other_vehicle_shapely = get_vehicle_shapely(
                    self.sg,
                    other_vehicle_node.state,
                )
                if shapely.intersects(current_vehicle_shapely, other_vehicle_shapely):
                    collision_probability += other_vehicle_node.data
        return collision_probability

    def heuristic(self, u: tree_node) -> float:
        # Increment this counter every time the heuristic is called, to judge the performance
        # of the algorithm
        # TODO: Define heuristic as distance from goal lane
        return 0

    def path(self, start: tree_node, depth_dicts: list[dict]) -> Path:
        # todo
        Q = [(float(0), start)]
        P = {start: None}
        C = {start: 0}
        H = {start: self.heuristic(start)}

        while Q:
            _, s = heapq.heappop(Q)

            if s.data == 1:
                path = []
                current = s
                while current is not None:
                    path.insert(0, current)
                    current = P[current]
                return path

            if s not in self.graph.adj_list:
                continue

            for snext in self.graph.adj_list[s]:
                we = self.graph.get_weight(s, snext)
                if we:
                    wn = C[s] + we
                else:
                    wn = C[s]

                # Check for collision and don't push the node if it collides
                if (
                    self.check_other_vehicle_collision(snext.state, depth_dicts, snext.depth)
                    > self.params.collision_rejection_threshold
                ):
                    if snext not in H:
                        H.update({snext: self.heuristic(snext)})

                    if snext not in C or wn < C[snext]:
                        P.update({snext: s})
                        C.update({snext: wn})
                        heapq.heappush(Q, (C[snext] + H[snext], snext))

        return []

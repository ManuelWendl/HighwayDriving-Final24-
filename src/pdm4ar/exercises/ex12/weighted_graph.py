from typing import Optional, TypeVar, Set, Mapping, Tuple, List

from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import List, Callable, Set, Optional
from dg_commons.eval.comfort import get_acc_rms

from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from matplotlib.pylab import f
import matplotlib.pyplot as plt


class tree_node:
    def __init__(self, state: VehicleState, depth: int, data: float = 0):
        self.successors = []
        self.state = state
        self.depth = depth
        self.data = data  # 0 for normal nodes, 1 for goal nodes (for ego vehicle), probability of being in state (for other vehicles)

    def __lt__(self, other):
        # TODO: Write more meaningful comparison for heapq comparison
        return True


class EdgeNotFound(Exception):
    pass


class CommandNotFound(Exception):
    pass


@dataclass
class WeightedGraph:
    weights: Mapping[tuple[tree_node, tree_node], float]
    cmds: Mapping[tuple[tree_node, tree_node], VehicleCommands]
    start: tree_node

    def get_weight(self, u: tree_node, v: tree_node) -> Optional[float]:
        """
        :param u: The "from" of the edge
        :param v: The "to" of the edge
        :return: The weight associated to the edge, raises an Exception if the edge does not exist
        """
        try:
            return self.weights[(u, v)]
        except KeyError:
            raise EdgeNotFound(f"Cannot find weight for edge: {(u, v)}")

    def get_cmds(self, u: tree_node, v: tree_node) -> Optional[VehicleCommands]:
        """
        :param u: The "from" of the edge
        :param v: The "to" of the edge
        :return: The weight associated to the edge, raises an Exception if the edge does not exist
        """
        try:
            return self.cmds[(u, v)]
        except KeyError:
            raise CommandNotFound(f"Cannot find comand for edge: {(u, v)}")

    def add_edge(self, u: tree_node, v: tree_node, weight: float, cmds: VehicleCommands) -> None:
        """
        :param u: The "from" of the edge
        :param v: The "to" of the edge
        :param weight: The weight of the edge
        """
        u.successors.append(v)
        self.weights[(u, v)] = weight
        self.cmds[(u, v)] = cmds

    def remove_node(self, u: tree_node):
        for v in u.successors:
            if v.successors != []:
                self.remove_node(v)
            del self.weights[(u, v)]
            del self.cmds[(u, v)]

    def draw_graph(self, lanes, trajectory=None, opponent_graphs=None):
        print("Drawing graph")
        plt.figure(figsize=(100, 50))
        for u, v in self.weights.keys():
            if v.data == 1:
                plt.plot([u.state.x, v.state.x], [u.state.y, v.state.y], "ro-", "LineWidth", 0.5)
            else:
                plt.plot([u.state.x, v.state.x], [u.state.y, v.state.y], "ko-", "LineWidth", 0.5)
        for lane in lanes:
            plt.plot(*lane.exterior.xy)
        if trajectory:
            x_states = [node.state.x for node in trajectory]
            y_states = [node.state.y for node in trajectory]
            plt.plot(x_states, y_states, "bo-", "LineWidth", 0.5)
        if opponent_graphs:
            for graph in opponent_graphs:
                for u, v in graph.weights.keys():
                    plt.plot([u.state.x, v.state.x], [u.state.y, v.state.y], "go-", "LineWidth", 0.5)

        plt.axis("equal")
        plt.savefig("graph.png")

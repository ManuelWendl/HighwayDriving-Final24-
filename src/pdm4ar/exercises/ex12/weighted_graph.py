from typing import Optional, TypeVar, Set, Mapping, Tuple, List

from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import List, Callable, Set, Optional
from dg_commons.eval.comfort import get_acc_rms

from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from sklearn import tree


class tree_node:
    def __init__(self, state: VehicleState, is_goal: bool = False):
        self.state = state
        self.is_goal = is_goal


AdjacencyList = Mapping[tree_node, Set[tree_node]]
"""An adjacency list from node to a set of nodes."""


class EdgeNotFound(Exception):
    pass


class CommandNotFound(Exception):
    pass


@dataclass
class WeightedGraph:
    adj_list: AdjacencyList
    weights: Mapping[tuple[tree_node, tree_node], float]
    cmds: Mapping[tuple[tree_node, tree_node], VehicleCommands]

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

    def add_edge(self, u: tree_node, v: tree_node, weight: float, cmds) -> None:
        """
        :param u: The "from" of the edge
        :param v: The "to" of the edge
        :param weight: The weight of the edge
        """
        if u not in self.adj_list:
            self.adj_list[u] = set()
        self.adj_list[u].add(v)
        self.weights[(u, v)] = weight
        self.cmds[(u, v)] = cmds

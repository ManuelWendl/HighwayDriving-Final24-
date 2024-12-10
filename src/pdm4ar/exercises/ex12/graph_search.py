from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq

from .weighted_graph import WeightedGraph, tree_node
from typing import Optional, List, Tuple


Path = Optional[List[tree_node]]


@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: tree_node) -> Path:
        # Abstract function. Nothing to do here.
        pass


@dataclass
class Astar(InformedGraphSearch):

    def heuristic(self, u: tree_node) -> float:
        # Increment this counter every time the heuristic is called, to judge the performance
        # of the algorithm
        # TODO: Define heuristic as distance from goal lane
        return 0

    def path(self, start: tree_node) -> Path:
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

                if snext not in H:
                    H.update({snext: self.heuristic(snext)})

                if snext not in C or wn < C[snext]:
                    P.update({snext: s})
                    C.update({snext: wn})
                    heapq.heappush(Q, (C[snext] + H[snext], snext))

        return []

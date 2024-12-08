import random
from dataclasses import dataclass
from typing import Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName, SE2Transform
from dg_commons.sim.goals import PlanningGoal, RefLaneGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.dynamics.bicycle_dynamic import BicycleDynamics, VehicleState

from typing import TypeVar


from .motion_primitives import MotionPrimitivesGenerator, MPGParam
from .weighted_graph import WeightedGraph, tree_node

from dg_commons.maps.lanes import DgLanelet
from dg_commons.seq import DgSampledSequence
from dg_commons.eval.comfort import get_acc_rms
from decimal import Decimal
from matplotlib import cm, pyplot as plt
import numpy as np
import shapely

X = TypeVar("X")


@dataclass(frozen=True)
class Pdm4arAgentParams:
    ctrl_timestep: float = 0.1
    ctrl_frequncy: float = 5

    n_velocity: int = 3
    n_steering: int = 3

    delta_angle_threshold: float = np.pi / 4

    max_tree_dpeth: int = 3


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal  # type: ignore
    sg: VehicleGeometry
    sp: VehicleParameters

    dyn: BicycleDynamics

    mpg_params: MPGParam
    mpg: MotionPrimitivesGenerator

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.myplanner = ()
        self.graph: WeightedGraph = None  # type: ignore

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name
        self.goal: RefLaneGoal = init_obs.goal  # type: ignore
        self.sg = init_obs.model_geometry  # type: ignore
        self.sp = init_obs.model_params  # type: ignore
        self.dyn = BicycleDynamics(self.sg, self.sp)
        self.mpg_params = MPGParam.from_vehicle_parameters(
            dt=Decimal(self.params.ctrl_timestep * self.params.ctrl_frequncy),
            n_steps=1,
            n_vel=self.params.n_velocity,
            n_steer=self.params.n_steering,
        )
        self.mpg = MotionPrimitivesGenerator(self.mpg_params, self.dyn.successor_ivp, self.sp)
        self.lanelet_network: LaneletNetwork = init_obs.dg_scenario.lanelet_network  # type: ignore
        self.boundary_obstacles = [
            obstacle.shape.envelope.buffer(-init_obs.dg_scenario.road_boundaries_buffer)  # type: ignore
            for obstacle in init_obs.dg_scenario.static_obstacles  # type: ignore
        ]

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """

        if self.graph is None:
            self.generate_graph(sim_obs)
            self.graph.draw_graph(self.boundary_obstacles)

        # todo implement here some better planning
        rnd_acc = random.random() * self.params.param1
        rnd_ddelta = (random.random() - 0.5) * self.params.param1

        return VehicleCommands(acc=rnd_acc, ddelta=rnd_ddelta)

    def generate_graph(self, sim_obs: SimObservations):
        """
        This method is used to generate a graph from the lanelet network
        This function is called for ego vehicle
        It generates the weighted graph of the motion primitives
        """

        def recursive_adding(state, graph, depth=1):
            trs, cmds = self.mpg.generate(state)
            u = tree_node(state, False)
            for tr, cmd in zip(trs, cmds):
                cost, is_goal, inside_playground, heading_delta_over_threshold = self.calculate_cost(
                    tr.values[-1], cmd, depth * float(tr.timestamps[-1])
                )

                if inside_playground and not heading_delta_over_threshold:
                    v = tree_node(tr.values[-1], is_goal)
                    graph.add_edge(u, v, cost, cmd)
                    if depth < self.params.max_tree_dpeth:
                        recursive_adding(tr.values[-1], graph, depth + 1)

        self.graph = WeightedGraph(
            adj_list={},
            weights={},
            cmds={},
        )

        init_state = sim_obs.players[self.name].state
        recursive_adding(init_state, self.graph)

    def calculate_cost(
        self, future_state: VehicleState, action: VehicleCommands, time: float, sim_obs: SimObservations
    ):
        # pass
        score = 100
        vehicle_shapely = self.get_vehicle_shapely(future_state)
        vehicle_centroid = shapely.Point((future_state.x, future_state.y))
        # 0. Check whether future state is still within playground
        inside_playground = True
        for obstacle in self.boundary_obstacles:
            if shapely.within(vehicle_centroid, obstacle):
                return float("inf"), False, False, False

        # 1. distance and heading wrt goal lane and whether it is a goal node
        inside_goal_lane = shapely.within(vehicle_shapely, self.goal.goal_polygon)

        state_se2transform = SE2Transform([future_state.x, future_state.y], future_state.psi)

        lanelet_id = self.lanelet_network.find_lanelet_by_position([self.goal.ref_lane.control_points[1].q.p])[0][0]
        # lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id=lanelet_id)
        lanelet_new = DgLanelet(self.goal.ref_lane.control_points)
        lane_pose = lanelet_new.lane_pose_from_SE2Transform(state_se2transform)
        heading_delta = lane_pose.relative_heading

        if np.abs(heading_delta) > np.pi / 4:
            heading_delta_over_threshold = True
        else:
            heading_delta_over_threshold = False

        if not inside_goal_lane:
            is_goal_state = False
            distance_to_goal_lane = lane_pose.distance_from_center
            score *= 0.7
            score -= 5 * distance_to_goal_lane

            if np.abs(heading_delta) >= 0.1:
                heading_penalty = (np.abs(heading_delta) - 0.1) * 10.0
                heading_penalty = np.clip(heading_penalty, 0.0, 1.0)
                score -= 5.0 * heading_penalty
        else:
            if np.abs(heading_delta) < 0.1:
                is_goal_state = True
            else:
                is_goal_state = False
                heading_penalty = (np.abs(heading_delta) - 0.1) * 10.0
                heading_penalty = np.clip(heading_penalty, 0.0, 1.0)
                score -= 5.0 * heading_penalty

        # 2. lane changing time
        lane_changing_penalty = (time - 5.0) / 5.0
        lane_changing_penalty = np.clip(lane_changing_penalty, 0.0, 1.0)
        score -= 10.0 * lane_changing_penalty

        # 3. time to collision
        time_to_collision = np.inf
        for player_name, player_obs in sim_obs.players.items():
            if player_name == self.name:
                continue
            other_state: VehicleState = player_obs.state  # type: ignore
            other_lanelet_ids = self.lanelet_network.find_lanelet_by_position(
                [np.array([other_state.x, other_state.y])]
            )

            # Check if the other player is on the same lanelet
            if lanelet_id in other_lanelet_ids:
                other_shapely = self.get_vehicle_shapely(other_state)
                # compute distance between shapelies
                distance = vehicle_shapely.distance(other_shapely)
                time_to_collision = min(time_to_collision, distance / future_state.vx)

        # 4. discomfort level of the action
        # ts = tuple(np.linspace(0, time, 10))
        # action_sequence = DgSampledSequence[VehicleCommands](timestamps=[0, 0.1], values=[action])
        # discomfort = get_acc_rms(action_sequence)
        # discomfort_penalty = (discomfort - 0.6) * 3.0
        # discomfort_penalty = np.clip(discomfort_penalty, 0.0, 1.0)
        # score -= 5.0 * discomfort_penalty

        # 5. vehicle speed
        velocity_difference = np.maximum(future_state.vx - 25.0, 5.0 - future_state.vx)
        velocity_penalty = velocity_difference / 5.0
        velocity_penalty = np.clip(velocity_penalty, 0.0, 1.0)
        score -= 5.0 * velocity_penalty

        return -score, is_goal_state, inside_playground, heading_delta_over_threshold

    def get_vehicle_shapely(self, state: VehicleState):
        cog = np.array([state.x, state.y])
        R = np.array([[np.cos(state.psi), -np.sin(state.psi)], [np.sin(state.psi), np.cos(state.psi)]])
        front_left = cog + R @ np.array([self.sg.lf, self.sg.w_half]).T
        front_right = cog + R @ np.array([self.sg.lf, -self.sg.w_half]).T
        back_left = cog + R @ np.array([self.sg.lr, self.sg.w_half]).T
        back_right = cog + R @ np.array([self.sg.lr, -self.sg.w_half]).T

        vehicle_shapely = shapely.Polygon((tuple(front_left), tuple(front_right), tuple(back_left), tuple(back_right)))

        return vehicle_shapely

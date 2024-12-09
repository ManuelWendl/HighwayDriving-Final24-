from calendar import c
from math import e
import random
from dataclasses import dataclass
from typing import Sequence
from copy import deepcopy

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
from .goal_lane_commands import goal_lane_commands
from .graph_search import Astar

from dg_commons.maps.lanes import DgLanelet
from dg_commons.seq import DgSampledSequence
from dg_commons.eval.comfort import get_acc_rms
from dg_commons.eval.efficiency import desired_lane_reached
from decimal import Decimal
from matplotlib import cm, pyplot as plt
import numpy as np
import shapely
from dg_commons.controllers.steer import SteerController, SteerControllerParam
from dg_commons.controllers.pure_pursuit import PurePursuit, PurePursuitParam
from dg_commons.sim.models.vehicle import VehicleModel


@dataclass(frozen=True)
class Pdm4arAgentParams:
    ctrl_timestep: float = 0.1
    ctrl_frequency: float = 5
    n_velocity: int = 1
    n_steering: int = 3
    n_discretization: int = 50
    delta_angle_threshold: float = np.pi / 4
    max_tree_dpeth: int = 8


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
    gs: Astar

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.myplanner = ()
        self.graph: WeightedGraph = None  # type: ignore
        self.gs = None  # type: ignore

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name
        self.goal: RefLaneGoal = init_obs.goal  # type: ignore
        self.sg = init_obs.model_geometry  # type: ignore
        self.sp = init_obs.model_params  # type: ignore
        self.dyn = BicycleDynamics(self.sg, self.sp)
        self.mpg_params = MPGParam.from_vehicle_parameters(
            dt=Decimal(self.params.ctrl_timestep * self.params.ctrl_frequency),
            n_steps=self.params.n_discretization,
            n_vel=self.params.n_velocity,
            n_steer=self.params.n_steering,
        )
        self.mpg = MotionPrimitivesGenerator(self.mpg_params, self.dyn.successor, self.sp)
        self.lanelet_network: LaneletNetwork = init_obs.dg_scenario.lanelet_network  # type: ignore
        self.boundary_obstacles = [
            obstacle.shape.envelope  # type: ignore
            for obstacle in init_obs.dg_scenario.static_obstacles  # type: ignore
        ]
        self.lanes = [lanelet_polygon.shapely_object for lanelet_polygon in self.lanelet_network.lanelet_polygons]
        self.road_boundaries_buffer = init_obs.dg_scenario.road_boundaries_buffer
        self.lanelet_network: LaneletNetwork = init_obs.dg_scenario.lanelet_network  # type: ignore
        self.goal_lanelet_id = self.lanelet_network.find_lanelet_by_position(
            [self.goal.ref_lane.get_control_points()[0].q.p]
        )[0][0]
        self.max_steering_angle = None
        # goal_lanelet = DgLanelet(self.goal.ref_lane.control_points)
        # width = goal_lanelet.control_points[0].r * 2
        self.goal_reached = False  # Helper variable to save some computation
        self.goal_lane_pid = SteerController.from_vehicle_params(self.sp)
        self.goal_lane_purepursuit = PurePursuit.from_model_geometry(self.sg)

        self.ctrl_num = 0  # Helper variable
        self.path = None  # Helper variable to store the path
        self.motion_primitives = {}

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        if self.max_steering_angle == None:
            state_se2transform = SE2Transform(
                (sim_obs.players["Ego"].state.x, sim_obs.players["Ego"].state.y), sim_obs.players["Ego"].state.psi
            )
            goal_lanelet = DgLanelet(self.goal.ref_lane.control_points)
            lane_pose = goal_lanelet.lane_pose_from_SE2Transform(state_se2transform)
            distance_from_center = lane_pose.distance_from_center / 4  # 4
            dx = sim_obs.players["Ego"].state.vx * self.params.ctrl_frequency * self.params.ctrl_timestep
            self.max_steering_angle = np.pi / 2 - np.arccos(distance_from_center / dx)

        if self.goal_reached:
            return goal_lane_commands(self, sim_obs)
        else:
            # This is taken from L162 ex12/perf_metrics.py
            if desired_lane_reached(
                self.lanelet_network, self.goal, sim_obs.players[self.name].state, pos_tol=0.8, heading_tol=0.08
            ):
                print("Goal lane reached")
                self.goal_reached = True
                return goal_lane_commands(self, sim_obs)

        # START lane change planning
        if self.graph is None:
            self.generate_graph(sim_obs)
            self.graph.draw_graph(self.lanes)
            self.gs = Astar(self.graph)

            self.path = self.gs.path(self.graph.start)
            self.graph.draw_graph(self.lanes, self.path)

        if self.path is not None and (self.ctrl_num // self.params.ctrl_frequency) < (len(self.path) - 1):
            indx = self.ctrl_num // self.params.ctrl_frequency
            commands = self.graph.get_cmds(self.path[indx], self.path[indx + 1])
            print(
                self.path[indx].state.x,
                sim_obs.players[self.name].state.x,
                self.path[indx].state.y,
                sim_obs.players[self.name].state.y,
            )
            acc = commands.acc
            ddelta = commands.ddelta

            self.ctrl_num += 1
            return VehicleCommands(acc=acc, ddelta=ddelta)
        else:
            print("Taking Random Action")
            rnd_acc = random.random() * 0.1
            rnd_ddelta = (random.random() - 0.5) * 0.1

            self.ctrl_num += 1
            return VehicleCommands(acc=rnd_acc, ddelta=rnd_ddelta)

    def generate_graph(self, sim_obs: SimObservations):
        """
        This method is used to generate a graph from the lanelet network
        This function is called for ego vehicle
        It generates the weighted graph of the motion primitives
        """

        def load_motion_primitives(u):
            """
            Load motion primitive if precomputed otherwise compute and store
            """
            if (u.state.vx, np.round(u.state.delta, 1)) not in self.motion_primitives:
                helper_state = VehicleState(x=0, y=0, psi=0, vx=u.state.vx, delta=u.state.delta)
                trs, cmds = self.mpg.generate(helper_state, self.max_steering_angle)
                self.motion_primitives[(u.state.vx, np.round(u.state.delta, 1))] = (
                    trs,
                    cmds,
                )
            else:
                trs, cmds = self.motion_primitives[(u.state.vx, np.round(u.state.delta, 1))]

            return trs, cmds

        def recursive_adding(u, graph, depth=1):
            """
            Recursively add motion primitives to the graph
            """
            if not u.is_goal:
                trs, cmds = load_motion_primitives(u)

                for tr, cmd in zip(deepcopy(trs), deepcopy(cmds)):
                    for val in tr.values:
                        x_temp = val.x * np.cos(u.state.psi) - val.y * np.sin(u.state.psi) + u.state.x
                        val.y = val.x * np.sin(u.state.psi) + val.y * np.cos(u.state.psi) + u.state.y
                        val.x = x_temp
                        val.psi += u.state.psi

                    cost, is_goal, inside_playground, heading_delta_over_threshold = self.calculate_cost(
                        tr.values[-1], cmd, depth * float(tr.timestamps[-1]), sim_obs
                    )

                    if inside_playground and not heading_delta_over_threshold:
                        v = tree_node(tr.values[-1], is_goal)
                        graph.add_edge(u, v, cost, cmd)
                        if depth < self.params.max_tree_dpeth:
                            recursive_adding(v, graph, depth + 1)
            else:
                print("Goal node reached")

        init_state = sim_obs.players[self.name].state
        init_node = tree_node(state=init_state, is_goal=False)

        self.graph = WeightedGraph(
            adj_list={},
            weights={},
            cmds={},
            start=init_node,
        )
        recursive_adding(init_node, self.graph)

    def calculate_cost(
        self, future_state: VehicleState, action: VehicleCommands, time: float, sim_obs: SimObservations
    ):
        # pass
        score = 100
        vehicle_shapely = self.get_vehicle_shapely(future_state)
        # 0. Check whether future state is still within playground
        inside_playground = True
        lanes_union = shapely.unary_union(self.lanes).buffer(self.road_boundaries_buffer / 2)
        if not shapely.within(vehicle_shapely, lanes_union):
            return float("inf"), False, False, False

        # 1. distance and heading wrt goal lane and whether it is a goal node

        state_se2transform = SE2Transform([future_state.x, future_state.y], future_state.psi)

        lanelet_id = self.lanelet_network.find_lanelet_by_position([self.goal.ref_lane.control_points[1].q.p])[0][0]
        # lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id=lanelet_id)
        lanelet_new = DgLanelet(self.goal.ref_lane.control_points)
        lane_pose = lanelet_new.lane_pose_from_SE2Transform(state_se2transform)
        heading_delta = lane_pose.relative_heading

        if np.abs(heading_delta) > np.pi / 4:
            heading_delta_over_threshold = True
            return float("inf"), False, False, True

        else:
            heading_delta_over_threshold = False

        if desired_lane_reached(self.lanelet_network, self.goal, future_state, pos_tol=0.8, heading_tol=0.08):
            is_goal_state = True
        else:
            is_goal_state = False
            distance_to_goal_lane = lane_pose.distance_from_center
            score *= 0.7
            score -= 5 * distance_to_goal_lane

        if np.abs(heading_delta) >= 0.1:
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
        # TODO: Would it be better here to take into account the previous action to evaluate the change in acc and ddelta?
        N_SAMPLES_DISCOMFORT = 10
        ts = tuple(np.linspace(0, self.params.ctrl_frequency * self.params.ctrl_timestep, N_SAMPLES_DISCOMFORT))
        # ts = tuple(np.linspace(0, 0.5, N_SAMPLES_DISCOMFORT))
        # acc = [action.acc] * N_SAMPLES_DISCOMFORT
        # ddelta = [action.ddelta] * N_SAMPLES_DISCOMFORT
        # cmds_list = [VehicleCommands(acc_i, ddelta_i) for acc_i, ddelta_i in zip(acc, ddelta)]
        cmds_list = [action] * N_SAMPLES_DISCOMFORT
        action_sequence = DgSampledSequence[VehicleCommands](timestamps=ts, values=cmds_list)
        discomfort = get_acc_rms(action_sequence)
        discomfort_penalty = (discomfort - 0.6) * 3.0
        discomfort_penalty = np.clip(discomfort_penalty, 0.0, 1.0)
        score -= 5.0 * discomfort_penalty

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

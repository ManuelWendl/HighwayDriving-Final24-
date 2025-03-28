from asyncio import FastChildWatcher
from hmac import new
from json import load
from operator import add
from tabnanny import verbose
from typing import List
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

from pdm4ar.exercises_def.ex09 import goal


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

from .utils import get_vehicle_shapely


@dataclass(frozen=False)
class Pdm4arAgentParams:
    ctrl_timestep: float = 0.1
    ctrl_frequency: float = 4
    n_velocity: int = 1  # keep this 1 for now (avoid dense trees and long computation time)
    n_steering: int = 3
    n_discretization: int = 50
    delta_angle_threshold: float = np.pi / 4
    max_tree_dpeth: int = 6
    num_lanes_outside_reach: int = 2
    use_velocity_variation = True
    goal_velocity = None
    min_velocity = 0.7

    n_velocity_opponent: int = 3
    probability_threshold_opponent: float = 0.01
    probability_good_opponent: float = 1 / 3
    max_acceleration_factor_opponent: float = 1 / 3

    verbose = False


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
        self.mpg_params = None
        self.mpg = None
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

        self.goal_lanelet = DgLanelet(self.goal.ref_lane.control_points)
        self.lanewidth = self.goal.ref_lane.radius(0) * 2

        self.max_steering_angle_change = None
        # goal_lanelet = DgLanelet(self.goal.ref_lane.control_points)
        # width = goal_lanelet.control_points[0].r * 2
        self.goal_reached = False  # Helper variable to save some computation
        self.goal_lane_pid = SteerController.from_vehicle_params(self.sp)
        self.goal_lane_purepursuit = PurePursuit.from_model_geometry(self.sg)

        self.ctrl_num = 0  # Helper variable
        self.path = None  # Helper variable to store the path
        self.motion_primitives = {}  # Helper variable to store the motion primitives
        self.last_next_state = None  # Helper variable to store the last next state
        self.pose_on_init = None

        self.in_merge = False  # Helper boolean to indicate if in merge opertion
        self.last_min_v = (0, None)  # Helper variable to store the last minimal velocity and corresponding player

    def optimize_ctrlfreq_steeringangle(self, sim_obs: SimObservations, goal_velocity: float):
        """
        This method is used to optimize the control frequency and the steering angle
        """
        state = VehicleState(x=0, y=0, psi=0, vx=goal_velocity, delta=0)
        time_count = 0
        # TODO: TUNE HEURISTICALLY TO FIT THE LANDEWIDTH
        if goal_velocity <= 5.5:
            factor = 25  # Factor for low speeds to get an additional motion primitive during lane change to force the other vehicles to slow down.
        else:
            factor = 8

        while state.y < self.lanewidth / factor:
            time_count += 1
            state = self.dyn.successor(state, VehicleCommands(acc=0, ddelta=self.sp.ddelta_max), 0.01)

        horizon = time_count * 0.01
        self.params.ctrl_frequency = np.ceil(horizon / self.params.ctrl_timestep)

        ddelta_max = self.sp.ddelta_max  # self.sp.delta_max / (self.params.ctrl_timestep * self.params.ctrl_frequency)

        assert ddelta_max <= self.sp.delta_max / (self.params.ctrl_timestep * self.params.ctrl_frequency)

        lowerbound_ddelta = ddelta_max / 2
        upperbound_ddelta = ddelta_max

        while np.abs(state.y - self.lanewidth / factor) >= 1e-3:
            state = VehicleState(x=0, y=0, psi=0, vx=goal_velocity, delta=0)
            for _ in range(int(self.params.ctrl_frequency * self.params.ctrl_timestep / 0.01)):
                state = self.dyn.successor(
                    state, VehicleCommands(acc=0, ddelta=(upperbound_ddelta + lowerbound_ddelta) / 2), 0.01
                )
            if state.y < self.lanewidth / factor:
                lowerbound_ddelta = (upperbound_ddelta + lowerbound_ddelta) / 2
            else:
                upperbound_ddelta = (upperbound_ddelta + lowerbound_ddelta) / 2

        self.max_steering_angle_change = (upperbound_ddelta + lowerbound_ddelta) / 2

    def get_nearest_goal_lane_velocity(self, sim_obs: SimObservations):
        """
        Returns the velocity, relative position (in front/ in back) and distance of closest car on goal lane
        """
        min_dist = float("inf")
        min_velocity = 0
        min_direction = 0
        min_player_name = None
        for player_name, player in sim_obs.players.items():
            if player_name != "Ego":
                state_se2transform = SE2Transform([player.state.x, player.state.y], player.state.psi)
                lanelet_new = DgLanelet(self.goal.ref_lane.control_points)
                lane_pose = lanelet_new.lane_pose_from_SE2Transform(state_se2transform)
                if lane_pose.distance_from_center <= self.lanewidth * 2 / 3:
                    # Calculate the distance to ego:
                    d_vec = np.array(
                        [
                            player.state.x - sim_obs.players["Ego"].state.x,
                            player.state.y - sim_obs.players["Ego"].state.y,
                        ]
                    )

                    distance_to_ego = np.linalg.norm(d_vec)
                    norm_ego = np.array(
                        [np.cos(sim_obs.players["Ego"].state.psi), np.sin(sim_obs.players["Ego"].state.psi)]
                    )
                    direction = np.sign(np.dot(d_vec, norm_ego))
                    if distance_to_ego < min_dist:
                        min_velocity = player.state.vx
                        min_direction = direction
                        min_dist = distance_to_ego
                        min_player_name = player_name

        return (
            min_velocity,
            min_direction,
            min_dist,
            min_player_name,
        )

    def get_avg_goal_lane_velocity(self, sim_obs: SimObservations):
        """
        Returns the average velocity on the goal lane
        """
        avg_velocity = 0
        counter = 0
        for player_name, player in sim_obs.players.items():
            if player_name != "Ego":
                state_se2transform = SE2Transform([player.state.x, player.state.y], player.state.psi)
                lanelet_new = DgLanelet(self.goal.ref_lane.control_points)
                lane_pose = lanelet_new.lane_pose_from_SE2Transform(state_se2transform)
                if lane_pose.distance_from_center <= self.lanewidth * 2 / 3:
                    avg_velocity += player.state.vx
                    counter += 1

        if counter != 0:
            avg_velocity /= counter
        else:
            avg_velocity = sim_obs.players["Ego"].state.vx

        return avg_velocity

    def set_mpg_params(self, sim_obs: SimObservations):
        self.optimize_ctrlfreq_steeringangle(sim_obs, self.goal_velocity)
        self.mpg_params = MPGParam.from_vehicle_parameters(
            dt=Decimal(self.params.ctrl_timestep * self.params.ctrl_frequency),
            n_steps=self.params.n_discretization,
            n_vel=self.params.n_velocity,
            n_steer=self.params.n_steering,
        )
        self.mpg = MotionPrimitivesGenerator(self.mpg_params, self.dyn.successor, self.sp)

    def klebe_am_arsch_vom_vordermann(self, sim_obs: SimObservations):
        dir_ego = np.array([np.cos(sim_obs.players["Ego"].state.psi), np.sin(sim_obs.players["Ego"].state.psi)])
        min_dist = float("inf")
        min_vel = 0
        for player_name, player in sim_obs.players.items():
            if player_name != "Ego":
                d_vec = np.array(
                    [
                        player.state.x - sim_obs.players["Ego"].state.x,
                        player.state.y - sim_obs.players["Ego"].state.y,
                    ]
                )
                if np.dot(d_vec, dir_ego) > 0 and np.linalg.norm(d_vec) < min_dist:
                    min_dist = np.linalg.norm(d_vec)
                    min_vel = player.state.vx

        return min_vel

    def get_bool_lane_merge(self, avg_v, last_min_v, min_v, min_dist, min_player_name):
        """
        Returns a boolean wether to start the lane merge:
        Watch that function only returns true in dense case!
        """
        if avg_v <= 4 and self.last_min_v[1] == min_player_name:
            vel_req = min_v <= 2 and last_min_v[0] - min_v >= 0.25
            dist_req = min_dist <= self.lanewidth * 2

            return vel_req and dist_req
        else:
            return False

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """

        # START On Initial get_commands call
        if self.pose_on_init is None:
            self.pose_on_init = sim_obs.players["Ego"].state

        if self.max_steering_angle_change is None:
            if self.params.use_velocity_variation:
                self.goal_velocity = max(4.75, self.get_avg_goal_lane_velocity(sim_obs))
            else:
                self.goal_velocity = sim_obs.players["Ego"].state.vx

            self.goal_velocity = np.clip(self.goal_velocity, self.params.min_velocity, self.sp.vx_limits[1])
            self.set_mpg_params(sim_obs)

        if self.graph is None:
            self.generate_ego_graph(sim_obs)
            if self.params.verbose:
                self.graph.draw_graph(
                    self.lanes, pose_on_init=self.pose_on_init, current_players=sim_obs.players, sg=self.sg
                )
            self.gs = Astar(self.graph, self.sg)

        # START If goal lane is reached
        if self.goal_reached:
            return goal_lane_commands(self, sim_obs)
        else:
            # This is taken from L162 ex12/perf_metrics.py
            if desired_lane_reached(
                self.lanelet_network,
                self.goal,
                sim_obs.players[self.name].state,
                pos_tol=0.8,
                heading_tol=0.1,  # allow larger heading difference
            ):
                print("Goal lane reached")
                self.goal_reached = True
                return goal_lane_commands(self, sim_obs)

        # START lane change planning
        if self.ctrl_num % self.params.ctrl_frequency == 0:

            # Generate the graph for the opponent vehicles
            opponent_graphs, depth_dicts = self.get_oponent_graph(sim_obs)

            # Get the average and the minimal velocity
            avg_velocity = self.get_avg_goal_lane_velocity(sim_obs)
            if avg_velocity <= 4:
                # Add additional buffer for dense lane merging
                self.gs.params.collision_buffer = 0.2

            min_velocity, min_dir, min_dist, min_player_name = self.get_nearest_goal_lane_velocity(sim_obs)
            if (
                self.get_bool_lane_merge(avg_velocity, self.last_min_v, min_velocity, min_dist, min_player_name)
                and not self.in_merge
            ):
                print("===Start lane merge===")
                self.goal_velocity = max(
                    min_velocity - 0.0 * min_dir,
                    sim_obs.players["Ego"].state.vx
                    + self.sp.acc_limits[0] * self.params.ctrl_frequency * self.params.ctrl_timestep,
                )
                self.goal_velocity = np.clip(self.goal_velocity, self.params.min_velocity, self.sp.vx_limits[1])
                self.set_mpg_params(sim_obs)
                self.generate_ego_graph(sim_obs)
                self.gs = Astar(self.graph, self.sg)
                self.path = []
                self.in_merge = True
                self.ctrl_num = 0

            elif self.in_merge:
                vel_car_in_front = self.klebe_am_arsch_vom_vordermann(sim_obs)
                self.goal_velocity = np.clip(vel_car_in_front, self.params.min_velocity, self.sp.vx_limits[1])
                self.set_mpg_params(sim_obs)
                self.generate_ego_graph(sim_obs)
                self.gs = Astar(self.graph, self.sg)
                self.path = []
                self.in_merge = True
                self.ctrl_num = 0

            elif self.last_next_state is not None:
                self.update_ego_tree(self.graph, self.last_next_state, sim_obs)
            # Set last minimal velocity
            self.last_min_v = (min_velocity, min_player_name)

            # Generate the current path
            if self.gs.goal_lanelet is None:
                self.gs.goal_lanelet = self.goal_lanelet
            self.path = self.gs.path(
                self.graph.start,
                depth_dicts,
                sim_obs,
                safe_depth=self.params.max_tree_dpeth + 1,
                in_merge=self.in_merge,
            )

            safe_depth = deepcopy(self.params.max_tree_dpeth)
            while self.path == [] and safe_depth > 1:
                safe_depth -= 1
                print("Search for safe path with depth:", safe_depth)
                self.path = self.gs.path(self.graph.start, depth_dicts, sim_obs, safe_depth=safe_depth)

            if self.path == []:
                print("No path found")
                # Get the next state that has the same delta (ie ddelta = 0)
                if self.graph.start.successors == []:
                    return VehicleCommands(acc=0, ddelta=0)
                else:
                    self.last_next_state = [
                        successor
                        for successor in self.graph.start.successors
                        if np.isclose(successor.state.delta, self.graph.start.state.delta)
                    ][0]
                    return VehicleCommands(acc=0, ddelta=0)
            if self.params.verbose:
                self.graph.draw_graph(
                    self.lanes, self.path, opponent_graphs, self.pose_on_init, sim_obs.players, self.sg
                )

        # Extract the commands from the path (MPC style)
        commands = self.graph.get_cmds(self.path[0], self.path[1])
        self.last_next_state = self.path[1]

        acc = commands.acc
        ddelta = commands.ddelta
        self.ctrl_num += 1
        return VehicleCommands(acc=acc, ddelta=ddelta)

    def load_motion_primitives(self, u):
        """
        Load motion primitive if precomputed otherwise compute and store
        """
        if (np.round(u.state.vx, 3), np.round(u.state.delta, 1)) not in self.motion_primitives or np.abs(
            u.state.vx - self.goal_velocity
        ) >= 1e-3:
            helper_state = VehicleState(x=0, y=0, psi=0, vx=u.state.vx, delta=u.state.delta)
            if np.abs(u.state.vx - self.goal_velocity) >= 1e-3:
                trs, cmds = self.mpg.generate(
                    helper_state, self.max_steering_angle_change, u.depth, goal_velocity=self.goal_velocity
                )
            else:
                trs, cmds = self.mpg.generate(
                    helper_state, self.max_steering_angle_change, u.depth, goal_velocity=u.state.vx
                )
            # Only store the motion primitives with zero acceleration
            trs_acc_zero = [trs for trs, cmd in zip(trs, cmds) if cmd.acc == 0]
            cmds_acc_zero = [cmd for trs, cmd in zip(trs, cmds) if cmd.acc == 0]
            if trs_acc_zero != []:
                self.motion_primitives[(np.round(u.state.vx, 3), np.round(u.state.delta, 1))] = (
                    trs_acc_zero,
                    cmds_acc_zero,
                )
        else:
            trs, cmds = self.motion_primitives[(np.round(u.state.vx, 3), np.round(u.state.delta, 1))]

        return trs, cmds

    def generate_ego_graph(self, sim_obs: SimObservations):
        """
        This method is used to generate a graph from the lanelet network
        This function is called for ego vehicle
        It generates the weighted graph of the motion primitives
        """

        def recursive_adding(u, graph, depth=1):
            """
            Recursively add motion primitives to the graph
            """
            if u.data != 1:
                trs, cmds = self.load_motion_primitives(u)

                for val, cmd in zip(deepcopy(trs), deepcopy(cmds)):
                    x_temp = val.x * np.cos(u.state.psi) - val.y * np.sin(u.state.psi) + u.state.x
                    val.y = val.x * np.sin(u.state.psi) + val.y * np.cos(u.state.psi) + u.state.y
                    val.x = x_temp
                    val.psi += u.state.psi

                    cost, is_goal, inside_playground, heading_delta_over_threshold, is_outside_reach = (
                        self.calculate_cost(
                            val, cmd, depth * self.params.ctrl_frequency * self.params.ctrl_timestep, sim_obs
                        )
                    )

                    if inside_playground and not heading_delta_over_threshold and not is_outside_reach:
                        v = tree_node(val, depth=depth, data=float(is_goal))
                        graph.add_edge(u, v, cost, cmd)
                        if depth < self.params.max_tree_dpeth:
                            recursive_adding(v, graph, depth + 1)
            else:
                print("Goal node reached")

        init_state = sim_obs.players[self.name].state
        init_node = tree_node(state=init_state, depth=0, data=0)

        self.graph = WeightedGraph(
            weights={},
            cmds={},
            start=init_node,
        )
        recursive_adding(init_node, self.graph)

    def update_ego_tree(self, graph, reached_node: tree_node, sim_obs: SimObservations):
        """
        This method is used to update the ego tree
        """

        def add_successors(node):
            for s in node.successors:
                if s.successors != []:
                    s.depth -= 1
                    add_successors(s)
                else:
                    s.depth -= 1
                    trs, cmds = self.load_motion_primitives(s)
                    for val, cmd in zip(deepcopy(trs), deepcopy(cmds)):
                        x_temp = val.x * np.cos(s.state.psi) - val.y * np.sin(s.state.psi) + s.state.x
                        val.y = val.x * np.sin(s.state.psi) + val.y * np.cos(s.state.psi) + s.state.y
                        val.x = x_temp
                        val.psi += s.state.psi

                        cost, is_goal, inside_playground, heading_delta_over_threshold, is_outside_reach = (
                            self.calculate_cost(
                                val, cmd, s.depth * self.params.ctrl_frequency * self.params.ctrl_timestep, sim_obs
                            )
                        )

                        if inside_playground and not heading_delta_over_threshold and not is_outside_reach:
                            v = tree_node(val, depth=s.depth + 1, data=float(is_goal))
                            graph.add_edge(s, v, cost, cmd)
                            if s.depth + 1 < self.params.max_tree_dpeth:
                                add_successors(v)

        newstart = None

        for node in graph.start.successors:
            del graph.weights[(graph.start, node)]
            del graph.cmds[(graph.start, node)]

            if node == reached_node:
                newstart = reached_node
                newstart.depth -= 1
                if self.params.n_velocity > 1:
                    # Added to avoid not adding new velocity branches to the graph at depth=0
                    trs, cmds = self.load_motion_primitives(newstart)
                    for val, cmd in zip(deepcopy(trs), deepcopy(cmds)):
                        if cmd.acc == 0:
                            continue
                        else:
                            x_temp = (
                                val.x * np.cos(newstart.state.psi)
                                - val.y * np.sin(newstart.state.psi)
                                + newstart.state.x
                            )
                            val.y = (
                                val.x * np.sin(newstart.state.psi)
                                + val.y * np.cos(newstart.state.psi)
                                + newstart.state.y
                            )
                            val.x = x_temp
                            val.psi += newstart.state.psi

                            cost, is_goal, inside_playground, heading_delta_over_threshold, is_outside_reach = (
                                self.calculate_cost(
                                    val,
                                    cmd,
                                    newstart.depth * self.params.ctrl_frequency * self.params.ctrl_timestep,
                                    sim_obs,
                                )
                            )

                            if inside_playground and not heading_delta_over_threshold and not is_outside_reach:
                                v = tree_node(val, depth=newstart.depth, data=float(is_goal))
                                graph.add_edge(newstart, v, cost, cmd)
                                add_successors(v)

                        for s in newstart.successors:
                            add_successors(s)
                else:
                    add_successors(newstart)
            else:
                graph.remove_node(node)

        if newstart is not None:
            graph.start = newstart

        else:
            print("Next node not found in the graph")

    def calculate_cost(
        self, future_state: VehicleState, action: VehicleCommands, time: float, sim_obs: SimObservations
    ):
        score = 100
        vehicle_shapely = get_vehicle_shapely(self.sg, future_state)
        # 0. Check whether future state is still within playground
        inside_playground = True
        lanes_union = shapely.unary_union(self.lanes).buffer(self.road_boundaries_buffer)
        if not shapely.within(vehicle_shapely, lanes_union):
            return float("inf"), False, False, False, False

        # 1. distance and heading wrt goal lane and whether it is a goal node

        state_se2transform = SE2Transform([future_state.x, future_state.y], future_state.psi)

        lanelet_id = self.lanelet_network.find_lanelet_by_position([self.goal.ref_lane.control_points[1].q.p])[0][0]
        # lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id=lanelet_id)
        lanelet_new = DgLanelet(self.goal.ref_lane.control_points)
        lane_pose = lanelet_new.lane_pose_from_SE2Transform(state_se2transform)
        heading_delta = lane_pose.relative_heading

        lane_normal_vector_angle = future_state.psi - heading_delta + np.pi / 2
        lane_normal_vector = np.array([np.cos(lane_normal_vector_angle), np.sin(lane_normal_vector_angle)]).T

        if (
            np.linalg.norm(
                (np.array([future_state.x - self.pose_on_init.x, future_state.y - self.pose_on_init.y]))
                @ lane_normal_vector
            )
            > self.params.num_lanes_outside_reach * self.lanewidth
        ):
            is_outside_reach = True
        else:
            is_outside_reach = False

        if np.abs(heading_delta) > np.pi / 3:
            heading_delta_over_threshold = True
            return float("inf"), False, False, True, False

        else:
            heading_delta_over_threshold = False

        if desired_lane_reached(self.lanelet_network, self.goal, future_state, pos_tol=0.8, heading_tol=0.1):
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
        lane_changing_penalty = time / 5.0
        lane_changing_penalty = np.clip(lane_changing_penalty, 0.0, 1.0)
        score -= 500.0 * lane_changing_penalty

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
                other_shapely = get_vehicle_shapely(self.sg, other_state)
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

        return -score, is_goal_state, inside_playground, heading_delta_over_threshold, is_outside_reach

    def get_oponent_graph(self, sim_obs: SimObservations):
        """
        Generates the graphs for the opponent vehicle
        """

        def recursive_adding(u, graph, depth_dict, depth):
            """
            Recursively add motion primitives to the graph
            """

            if self.params.n_velocity_opponent == 1:
                velocities = [u.state.vx]
            else:
                symmetric_steps = self.params.n_velocity_opponent // 2

                rel_pos_factor = 0
                ego_state = sim_obs.players["Ego"].state
                ego_x = (
                    ego_state.x
                    + np.cos(ego_state.psi)
                    * ego_state.vx
                    * self.params.ctrl_timestep
                    * self.params.ctrl_frequency
                    * depth
                )
                ego_y = (
                    ego_state.y
                    + np.sin(ego_state.psi)
                    * ego_state.vx
                    * self.params.ctrl_timestep
                    * self.params.ctrl_frequency
                    * depth
                )

                ego_direction = np.array([np.cos(ego_state.psi), np.sin(ego_state.psi)])
                difference_vector = np.array([u.state.x - ego_x, u.state.y - ego_y])
                direction = np.sign(np.dot(ego_direction, difference_vector))

                if direction == 1:
                    if self.in_merge:
                        rel_pos_factor_upper = 3
                        rel_pos_factor_lower = 0
                    else:
                        rel_pos_factor_upper = 3
                        rel_pos_factor_lower = 1 / 3
                elif direction == -1:
                    if self.in_merge:
                        rel_pos_factor_upper = 0
                        rel_pos_factor_lower = 3
                    else:
                        rel_pos_factor_upper = 1 / 3
                        rel_pos_factor_lower = 3

                velocities_lower = np.array(
                    [
                        u.state.vx
                        + self.sp.acc_limits[0]
                        * self.params.max_acceleration_factor_opponent
                        * rel_pos_factor_lower
                        / symmetric_steps
                        * step
                        * float(self.params.ctrl_timestep * self.params.ctrl_frequency)
                        for step in range(1, symmetric_steps + 1)
                    ]
                )

                velocities_upper = np.array(
                    [
                        u.state.vx
                        + self.sp.acc_limits[1]
                        * self.params.max_acceleration_factor_opponent
                        * rel_pos_factor_upper
                        / symmetric_steps
                        * step
                        * float(self.params.ctrl_timestep * self.params.ctrl_frequency)
                        for step in range(1, symmetric_steps + 1)
                    ]
                )

                velocities = np.concatenate((velocities_lower, np.array([u.state.vx]), velocities_upper))
                velocities = np.clip(velocities, 0, self.sp.vx_limits[1])

            d_x_normal = np.cos(u.state.psi)
            d_y_normal = np.sin(u.state.psi)
            dt = self.params.ctrl_timestep * self.params.ctrl_frequency

            for v in velocities:
                dx = d_x_normal * (v + u.state.vx) / 2 * dt
                dy = d_y_normal * (v + u.state.vx) / 2 * dt
                next_state = VehicleState(
                    x=u.state.x + dx, y=u.state.y + dy, psi=u.state.psi, vx=v, delta=u.state.delta
                )
                if v == u.state.vx:
                    cp = self.params.probability_good_opponent
                else:
                    cp = (1 - self.params.probability_good_opponent) / (self.params.n_velocity_opponent - 1)

                p = u.data * cp
                if p > self.params.probability_threshold_opponent:
                    v = tree_node(next_state, depth=depth, data=p)
                    if depth not in depth_dict:
                        depth_dict[depth] = []
                    depth_dict[depth].append(v)
                    graph.add_edge(u, v, 0, VehicleCommands(acc=0, ddelta=0))

                    if depth < self.params.max_tree_dpeth:
                        recursive_adding(v, graph, depth_dict, depth + 1)

        graphs = {}
        depth_dicts = {}
        for player_name, player_obs in sim_obs.players.items():
            if player_name == self.name:
                continue
            init_state = player_obs.state
            init_node = tree_node(state=init_state, depth=0, data=1)

            graph = WeightedGraph(
                weights={},
                cmds={},
                start=init_node,
            )
            depth_dict = {}
            recursive_adding(init_node, graph, depth_dict=depth_dict, depth=1)
            graphs[player_name] = graph
            depth_dicts[player_name] = depth_dict

        return graphs, depth_dicts

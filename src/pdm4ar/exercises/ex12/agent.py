from hmac import new
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
    n_velocity: int = 1
    n_steering: int = 3
    n_discretization: int = 50
    delta_angle_threshold: float = np.pi / 4
    max_tree_dpeth: int = 10
    num_lanes_outside_reach: int = 1.5

    n_velocity_opponent: int = 3
    probability_threshold_opponent: float = 0.01
    probability_good_opponent: float = 0.5


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

        state_se2transform = SE2Transform([0, 0], 0)
        goal_lanelet = DgLanelet(self.goal.ref_lane.control_points)
        # lane_pose = goal_lanelet.lane_pose_from_SE2Transform(state_se2transform)
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

    def optimize_ctrlfreq_steeringangle(self, sim_obs: SimObservations):
        """
        This method is used to optimize the control frequency and the steering angle
        """
        state = VehicleState(x=0, y=0, psi=0, vx=sim_obs.players["Ego"].state.vx, delta=0)
        time_count = 0
        # TODO: TUNE HEURISTICALLY TO FIT THE LANDEWIDTH
        while state.y < self.lanewidth / 8:
            time_count += 1
            state = self.dyn.successor(state, VehicleCommands(acc=0, ddelta=self.sp.ddelta_max), 0.01)

        horizon = time_count * 0.01
        self.params.ctrl_frequency = np.ceil(horizon / self.params.ctrl_timestep)

        ddelta_max = self.sp.ddelta_max  # self.sp.delta_max / (self.params.ctrl_timestep * self.params.ctrl_frequency)

        assert ddelta_max <= self.sp.delta_max / (self.params.ctrl_timestep * self.params.ctrl_frequency)

        lowerbound_ddelta = ddelta_max / 2
        upperbound_ddelta = ddelta_max

        while np.abs(state.y - self.lanewidth / 8) >= 1e-3:
            state = VehicleState(x=0, y=0, psi=0, vx=sim_obs.players["Ego"].state.vx, delta=0)
            for _ in range(int(self.params.ctrl_frequency * self.params.ctrl_timestep / 0.01)):
                state = self.dyn.successor(
                    state, VehicleCommands(acc=0, ddelta=(upperbound_ddelta + lowerbound_ddelta) / 2), 0.01
                )
            if state.y < self.lanewidth / 8:
                lowerbound_ddelta = (upperbound_ddelta + lowerbound_ddelta) / 2
            else:
                upperbound_ddelta = (upperbound_ddelta + lowerbound_ddelta) / 2

        self.max_steering_angle_change = (upperbound_ddelta + lowerbound_ddelta) / 2

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """
        if self.max_steering_angle_change == None:
            self.optimize_ctrlfreq_steeringangle(sim_obs)
            self.mpg_params = MPGParam.from_vehicle_parameters(
                dt=Decimal(self.params.ctrl_timestep * self.params.ctrl_frequency),
                n_steps=self.params.n_discretization,
                n_vel=self.params.n_velocity,
                n_steer=self.params.n_steering,
            )
            self.mpg = MotionPrimitivesGenerator(self.mpg_params, self.dyn.successor, self.sp)

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
            self.generate_ego_graph(sim_obs)
            self.graph.draw_graph(self.lanes)
            self.gs = Astar(self.graph, self.sg)

        if self.ctrl_num % self.params.ctrl_frequency == 0:
            # Generate the graph for the opponent vehicles
            opponent_graphs, depth_dicts = self.get_oponent_graph(sim_obs)
            # TODO: Implement ego tree update
            if self.last_next_state is not None:
                self.update_ego_tree(self.graph, self.last_next_state, sim_obs)
            # Generate the current path
            self.path = self.gs.path(self.graph.start, depth_dicts, sim_obs)
            if self.path == []:
                print("No path found")
                # Get the next state that has the same delta (ie ddelta = 0)
                self.last_next_state = [
                    successor
                    for successor in self.graph.start.successors
                    if np.isclose(successor.state.delta, self.graph.start.state.delta)
                ][0]
                return VehicleCommands(acc=0, ddelta=0)
            self.graph.draw_graph(self.lanes, self.path, opponent_graphs)

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
        if (u.state.vx, np.round(u.state.delta, 1)) not in self.motion_primitives:
            helper_state = VehicleState(x=0, y=0, psi=0, vx=u.state.vx, delta=u.state.delta)
            trs, cmds = self.mpg.generate(helper_state, self.max_steering_angle_change)
            self.motion_primitives[(u.state.vx, np.round(u.state.delta, 1))] = (
                trs,
                cmds,
            )
        else:
            trs, cmds = self.motion_primitives[(u.state.vx, np.round(u.state.delta, 1))]

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

                for tr, cmd in zip(deepcopy(trs), deepcopy(cmds)):
                    for val in tr.values:
                        x_temp = val.x * np.cos(u.state.psi) - val.y * np.sin(u.state.psi) + u.state.x
                        val.y = val.x * np.sin(u.state.psi) + val.y * np.cos(u.state.psi) + u.state.y
                        val.x = x_temp
                        val.psi += u.state.psi

                    cost, is_goal, inside_playground, heading_delta_over_threshold, is_outside_reach = (
                        self.calculate_cost(tr.values[-1], cmd, depth * float(tr.timestamps[-1]), sim_obs)
                    )

                    if inside_playground and not heading_delta_over_threshold and not is_outside_reach:
                        v = tree_node(tr.values[-1], depth=depth, data=float(is_goal))
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
                    add_successors(s)
                    s.depth -= 1
                else:
                    trs, cmds = self.load_motion_primitives(s)
                    for tr, cmd in zip(deepcopy(trs), deepcopy(cmds)):
                        for val in tr.values:
                            x_temp = val.x * np.cos(s.state.psi) - val.y * np.sin(s.state.psi) + s.state.x
                            val.y = val.x * np.sin(s.state.psi) + val.y * np.cos(s.state.psi) + s.state.y
                            val.x = x_temp
                            val.psi += s.state.psi

                        cost, is_goal, inside_playground, heading_delta_over_threshold, is_outside_reach = (
                            self.calculate_cost(tr.values[-1], cmd, s.depth * float(tr.timestamps[-1]), sim_obs)
                        )

                        if inside_playground and not heading_delta_over_threshold and not is_outside_reach:
                            v = tree_node(tr.values[-1], depth=s.depth, data=float(is_goal))
                            graph.add_edge(s, v, cost, cmd)

        newstart = None

        for node in graph.start.successors:
            del graph.weights[(graph.start, node)]
            del graph.cmds[(graph.start, node)]

            if node == reached_node:
                newstart = reached_node
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
                (np.array([future_state.x - self.graph.start.state.x, future_state.y - self.graph.start.state.y]))
                @ lane_normal_vector
            )
            > self.params.num_lanes_outside_reach * self.lanewidth
        ):
            is_outside_reach = True
        else:
            is_outside_reach = False

        if np.abs(heading_delta) > np.pi / 4:
            heading_delta_over_threshold = True
            return float("inf"), False, False, True, False

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
        lane_changing_penalty = time / 5.0
        lane_changing_penalty = np.clip(lane_changing_penalty, 0.0, 1.0)
        score -= 100.0 * lane_changing_penalty

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

                velocities_lower = np.array(
                    [
                        u.state.vx
                        + self.sp.acc_limits[0]
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
                        / symmetric_steps
                        * step
                        * float(self.params.ctrl_timestep * self.params.ctrl_frequency)
                        for step in range(1, symmetric_steps + 1)
                    ]
                )

                velocities = np.concatenate((velocities_lower, np.array([u.state.vx]), velocities_upper))

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

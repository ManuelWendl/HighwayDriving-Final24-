import random
from dataclasses import dataclass
from typing import Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName, SE2Transform
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters

from .motion_primitives import MotionPrimitivesGenerator, MPGParam
from dg_commons.dynamics.bicycle_dynamic import BicycleDynamics, VehicleState
from dg_commons.maps.lanes import DgLanelet
from dg_commons.seq import DgSampledSequence
from dg_commons.eval.comfort import get_acc_rms
from decimal import Decimal
from matplotlib import pyplot as plt
import numpy as np
import shapely


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    sg: VehicleGeometry
    sp: VehicleParameters

    mpg_params: MPGParam
    mpg: MotionPrimitivesGenerator
    dyn: BicycleDynamics

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.myplanner = ()

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params
        self.dyn = BicycleDynamics(self.sg, self.sp)
        self.mpg_params = MPGParam.from_vehicle_parameters(dt=Decimal(0.01), n_steps=10, n_vel=5, n_steer=5)
        self.mpg = MotionPrimitivesGenerator(self.mpg_params, self.dyn.successor_ivp, self.sp)
        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.boundary_obstacles = [
            obstacle.shape.envelope.buffer(-init_obs.dg_scenario.road_boundaries_buffer)
            for obstacle in init_obs.dg_scenario.static_obstacles
        ]

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """

        # Update linspace of velocities and actions
        current_ego_state = sim_obs.players[self.name].state
        tr, cmds = self.generate_Motion_Primitives(current_ego_state, True)
        _, _ = self.calculate_cost(current_ego_state, list(cmds)[0], 0)

        # todo implement here some better planning
        rnd_acc = random.random() * self.params.param1
        rnd_ddelta = (random.random() - 0.5) * self.params.param1

        return VehicleCommands(acc=rnd_acc, ddelta=rnd_ddelta)

    def generate_Motion_Primitives(self, current_state: VehicleState, verbose=False):
        """
        This method is used to generate motion primitives for the current state of the ego vehicle
        """
        # Update linspace of velocities and actions
        # Generate trajector
        tr, cmds = self.mpg.generate(current_state)

        # Plot trajectories
        if verbose:
            plt.figure()
            for t in tr:
                x = np.array([v.x for v in t.values])
                y = np.array([v.y for v in t.values])
                plt.plot(x, y)
            plt.savefig("trajectories.png")

        return tr, cmds

    def calculate_cost(self, future_state: VehicleState, action: VehicleCommands, time: float):
        # pass
        score = 100
        vehicle_shapely = self.get_vehicle_shapely(future_state)

        # 0. Check whether future state is still within playground
        inside_playground = True
        # for obstacle in self.boundary_obstacles:
        #     if vehicle_shapely.intersects(obstacle):
        #         return float("inf"), False, False

        # 1. distance and heading wrt goal lane and whether it is a goal node
        inside_goal_lane = shapely.within(vehicle_shapely, self.goal.goal_polygon)

        state_se2transform = SE2Transform([future_state.x, future_state.y], future_state.psi)

        # lanelet_id = self.lanelet_network.find_lanelet_by_position([self.goal.ref_lane.control_points[1].q.p])[0][0]
        # lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id=lanelet_id)
        lanelet_new = DgLanelet(self.goal.ref_lane.control_points)
        lane_pose = lanelet_new.lane_pose_from_SE2Transform(state_se2transform)
        heading_delta = lane_pose.relative_heading

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
        # TODO

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

        return -score, is_goal_state, inside_playground

    def get_vehicle_shapely(self, state: VehicleState):
        cog = np.array([state.x, state.y])
        R = np.array([[np.cos(state.psi), -np.sin(state.psi)], [np.sin(state.psi), np.cos(state.psi)]])
        front_left = cog + R @ np.array([self.sg.lf, self.sg.w_half]).T
        front_right = cog + R @ np.array([self.sg.lf, -self.sg.w_half]).T
        back_left = cog + R @ np.array([self.sg.lr, self.sg.w_half]).T
        back_right = cog + R @ np.array([self.sg.lr, -self.sg.w_half]).T

        vehicle_shapely = shapely.Polygon((tuple(front_left), tuple(front_right), tuple(back_left), tuple(back_right)))

        return vehicle_shapely

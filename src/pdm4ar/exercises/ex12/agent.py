import random
from dataclasses import dataclass
from typing import Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters

from .motion_primitives import MotionPrimitivesGenerator, MPGParam
from dg_commons.dynamics.bicycle_dynamic import BicycleDynamics, VehicleState
from decimal import Decimal
from matplotlib import pyplot as plt
import numpy as np


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

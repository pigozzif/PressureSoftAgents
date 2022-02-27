import math
import random

import numpy as np
import torch

from es import OpenES, SimpleGA, CMAES, PEPG
from pressure import PressureSoftBody
from tensegrity import TensegritySoftBody
from voxel import VoxelSoftBody


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_solution(config):
    return np.random.random(config["n_params"])


def create_solver(config):
    name = config["solver"]
    n_params = config["n_params"]
    if name == "es":
        return OpenES(n_params, popsize=40, rank_fitness=False, forget_best=False)
    elif name == "ga":
        return SimpleGA(n_params, popsize=96)
    elif name == "cmaes":
        pop_size = 4 + math.floor(3 * math.log(n_params))
        return CMAES(n_params, sigma_init=0.5, popsize=pop_size + (config["np"] - pop_size % config["np"]))
    elif name == "pepg":
        return PEPG(n_params, forget_best=False)
    raise ValueError("Invalid solver name: {}".format(name))


def create_soft_body(config, pos, world):
    if config["body"] == "tensegrity":
        return TensegritySoftBody(world, pos[0], pos[1])
    elif config["body"] == "pressure":
        return PressureSoftBody(config, world, pos[0], pos[1], control_pressure=config["control_pressure"])
    elif config["body"] == "voxel":
        return VoxelSoftBody(world, pos[0], pos[1])
    raise ValueError("Invalid soft body name: {}".format(config["body"]))

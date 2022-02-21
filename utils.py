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


def random_solution(args):
    return np.random.random(args.n_params)


def create_solver(args, n_params):
    name = args.solver
    if name == "es":
        return OpenES(n_params, popsize=40, rank_fitness=False, forget_best=False)
    elif name == "ga":
        return SimpleGA(n_params, popsize=96)
    elif name == "cmaes":
        pop_size = 4 + math.floor(3 * math.log(n_params))
        return CMAES(n_params, sigma_init=0.5, popsize=pop_size + (args.np - pop_size % args.np))
    elif name == "pepg":
        return PEPG(n_params, forget_best=False)
    raise ValueError("Invalid solver name: {}".format(name))


def create_soft_body(args, config, pos, world):
    if args.body == "tensegrity":
        return TensegritySoftBody(world, pos[0], pos[1])
    elif args.body == "pressure":
        return PressureSoftBody(config, world, pos[0], pos[1], control_pressure=True)  # args.task == "escape")
    elif args.body == "voxel":
        return VoxelSoftBody(world, pos[0], pos[1])
    raise ValueError("Invalid soft body name: {}".format(args.body))

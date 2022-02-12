import argparse
import logging

import numpy as np

from agents import BaseController
from es import OpenES
from listener import FileListener
from simulation import simulation, parallel_solve, solve
from utils import set_seed


def parse_arguments():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--body", type=str, default="pressure", help="kind of soft body to simulate")
    parser.add_argument("--brain", type=str, default="phase", help="kind of controller to simulate")
    parser.add_argument("--task", type=str, default="flat", help="task to simulate")
    parser.add_argument("--timesteps", type=int, default=1800, help="number of time steps to simulate")
    parser.add_argument("--mode", default="opt-parallel", type=str, help="run mode")
    parser.add_argument("--iterations", default=750, type=int, help="solver iterations")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--np", type=int, default=5, help="number of parallel processes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    n_params = BaseController.get_number_of_params_for_controller(args.brain)
    if args.mode == "random":
        simulation(args, np.random.random(n_params), render=True)
    elif args.mode.startswith("opt"):
        listener = FileListener("metadata.txt")
        solver = OpenES(n_params,  # number of model parameters
                        sigma_init=0.5,  # initial standard deviation
                        sigma_decay=0.999,  # don't anneal standard deviation
                        learning_rate=0.1,  # learning rate for standard deviation
                        learning_rate_decay=1.0,  # annealing the learning rate
                        popsize=40,  # population size
                        antithetic=False,  # whether to use antithetic sampling
                        weight_decay=0.00,  # weight decay coefficient
                        rank_fitness=False,  # use rank rather than fitness numbers
                        forget_best=False)
        if args.mode.endswith("parallel"):
            best = parallel_solve(solver, args.iterations, args, listener)
        else:
            best = solve(solver, args.iterations, args, listener)
        logging.warning("fitness score at this local optimum: {}".format(best[1]))
    elif args.mode == "best":
        best = np.load("best.npy")
        simulation(args, best, render=True)
    raise ValueError("Invalid mode: {}".format(args.mode))

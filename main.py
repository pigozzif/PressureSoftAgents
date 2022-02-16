import argparse
import logging
import os

import numpy as np

from agents import BaseController
from listener import FileListener
from simulation import simulation, parallel_solve, solve
from utils import set_seed, create_solver


def parse_arguments():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--body", type=str, default="pressure", help="kind of soft body to simulate")
    parser.add_argument("--brain", type=str, default="mlp", help="kind of controller to simulate")
    parser.add_argument("--solver", type=str, default="ga", help="solver for optimization")
    parser.add_argument("--task", type=str, default="escape", help="task to simulate")
    parser.add_argument("--timesteps", type=int, default=1800, help="number of time steps to simulate")
    parser.add_argument("--mode", default="random", type=str, help="run mode")
    parser.add_argument("--evaluations", default=20000, type=int, help="number of solver evaluations")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--np", type=int, default=8, help="number of parallel processes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    n_params = BaseController.get_number_of_params_for_controller(args.brain)
    file_name = os.path.join("output", ".".join([args.solver, str(args.seed), args.task.split("-")[0], "txt"]))
    if args.mode == "random":
        simulation(args, np.random.random(n_params), render=True)
    elif args.mode.startswith("opt"):
        listener = FileListener(file_name, ["iteration", "evaluations", "best.fitness", "best.genotype"])
        solver = create_solver(args, n_params)
        if args.mode.endswith("parallel"):
            best = parallel_solve(solver, args.evaluations // solver.popsize, args, listener)
        else:
            best = solve(solver, args.evaluations // solver.popsize, args, listener)
        logging.warning("fitness score at this local optimum: {}".format(best[1]))
    elif args.mode == "best":
        best = list(map(lambda x: float(x), open(file_name, "r").readlines()[-1].strip().split(";")[-1].split(",")))
        print("fitness: {}".format(simulation(args, best, render=True)))
    else:
        raise ValueError("Invalid mode: {}".format(args.mode))

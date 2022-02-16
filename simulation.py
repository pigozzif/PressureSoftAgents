import logging
from multiprocessing import Pool

import numpy as np

from simulators import RenderSimulator, NoRenderSimulator


def solve(solver, iterations, args, listener):
    result = None
    for j in range(iterations):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        for i in range(solver.popsize):
            fitness_list[i] = simulation(args, solutions[i], render=False)
        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        if (j + 1) % 10 == 0:
            logging.warning("fitness at iteration {}: {}".format(j + 1, result[1]))
        listener.listen(**{"iteration": j, "evaluations": j * solver.popsize, "best.fitness": result[1], "best.genotype":
                        ",".join(list(map(lambda x: str(x), result[0])))})
    return result


def parallel_solve(solver, iterations, args, listener):
    num_workers = args.np
    if solver.popsize % num_workers != 0:
        raise RuntimeError("better to have n. workers divisor of pop size")
    result = None
    for j in range(iterations):
        solutions = solver.ask()
        with Pool(num_workers) as pool:
            results = pool.map(parallel_wrapper, [(args, solutions[i], i) for i in range(solver.popsize)])
        fitness_list = [value for _, value in sorted(results, key=lambda x: x[0])]
        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        if (j + 1) % 10 == 0:
            logging.warning("fitness at iteration {}: {}".format(j + 1, result[1]))
        listener.listen(**{"iteration": j, "evaluations": j * solver.popsize, "best.fitness": result[1], "best.genotype":
                        ",".join(list(map(lambda x: str(x), result[0])))})
    return result


def parallel_wrapper(args):
    arguments, solution, i = args
    fitness = simulation(arguments, solution, render=False)
    return i, fitness


def simulation(args, solution, render):
    if render:
        framework = RenderSimulator(args.body, args.brain, solution, args.task)
    else:
        framework = NoRenderSimulator(args.body, args.brain, solution, args.task)
    while framework.get_step_count() < args.timesteps:
        framework.step()
    framework.reset()
    return framework.get_reward() / (args.timesteps / 60.0)

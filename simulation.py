import logging
import math
import time
from multiprocessing import Pool

import numpy as np

from simulators import RenderSimulator, NoRenderSimulator


def parallel_solve(solver, iterations, config, listener):
    num_workers = config["np"]
    if solver.popsize % num_workers != 0:
        raise RuntimeError("better to have n. workers divisor of pop size")
    best_result = None
    best_fitness = float("-inf")
    start_time = time.time()
    for j in range(iterations):
        solutions = solver.ask()
        with Pool(num_workers) as pool:
            results = pool.map(parallel_wrapper, [(config, solutions[i], i) for i in range(solver.popsize)])
        fitness_list = [value for _, value in sorted(results, key=lambda x: x[0])]
        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        if (j + 1) % 10 == 0:
            logging.warning("fitness at iteration {}: {}".format(j + 1, result[1]))
        listener.listen(**{"iteration": j, "elapsed.sec": time.time() - start_time,
                           "evaluations": j * solver.popsize, "best.fitness": result[1]})
        if result[1] >= best_fitness or best_result is None:
            best_result = result[0]
            best_fitness = result[1]
            listener.save_best(best_result)
    return best_result, best_fitness


def parallel_wrapper(args):
    config, solution, i = args
    fitness = simulation(config, solution, render=False)
    return i, fitness


def simulation(config, solution, render):
    if render:
        framework = RenderSimulator(config, solution, save_video=int(config["save_video"]))
    else:
        framework = NoRenderSimulator(config, solution, save_video=int(config["save_video"]))
    while framework.should_step():
        framework.step()
    fitness = framework.env.get_fitness(framework.morphology, config["timesteps"])
    framework.reset()
    return fitness


def inflate_simulation(config, listener, render):
    solution = np.empty(0)
    if render:
        framework = RenderSimulator(config, solution, save_video=int(config["save_video"]))
    else:
        framework = NoRenderSimulator(config, solution, save_video=int(config["save_video"]))
    framework.morphology.pressure.min = 0
    framework.morphology.pressure.current = 0
    while framework.should_step():
        framework.step()
        area = framework.morphology.get_area()
        if framework.get_step_count() > 360:
            listener.listen(**{"t": framework.get_step_count(), "p": framework.morphology.pressure.current,
                               "a": area, "r": area / (config["r"] ** 2 * math.pi)})
    framework.reset()

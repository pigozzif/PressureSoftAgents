import logging
import time
from multiprocessing import Pool

from simulators import RenderSimulator, NoRenderSimulator


def parallel_solve(solver, iterations, args, config, listener):
    num_workers = args.np
    if solver.popsize % num_workers != 0:
        raise RuntimeError("better to have n. workers divisor of pop size")
    best_result = None
    best_fitness = float("-inf")
    start_time = time.time()
    for j in range(iterations):
        solutions = solver.ask()
        with Pool(num_workers) as pool:
            results = pool.map(parallel_wrapper, [(args, config, solutions[i], i) for i in range(solver.popsize)])
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
    arguments, config, solution, i = args
    fitness = simulation(arguments, config, solution, render=False)
    return i, fitness


def simulation(args, config, solution, render):
    if render:
        framework = RenderSimulator(args, config, solution)
    else:
        framework = NoRenderSimulator(args, config, solution)
    while framework.should_step():
        framework.step()
    return framework.env.get_fitness(framework.morphology, framework.get_step_count())

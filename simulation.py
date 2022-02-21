import logging
import time
from multiprocessing import Pool

import gym
import numpy as np
import torch.nn
from stable_baselines3 import PPO

from simulators import RenderSimulator, NoRenderSimulator
from utils import random_solution


def parallel_solve(solver, iterations, args, config, listener):
    num_workers = args.np
    if solver.popsize % num_workers != 0:
        raise RuntimeError("better to have n. workers divisor of pop size")
    result = None
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
    listener.save_best(result[0])
    return result


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


def rl_solve(args, config, listener):
    env = gym.make(id="RL-v0", args=args, config=config, solution=random_solution(args), listener=listener)
    model = PPO("MlpPolicy", env, policy_kwargs={"activation_fn": torch.nn.Tanh,
                                                 "net_arch": [{"pi": [],
                                                               "vf": []}]}, verbose=1)
    model.learn(total_timesteps=100)
    solution = np.empty(0)
    for k, v in model.get_parameters()["policy"].items():
        if k.startswith("action_net"):
            solution = np.append(solution, v.flatten().detach().numpy())
    listener.save_best(solution)

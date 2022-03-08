import abc
import logging
import math
import os
import shutil
import time
from multiprocessing import Pool

import gym
import numpy as np
from Box2D import b2World
from Box2D.examples.framework import FrameworkBase

from controllers import BaseController
from renderer import BaseRenderer
from soft_body import BaseSoftBody
from tasks import BaseEnv


class Environment(abc.ABC, FrameworkBase, gym.Env):

    def __init__(self, config, solution, render, save_video=False):
        FrameworkBase.__init__(self)
        self.config = config
        self.world = b2World(gravity=(0, -9.80665), doSleep=True)
        self.env = BaseEnv.create_env(self.config, self.world)
        self.morphology = BaseSoftBody.create_soft_body(self.config, self.env.get_initial_pos(), self.world)
        self.controller = BaseController.create_controller(self.config, self.morphology.get_input_dim(),
                                                           self.morphology.get_output_dim(), self.config["brain"],
                                                           solution)
        self.action_space = gym.spaces.Box(low=np.array([-1.0 for _ in range(self.morphology.get_output_dim())],
                                                        dtype=np.float32),
                                           high=np.array([1.0 for _ in range(self.morphology.get_output_dim())],
                                                         dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=np.array([0.0 for _ in range(self.morphology.get_input_dim())],
                                                             dtype=np.float32),
                                                high=np.array([1.0 for _ in range(self.morphology.get_input_dim())],
                                                              dtype=np.float32))
        self.renderer = None
        self.world.renderer = self.renderer
        self._renderer = BaseRenderer.create_renderer(render, save_video)

        if save_video:
            self.save_dir = os.path.join(os.getcwd(), "frames")
            if os.path.isdir(self.save_dir):
                shutil.rmtree(self.save_dir)
            os.mkdir(self.save_dir)
        else:
            self.save_dir = None

    def step(self, action):
        self.morphology.apply_control(action)
        self.SimulationLoop()
        return self.morphology.get_obs(), self.env.get_reward(self.morphology, self.stepCount), \
            not self.should_step(), {}

    def SimulationLoop(self):
        self.Step(self.settings)

    def Step(self, settings):
        FrameworkBase.Step(self, settings)
        self.morphology.physics_step()

    def reset(self):
        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None
        obs = self.morphology.get_obs()
        for body in self.world.bodies:
            for fixture in body.fixtures:
                body.DestroyFixture(fixture)
            self.world.DestroyBody(body)
        if self.save_dir is not None:
            self._save_video()
            shutil.rmtree(self.save_dir)
        return obs

    def render(self, mode="human"):
        if self.save_dir is not None:
            self._renderer.draw_image(self, os.path.join(self.save_dir, ".".join([str(self.stepCount), "png"])))
        else:
            self._renderer.draw_image(self)
            self._renderer.render()

    @staticmethod
    def _save_video():
        os.system("ffmpeg -r 60 -i ./frames/%d.png -vcodec mpeg4 -vf format=yuv420p -y video.mp4")

    def should_step(self):
        return self.stepCount < self.config["timesteps"] and self.env.should_step(self.morphology)

    def act(self, obs):
        return self.controller.control(self.stepCount, obs)


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
    env = Environment(config, solution, render, save_video=bool(int(config["save_video"])))
    obs = env.morphology.get_obs()
    done = False
    while not done:
        action = env.act(obs)
        obs, r, done, info = env.step(action)
        env.render()
    fitness = env.env.get_fitness(env.morphology, env.stepCount)
    env.reset()
    return fitness


def inflate_simulation(config, listener, render):
    solution = np.empty(0)
    env = Environment(config, solution, render, save_video=bool(int(config["save_video"])))
    env.morphology.pressure.min = 0
    env.morphology.pressure.current = 0
    obs = env.morphology.get_obs()
    done = False
    while not done:
        action = env.act(obs)
        obs, r, done, info = env.step(action)
        env.render()
        area = env.morphology.get_area()
        if env.stepCount > 360:
            listener.listen(**{"t": env.stepCount, "p": env.morphology.pressure.current,
                               "a": area, "r": area / (config["r"] ** 2 * math.pi)})
    env.reset()

import abc
import time
import os
import shutil

import gym
import numpy as np
import pygame
from Box2D.examples.framework import Framework
from Box2D.examples.framework import FrameworkBase

from controllers import BaseController
from tasks import BaseEnv
from utils import create_soft_body


class BaseSimulator(abc.ABC):

    def __init__(self, config, solution):
        self.config = config
        self.init_objects(solution)
        self.name = "{}-based Soft Agent".format(config["body"].capitalize())
        self.description = "Demonstration of a {}-based soft agent simulation.".format(config["body"])

    def init_objects(self, solution):
        self.env = BaseEnv.create_env(self.config, self.get_world())
        self.morphology = create_soft_body(self.config, self.env.get_initial_pos(), self.get_world())
        self.controller = BaseController.create_controller(self.morphology.get_input_dim(),
                                                           self.morphology.get_output_dim(), self.config["brain"],
                                                           solution)

    @abc.abstractmethod
    def get_world(self):
        pass

    @abc.abstractmethod
    def get_step_count(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    def reset(self):
        world = self.get_world()
        world.contactListener = None
        world.destructionListener = None
        world.renderer = None
        for body in world.bodies:
            for fixture in body.fixtures:
                body.DestroyFixture(fixture)
            world.DestroyBody(body)
#        self.init_objects(np.concatenate([param.view(-1).detach() for param in self.controller.get_params()]))
        return self.morphology.get_obs()

    def should_step(self):
        return self.get_step_count() < self.config["timesteps"]

    def act(self, t):
        obs = self.morphology.get_obs()
        control = self.controller.control(t, obs)
        self.morphology.apply_control(control)


class RenderSimulator(Framework, BaseSimulator):

    def __init__(self, config, solution, save_video=False):
        Framework.__init__(self)
        BaseSimulator.__init__(self, config, solution)
        self.gui_table.updateGUI(self.settings)
        self.clock = pygame.time.Clock()
        if save_video:
            self.save_dir = os.path.join(os.getcwd(), "frames")
            if os.path.isdir(self.save_dir):
                shutil.rmtree(self.save_dir)
            os.mkdir(self.save_dir)
        else:
            self.save_dir = None
        self.frame_rate = 3

    def get_world(self):
        return self.world

    def get_step_count(self):
        return self.stepCount

    def step(self):
        self.checkEvents()
        self.screen.fill((0, 0, 0))
        self.CheckKeys()
        self.SimulationLoop()
        if self.settings.drawMenu:
            self.gui_app.paint(self.screen)
        #self._scroll()
        pygame.display.flip()
        self.clock.tick(self.settings.hz)
        self.fps = self.clock.get_fps()
        if self.save_dir is not None and self.get_step_count() % self.frame_rate == 0:
            pygame.image.save(self.screen, os.path.join(self.save_dir, ".".join([str(self.get_step_count()), "jpg"])))

    def _scroll(self):
        self.viewCenter += self.morphology.get_center_of_mass() - self.viewCenter

    def Step(self, settings):
        settings.drawMenu = False
        settings.drawStats = False
        settings.drawFPS = False
        FrameworkBase.Step(self, settings)
        self.morphology.physics_step()
        self.act(self.stepCount)

    def reset(self):
        out = super(RenderSimulator, self).reset()
        if self.save_dir is not None:
            self._save_video()
            shutil.rmtree(self.save_dir)
        return out

    def _save_video(self):
        import imageio
        images = [imageio.imread(os.path.join(self.save_dir, img)) for img in sorted(os.listdir(self.save_dir))]
        imageio.mimsave("movie.gif", images)


class NoRenderSimulator(BaseSimulator, FrameworkBase):

    def __init__(self, config, solution):
        FrameworkBase.__init__(self)
        BaseSimulator.__init__(self, config, solution)
        self.renderer = None
        self.world.renderer = self.renderer
        self.groundbody = self.world.CreateBody()

    def get_world(self):
        return self.world

    def get_step_count(self):
        return self.stepCount

    def step(self):
        self.SimulationLoop()

    def SimulationLoop(self):
        self.Step(self.settings)

    def Step(self, settings):
        FrameworkBase.Step(self, settings)
        self.morphology.physics_step()
        self.act(self.stepCount)


class NoRenderRLSimulator(BaseSimulator, FrameworkBase, gym.Env):

    def __init__(self, config, solution, listener):
        FrameworkBase.__init__(self)
        BaseSimulator.__init__(self, config, solution)
        self.renderer = None
        self.world.renderer = self.renderer
        self.groundbody = self.world.CreateBody()
        self.action_space = gym.spaces.Box(low=np.array([-1.0 for _ in range(self.morphology.get_output_dim())],
                                                        dtype=np.float32),
                                           high=np.array([1.0 for _ in range(self.morphology.get_output_dim())],
                                                         dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=np.array([0.0 for _ in range(self.morphology.get_input_dim())],
                                                             dtype=np.float32),
                                                high=np.array([1.0 for _ in range(self.morphology.get_input_dim())],
                                                              dtype=np.float32))
        self.listener = listener
        self.start = time.time()

    def get_world(self):
        return self.world

    def get_step_count(self):
        return self.stepCount

    def step(self, action):
        self.SimulationLoop()
        self.morphology.apply_control(action)
        obs = self.morphology.get_obs()
        reward = self.env.get_reward(self.morphology, self.get_step_count())
        return obs, reward, False, {}

    def render(self, mode="human"):
        pass

    def SimulationLoop(self):
        self.Step(self.settings)

    def Step(self, settings):
        FrameworkBase.Step(self, settings)
        self.morphology.physics_step()


gym.envs.registration.register(id="RL-v0", entry_point="simulators:NoRenderRLSimulator", max_episode_steps=600)

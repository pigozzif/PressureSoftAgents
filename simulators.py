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

    def __init__(self, config, solution, save_video=False):
        self.config = config
        self.init_objects(solution)
        self.name = "{}-based Soft Agent".format(config["body"].capitalize())
        self.description = "Demonstration of a {}-based soft agent simulation.".format(config["body"])
        if save_video:
            self.save_dir = os.path.join(os.getcwd(), "frames")
            if os.path.isdir(self.save_dir):
                shutil.rmtree(self.save_dir)
            os.mkdir(self.save_dir)
            pygame.init()
            self.screen = pygame.Surface((750, 750))
            self.screen.fill((0, 0, 0))
        else:
            self.save_dir = None
        self.magnify = 15

    def init_objects(self, solution):
        self.env = BaseEnv.create_env(self.config, self.get_world())
        self.morphology = create_soft_body(self.config, self.env.get_initial_pos(), self.get_world())
        self.controller = BaseController.create_controller(self.config, self.morphology.get_input_dim(),
                                                           self.morphology.get_output_dim(), self.config["brain"],
                                                           solution)

    @abc.abstractmethod
    def get_world(self):
        pass

    @abc.abstractmethod
    def get_step_count(self):
        pass

    def step(self):
        self.inner_step()
        if self.save_dir is not None:
            self._draw_image(os.path.join(self.save_dir, ".".join([str(self.get_step_count()), "png"])))

    @abc.abstractmethod
    def inner_step(self):
        pass

    def reset(self):
        world = self.get_world()
        world.contactListener = None
        world.destructionListener = None
        world.renderer = None
        obs = self.morphology.get_obs()
        for body in world.bodies:
            for fixture in body.fixtures:
                body.DestroyFixture(fixture)
            world.DestroyBody(body)
        if self.save_dir is not None:
            self._save_video()
            shutil.rmtree(self.save_dir)
        return obs

    def _draw_image(self, file_name):
        w, h = self.screen.get_width(), self.screen.get_height()
        center_x, center_y = self.morphology.get_center_of_mass()
        for body in self.env.bodies:
            vertices = body.fixtures[0].shape.vertices
            l_x, l_y, r_x, r_y = (vertices[0][0] - center_x) * self.magnify + w / 2, \
                                 (vertices[0][1] - center_y) * self.magnify + h / 2, \
                                 (vertices[1][0] - center_x) * self.magnify + w / 2, \
                                 (vertices[1][1] - center_y) * self.magnify + h / 2
            pygame.draw.lines(self.screen, (0, 0, 255), False, [(l_x, h - l_y), (r_x, h - r_y)], 5)
        for mass in self.morphology.masses:
            shape = mass.fixtures[0].shape
            half_width = abs(shape.vertices[0][0] - shape.vertices[1][0]) / 2
            half_height = abs(shape.vertices[0][1] - shape.vertices[2][1]) / 2
            cx, cy = mass.position.x, mass.position.y
            vertices = [(cx - half_width, cy - half_height), (cx + half_width, cy - half_height),
                        (cx + half_width, cy + half_height), (cx - half_width, cy + half_height)]
            new_vertices = [
                ((x - center_x) * self.magnify + w / 2, h - ((y - center_y) * self.magnify + h / 2)) for
                x, y in vertices]
            pygame.draw.rect(self.screen, (255, 0, 0),
                             (new_vertices[3][0], new_vertices[3][1],
                              half_width * 2 * self.magnify, half_height * 2 * self.magnify), 0)
            pygame.draw.rect(self.screen, (219, 112, 147),
                             (new_vertices[3][0], new_vertices[3][1],
                              half_width * 2 * self.magnify, half_height * 2 * self.magnify), 2)
        for joint in self.morphology.joints:
            l_x, l_y, r_x, r_y = (joint.anchorA.x - center_x) * self.magnify + w / 2, \
                                 (joint.anchorA.y - center_y) * self.magnify + h / 2, \
                                 (joint.anchorB.x - center_x) * self.magnify + w / 2, \
                                 (joint.anchorB.y - center_y) * self.magnify + h / 2
            pygame.draw.lines(self.screen, (255, 255, 255), False, [(l_x, h - l_y), (r_x, h - r_y)], 3)
        pygame.image.save(self.screen, file_name)
        self.screen.fill((0, 0, 0))

    def _save_video(self):
        os.system("ffmpeg -r 60 -i ./frames/%d.png -vcodec mpeg4 -vf format=yuv420p -y escape1.mp4")

    def should_step(self):
        return self.get_step_count() < self.config["timesteps"] and self.env.should_step(self.morphology)

    def act(self, t):
        obs = self.morphology.get_obs()
        control = self.controller.control(t, obs)
        self.morphology.apply_control(control)


class RenderSimulator(Framework, BaseSimulator):

    def __init__(self, config, solution, save_video=False):
        Framework.__init__(self)
        BaseSimulator.__init__(self, config, solution, save_video)
        self.gui_table.updateGUI(self.settings)
        self.clock = pygame.time.Clock()

    def get_world(self):
        return self.world

    def get_step_count(self):
        return self.stepCount

    def inner_step(self):
        self.checkEvents()
        self.screen.fill((0, 0, 0))
        self.CheckKeys()
        self.SimulationLoop()
        if self.settings.drawMenu:
            self.gui_app.paint(self.screen)
        pygame.display.flip()
        self.clock.tick(self.settings.hz)
        self.fps = self.clock.get_fps()

    def Step(self, settings):
        settings.drawMenu = False
        settings.drawStats = False
        settings.drawFPS = False
        FrameworkBase.Step(self, settings)
        self.morphology.physics_step()
        self.act(self.stepCount)


class NoRenderSimulator(BaseSimulator, FrameworkBase):

    def __init__(self, config, solution, save_video=False):
        FrameworkBase.__init__(self)
        BaseSimulator.__init__(self, config, solution, save_video)
        self.renderer = None
        self.world.renderer = self.renderer
        self.groundbody = self.world.CreateBody()

    def get_world(self):
        return self.world

    def get_step_count(self):
        return self.stepCount

    def inner_step(self):
        self.SimulationLoop()

    def SimulationLoop(self):
        self.Step(self.settings)

    def Step(self, settings):
        FrameworkBase.Step(self, settings)
        self.morphology.physics_step()
        self.act(self.stepCount)


class NoRenderRLSimulator(BaseSimulator, FrameworkBase, gym.Env):

    def __init__(self, config, solution, listener, save_video=False):
        FrameworkBase.__init__(self)
        BaseSimulator.__init__(self, config, solution, save_video)
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

    def inner_step(self, action):
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

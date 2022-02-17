import abc

import pygame
from Box2D.examples.framework import Framework
from Box2D.examples.framework import FrameworkBase

from controllers import BaseController
from tasks import BaseEnv
from utils import create_soft_body


class BaseSimulator(abc.ABC):

    def __init__(self, args, solution):
        self.env = BaseEnv.create_env(args, self.get_world())
        self.morphology = create_soft_body(args, self.env.get_initial_pos(), self.get_world())
        self.controller = BaseController.create_controller(self.morphology.get_input_dim(),
                                                           self.morphology.get_output_dim(), args.brain, solution)
        self.name = "{}-based Soft Body".format(args.body.capitalize())
        self.description = "Demonstration of a {}-based soft body simulation.".format(args.body)

    @abc.abstractmethod
    def get_world(self):
        pass

    @abc.abstractmethod
    def get_step_count(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    def act(self, t):
        obs = self.morphology.get_obs()
        control = self.controller.control(t, obs)
        self.morphology.apply_control(control)


class RenderSimulator(Framework, BaseSimulator):

    def __init__(self, args, solution):
        Framework.__init__(self)
        BaseSimulator.__init__(self, args, solution)
        self.gui_table.updateGUI(self.settings)
        self.clock = pygame.time.Clock()

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
        pygame.display.flip()
        self.clock.tick(self.settings.hz)
        self.fps = self.clock.get_fps()

    def reset(self):
        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None

    def Step(self, settings):
        FrameworkBase.Step(self, settings)
        self.morphology.physics_step()
        self.act(self.stepCount)


class NoRenderSimulator(BaseSimulator, FrameworkBase):

    def __init__(self, args, solution):
        FrameworkBase.__init__(self)
        BaseSimulator.__init__(self, args, solution)
        self.renderer = None
        self.world.renderer = self.renderer
        self.groundbody = self.world.CreateBody()

    def get_world(self):
        return self.world

    def get_step_count(self):
        return self.stepCount

    def step(self):
        self.SimulationLoop()

    def reset(self):
        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None

    def SimulationLoop(self):
        self.Step(self.settings)

    def Step(self, settings):
        FrameworkBase.Step(self, settings)
        self.morphology.physics_step()
        self.act(self.stepCount)

import abc

import pygame
from Box2D.examples.framework import Framework
from Box2D.examples.framework import FrameworkBase

from agents import Agent
from tasks import BaseEnv
from utils import create_task


class BaseSimulator(abc.ABC):

    def __init__(self, args, solution):
        self.agent = Agent.create_agent(args.body, args.brain, solution, self.get_world())
        self.env = BaseEnv.create_env(args, self.get_world())
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
        self.agent.morphology.physics_step()
        self.agent.act(self.stepCount)


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
        self.agent.morphology.physics_step()
        self.agent.act(self.stepCount)

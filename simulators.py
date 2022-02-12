import abc

import pygame
from Box2D import b2World
from Box2D.examples.framework import Framework
from Box2D.examples.settings import fwSettings

from agents import Agent
from utils import create_task


class BaseSimulator(abc.ABC):

    def __init__(self, soft_body_name, controller_name, solution, task_name):
        self.agent = Agent.create_agent(soft_body_name, controller_name, solution, self.get_world())
        create_task(self.get_world(), task_name)

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

    def get_reward(self):
        return self.agent.morphology.get_center_of_mass()[0]


class RenderSimulator(Framework, BaseSimulator):

    def __init__(self, soft_body_name, controller_name, solution, task_name):
        Framework.__init__(self)
        BaseSimulator.__init__(self, soft_body_name, controller_name, solution, task_name)
        self.name = self.agent.morphology.name
        self.description = self.agent.morphology.description
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
        Framework.Step(self, settings)
        self.agent.morphology.physics_step()
        self.agent.act(self.stepCount)


class NoRenderSimulator(BaseSimulator):

    def __init__(self, soft_body_name, controller_name, solution, task_name):
        self.world = b2World(gravity=(0, -10), doSleep=True)
        super(NoRenderSimulator, self).__init__(soft_body_name, controller_name, solution, task_name)
        self.time_step = 1.0 / 60.0
        self.steps = 0

    def get_world(self):
        return self.world

    def get_step_count(self):
        return self.steps

    def step(self):
        self.world.Step(self.time_step, fwSettings.velocityIterations,
                        fwSettings.positionIterations)
        self.agent.morphology.physics_step()
        self.agent.act(self.steps)
        self.world.ClearForces()
        self.steps += 1

    def reset(self):
        pass

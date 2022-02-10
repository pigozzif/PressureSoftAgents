import abc

import pygame
from Box2D import b2World
from Box2D.examples.framework import Framework
from Box2D.examples.settings import fwSettings

from src.utils import create_soft_body, create_task


class AbstractFramework(abc.ABC):

    def __init__(self, soft_body_name, task_name):
        self.soft_body = create_soft_body(soft_body_name, self.get_world())
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
        return self.soft_body.get_center_of_mass()[0]


class RenderFramework(Framework, AbstractFramework):

    def __init__(self, soft_body_name, task_name):
        Framework.__init__(self)
        AbstractFramework.__init__(self, soft_body_name=soft_body_name, task_name=task_name)
        self.name = self.soft_body.name
        self.description = self.soft_body.description
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
        self.soft_body.physics_step()
        obs = self.soft_body.sense()


class NoRenderFramework(AbstractFramework):

    def __init__(self, soft_body_name, task_name):
        super(NoRenderFramework, self).__init__(soft_body_name, task_name)
        self.world = b2World(gravity=(0, -10), doSleep=True)
        self.time_step = 1.0 / 60.0
        self.steps = 0

    def get_world(self):
        return self.world

    def get_step_count(self):
        return self.steps

    def step(self):
        self.world.Step(self.time_step, fwSettings.velocityIterations,
                        fwSettings.positionIterations)
        obs = self.soft_body.sense()
        self.soft_body.physics_step()
        self.world.ClearForces()
        self.steps += 1

    def reset(self):
        pass

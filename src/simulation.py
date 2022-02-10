import pygame
import argparse

from Box2D import b2World
from Box2D.examples.framework import Framework
from Box2D.examples.settings import fwSettings

from src.utils import create_soft_body, create_task


def render_simulation(args):
    framework = SoftBodyFramework(args.body, args.task)
    framework.gui_table.updateGUI(framework.settings)
    running = True
    clock = pygame.time.Clock()
    while running and framework.stepCount < args.timesteps:
        framework.checkEvents()
        framework.screen.fill((0, 0, 0))
        framework.CheckKeys()
        framework.SimulationLoop()
        if framework.settings.drawMenu:
            framework.gui_app.paint(framework.screen)
        pygame.display.flip()
        clock.tick(framework.settings.hz)
        framework.fps = clock.get_fps()
    framework.world.contactListener = None
    framework.world.destructionListener = None
    framework.world.renderer = None


def no_render_simulation(args):
    running = True
    step_count = 0
    time_step = 1.0 / 60.0
    world = b2World(gravity=(0, -10), doSleep=True)
    soft_body = create_soft_body(args.body, world)
    create_task(world, args.task)
    while running and step_count < args.timesteps:
        world.Step(time_step, fwSettings.velocityIterations,
                   fwSettings.positionIterations)
        soft_body.physics_step()
        world.ClearForces()
        step_count += 1


class SoftBodyFramework(Framework):

    def __init__(self, soft_body_name, task_name):
        super(SoftBodyFramework, self).__init__()
        self.soft_body = create_soft_body(soft_body_name, self.world)
        create_task(self.world, task_name)
        self.name = self.soft_body.name
        self.description = self.soft_body.description

    def Step(self, settings):
        Framework.Step(self, settings)
        self.soft_body.physics_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--body", type=str, default="pressure", help="kind of soft body to simulate")
    parser.add_argument("--task", type=str, default="flat", help="task to simulate")
    parser.add_argument("--timesteps", type=int, default=900, help="number of time steps to simulate")
    parser.add_argument("--render", type=bool, default=True, help="render simulation on screen")
    args = parser.parse_args()
    if args.render:
        render_simulation(args)
    else:
        no_render_simulation(args)

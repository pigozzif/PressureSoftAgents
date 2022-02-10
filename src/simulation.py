import math

import pygame
import argparse

from Box2D import b2EdgeShape, b2FixtureDef, b2PolygonShape
from Box2D.examples.framework import Framework

from src.utils import create_soft_body


def simulation(args, render):
    framework = SoftBodyFramework(args.body, args.task)
    if render:
        framework.gui_table.updateGUI(framework.settings)
    running = True
    clock = pygame.time.Clock()
    while running:
        if render:
            framework.checkEvents()
        framework.screen.fill((0, 0, 0))
        if render:
            framework.CheckKeys()
        framework.SimulationLoop()
        if render and framework.settings.drawMenu:
            framework.gui_app.paint(framework.screen)
        pygame.display.flip()
        clock.tick(framework.settings.hz)
        framework.fps = clock.get_fps()
    framework.world.contactListener = None
    framework.world.destructionListener = None
    framework.world.renderer = None


class SoftBodyFramework(Framework):

    def __init__(self, soft_body_name, task_name):
        super(SoftBodyFramework, self).__init__()
        self.soft_body = create_soft_body(soft_body_name, self.world)
        self.create_task(task_name)
        self.name = self.soft_body.name
        self.description = self.soft_body.description

    def Step(self, settings):
        Framework.Step(self, settings)
        self.soft_body.physics_step()

    def create_task(self, task_name):
        if task_name == "flat":
            self.world.CreateBody(
                shapes=b2EdgeShape(vertices=[(-20, -10), (1000, -10)])
            )
            self.world.CreateBody(
                shapes=b2EdgeShape(vertices=[(-20, 100), (-20, -100)])
            )
            return
        elif task_name == "obstacles":
            self.world.CreateBody(
                shapes=b2EdgeShape(vertices=[(-100, -100), (100, -100)])
            )
            box1 = self.world.CreateStaticBody(
                position=(0, -15),
                allowSleep=True,
                fixtures=b2FixtureDef(friction=0.8,
                                      shape=b2PolygonShape(box=(25.0, 2.5)),
                                      ))
            box1.fixedRotation = True
            box1.angle = -25 * math.pi / 180.0

            box2 = self.world.CreateStaticBody(
                position=(55, -45),
                allowSleep=True,
                fixtures=b2FixtureDef(friction=0.8,
                                      shape=b2PolygonShape(box=(20.0, 2.5)),
                                      ))
            box2.fixedRotation = True
            box2.angle = 45 * math.pi / 180.0

            box3 = self.world.CreateStaticBody(
                position=(0, -100),
                allowSleep=True,
                fixtures=b2FixtureDef(friction=0.8,
                                      shape=b2PolygonShape(vertices=[(-50, 0.0),
                                                                     (0, 0.0),
                                                                     (-45, 20),
                                                                     ]
                                                           )))
            box3.fixedRotation = True
            return
        raise ValueError("Invalid task name: {}".format(task_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--body", type=str, default="pressure", help="kind of soft body to simulate")
    parser.add_argument("--task", type=str, default="flat", help="task to simulate")
    args = parser.parse_args()
    simulation(args, True)

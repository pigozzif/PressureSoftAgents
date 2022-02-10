import pygame
import argparse

from Box2D.examples.framework import Framework

from src.utils import create_soft_body


def simulation(name, render):
    framework = SoftBodyFramework(name, -100, 100)
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

    def __init__(self, soft_body_name, min_x, max_x):
        super(SoftBodyFramework, self).__init__()
        self.soft_body = create_soft_body(soft_body_name, self.world, min_x, max_x)
        self.soft_body.create_test_obstacles()
        self.name = self.soft_body.name
        self.description = self.soft_body.description

    def Step(self, settings):
        Framework.Step(self, settings)
        self.soft_body.physics_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--body", type=str, default="pressure", help="kind of soft body to simulate")
    args = parser.parse_args()
    simulation(args.body, True)

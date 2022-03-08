import abc

import pygame.time
from Box2D.examples.settings import fwSettings


class BaseRenderer(abc.ABC):

    @abc.abstractmethod
    def draw_image(self, env, file_name=None):
        pass

    @abc.abstractmethod
    def render(self):
        pass

    @classmethod
    def create_renderer(cls, render, save_video):
        return PygameRenderer(750, save_video) if render or save_video else NoneRenderer()


class NoneRenderer(BaseRenderer):

    def draw_image(self, env, file_name=None):
        return

    def render(self):
        return


class PygameRenderer(BaseRenderer):

    def __init__(self, size, save_video):
        self.clock = pygame.time.Clock()
        pygame.init()
        if save_video:
            self.screen = pygame.Surface((size, size))
        else:
            self.screen = pygame.display.set_mode((size, size))
        self.screen.fill((0, 0, 0))
        self.magnify = size / 50

    def draw_image(self, env, file_name=None):
        w, h = self.screen.get_size()
        center_x, center_y = env.morphology.get_center_of_mass()
        for body in env.env.bodies:
            vertices = body.fixtures[0].shape.vertices
            l_x, l_y, r_x, r_y = (vertices[0][0] - center_x) * self.magnify + w / 2, \
                                 (vertices[0][1] - center_y) * self.magnify + h / 2, \
                                 (vertices[1][0] - center_x) * self.magnify + w / 2, \
                                 (vertices[1][1] - center_y) * self.magnify + h / 2
            pygame.draw.lines(self.screen, (0, 0, 255), False, [(l_x, h - l_y), (r_x, h - r_y)], 5)
        for mass in env.morphology.masses:
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
        for joint in env.morphology.joints:
            l_x, l_y, r_x, r_y = (joint.anchorA.x - center_x) * self.magnify + w / 2, \
                                 (joint.anchorA.y - center_y) * self.magnify + h / 2, \
                                 (joint.anchorB.x - center_x) * self.magnify + w / 2, \
                                 (joint.anchorB.y - center_y) * self.magnify + h / 2
            pygame.draw.lines(self.screen, (255, 255, 255), False, [(l_x, h - l_y), (r_x, h - r_y)], 3)
        if file_name is not None:
            pygame.image.save(self.screen, file_name)
        self.screen.fill((0, 0, 0))

    def render(self):
        pygame.display.flip()
        self.clock.tick(fwSettings.hz)

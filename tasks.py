import abc
import os
import random

import pygame
from Box2D import b2EdgeShape, b2FixtureDef, b2PolygonShape


class BaseEnv(abc.ABC):

    def __init__(self, world):
        self.world = world
        self.bodies = []

    def should_step(self, morphology):
        return True

    @abc.abstractmethod
    def init_env(self):
        pass

    @abc.abstractmethod
    def get_initial_pos(self):
        pass

    @abc.abstractmethod
    def get_reward(self, morphology, t):
        pass

    @abc.abstractmethod
    def get_fitness(self, morphology, t):
        pass

    @abc.abstractmethod
    def draw_env(self, w, h, center, screen, magnify):
        pass

    @classmethod
    def create_env(cls, config, world):
        name = config["task"]
        if name == "flat":
            env = FlatLocomotion(world, config)
        elif name.startswith("hilly"):
            env = HillyLocomotion(world, config)
        elif name == "escape":
            env = Escape(world, config)
        elif name == "climber":
            env = Climber(world, config)
        elif name == "cave":
            env = CaveCrawler(world, config)
        elif name == "carrier":
            env = Carrier(world, config)
        else:
            raise ValueError("Invalid task name: {}".format(config["task"]))
        env.init_env()
        return env


class FlatLocomotion(BaseEnv):

    def __init__(self, world, config):
        BaseEnv.__init__(self, world)
        self.r = config["r"]
        self.prev_pos = self.get_initial_pos()[0]

    def init_env(self):
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-500, 0), (500, 0)]),
        )
        self.bodies.append(ground)

    def get_initial_pos(self):
        return self.r, self.r + 1

    def get_reward(self, morphology, t):
        pos = morphology.get_center_of_mass()[0]
        r = pos - self.prev_pos
        self.prev_pos = pos
        return r

    def get_fitness(self, morphology, t):
        return (morphology.get_center_of_mass()[0] - self.get_initial_pos()[0]) / (t / 60.0)

    def draw_env(self, w, h, center, screen, magnify):
        center_x, center_y = center
        for body in self.bodies:
            vertices = body.fixtures[0].shape.vertices
            l_x, l_y, r_x, r_y = (vertices[0][0] - center_x) * magnify + w / 2, \
                                 (vertices[0][1] - center_y) * magnify + h / 2, \
                                 (vertices[1][0] - center_x) * magnify + w / 2, \
                                 (vertices[1][1] - center_y) * magnify + h / 2
            pygame.draw.lines(screen, (0, 0, 255), False, [(l_x, h - l_y), (r_x, h - r_y)], 5)


class HillyLocomotion(BaseEnv):

    def __init__(self, world, config):
        BaseEnv.__init__(self, world)
        self.h = int(config["task"].split("-")[1])
        self.w = int(config["task"].split("-")[2])
        self.r = config["r"]
        if not os.path.isdir(os.path.join(os.getcwd(), "terrains")):
            os.mkdir(os.path.join(os.getcwd(), "terrains"))
        self.file_name = os.path.join(os.getcwd(), "terrains", ".".join(["hilly", str(config["seed"]), "txt"]))
        self.prev_pos = self.get_initial_pos()[0]

    def init_env(self):
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, 100), (-20, -100)])
        )
        if not os.path.isfile(self.file_name):
            self._write_terrain()
        start = self._read_terrain()
        wall = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, 100), (start, -100)])
        )
        self.bodies.append(ground)
        self.bodies.append(wall)

    def _read_terrain(self):
        with open(self.file_name, "r") as file:
            file.readline()
            for line in file:
                start, end, height, prev_height = tuple(map(lambda x: float(x), line.split(";")))
                half_x, y = (end - start) / 2, 0
                bump = self.world.CreateBody(
                    shapes=b2EdgeShape(vertices=[(end, y + height), (start, y + prev_height)])
                )
                self.bodies.append(bump)
        return start

    def _write_terrain(self):
        width = 400
        content = ";".join(["start", "end", "height", "prev_height"]) + "\n"
        start = -20
        end = start + max(random.gauss(1, 0.25) * self.w, 1.0)
        prev_height = abs(random.gauss(0, self.h))
        height = abs(random.gauss(0, self.h))
        while end < width:
            content += ";".join([str(start), str(end), str(height), str(prev_height)]) + "\n"
            start = end
            prev_height = height
            end += max(random.gauss(1, 0.25) * self.w, 1.0)
            height = abs(random.gauss(0, self.h))
        with open(self.file_name, "w") as file:
            file.write(content)

    def get_initial_pos(self):
        return 0, self.r * 1.5

    def get_reward(self, morphology, t):
        pos = morphology.get_center_of_mass()[0]
        r = pos - self.prev_pos
        self.prev_pos = pos
        return r

    def get_fitness(self, morphology, t):
        return (morphology.get_center_of_mass()[0] - self.get_initial_pos()[0]) / (t / 60.0)

    def draw_env(self, w, h, center, screen, magnify):
        center_x, center_y = center
        for body in self.bodies:
            vertices = body.fixtures[0].shape.vertices
            l_x, l_y, r_x, r_y = (vertices[0][0] - center_x) * magnify + w / 2, \
                                 (vertices[0][1] - center_y) * magnify + h / 2, \
                                 (vertices[1][0] - center_x) * magnify + w / 2, \
                                 (vertices[1][1] - center_y) * magnify + h / 2
            pygame.draw.lines(screen, (0, 0, 255), False, [(l_x, h - l_y), (r_x, h - r_y)], 5)


class Escape(BaseEnv):

    def __init__(self, world, config):
        BaseEnv.__init__(self, world)
        self.side = config["r"] * 3
        self.prev_pos = self.get_initial_pos()[0]

    def should_step(self, morphology):
        return any([abs(mass.position.x) <= self.side / 2 + 1 for mass in morphology.masses])

    def init_env(self):
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-100, 0), (100, 0)])
        )
        self.bodies.append(ground)
        roof1 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(- self.side / 2, self.side), (self.side / 2, self.side)])
        )
        self.bodies.append(roof1)
        roof2 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(- self.side / 2 - 1, self.side + 1), (self.side / 2 + 1, self.side + 1)])
        )
        self.bodies.append(roof2)
        aperture = self.side / 3
        wall1 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(- self.side / 2, self.side), (- self.side / 2, aperture)])
        )
        self.bodies.append(wall1)
        wall2 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(- self.side / 2 - 1, self.side + 1), (- self.side / 2 - 1, aperture)])
        )
        self.bodies.append(wall2)
        wall3 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(self.side / 2, self.side), (self.side / 2, aperture)])
        )
        self.bodies.append(wall3)
        wall4 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(self.side / 2 + 1, self.side + 1), (self.side / 2 + 1, aperture)])
        )
        self.bodies.append(wall4)
        self.bodies.append(wall2)
        wall5 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(- self.side / 2 - 1, aperture), (- self.side / 2, aperture)])
        )
        self.bodies.append(wall5)
        wall6 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(self.side / 2, aperture), (self.side / 2 + 1, aperture)])
        )
        self.bodies.append(wall6)

    def get_initial_pos(self):
        return 0, self.side / 2

    def get_reward(self, morphology, t):
        pos = morphology.get_center_of_mass()[0]
        r = abs(pos - self.prev_pos)
        self.prev_pos = pos
        return r

    def get_fitness(self, morphology, t):
        return abs(morphology.get_center_of_mass()[0]) - self.get_initial_pos()[0]

    def draw_env(self, w, h, center, screen, magnify):
        center_x, center_y = center
        for body in self.bodies:
            vertices = body.fixtures[0].shape.vertices
            l_x, l_y, r_x, r_y = (vertices[0][0] - center_x) * magnify + w / 2, \
                                 (vertices[0][1] - center_y) * magnify + h / 2, \
                                 (vertices[1][0] - center_x) * magnify + w / 2, \
                                 (vertices[1][1] - center_y) * magnify + h / 2
            pygame.draw.lines(screen, (0, 0, 255), False, [(l_x, h - l_y), (r_x, h - r_y)], 5)


class Climber(BaseEnv):

    def __init__(self, world, config):
        BaseEnv.__init__(self, world)
        self.r = config["r"]
        self.prev_pos = self.get_initial_pos()[1]

    def init_env(self):
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-100, 0), (100, 0)])
        )
        self.bodies.append(ground)
        side = 6
        height = 100
        wall1 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(- side, 0), (- side, height)])
        )
        self.bodies.append(wall1)
        wall2 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(side, 0), (side, height)])
        )
        self.bodies.append(wall2)

    def get_initial_pos(self):
        return 0, self.r + 1

    def get_reward(self, morphology, t):
        pos = morphology.get_center_of_mass()[1]
        r = pos - self.prev_pos
        self.prev_pos = pos
        return r

    def get_fitness(self, morphology, t):
        return morphology.get_center_of_mass()[1] - self.get_initial_pos()[1]

    def draw_env(self, w, h, center, screen, magnify):
        center_x, center_y = center
        for body in self.bodies:
            vertices = body.fixtures[0].shape.vertices
            l_x, l_y, r_x, r_y = (vertices[0][0] - center_x) * magnify + w / 2, \
                                 (vertices[0][1] - center_y) * magnify + h / 2, \
                                 (vertices[1][0] - center_x) * magnify + w / 2, \
                                 (vertices[1][1] - center_y) * magnify + h / 2
            pygame.draw.lines(screen, (0, 0, 255), False, [(l_x, h - l_y), (r_x, h - r_y)], 5)


class CaveCrawler(BaseEnv):

    def __init__(self, world, config):
        BaseEnv.__init__(self, world)
        self.r = config["r"]
        self.prev_pos = self.get_initial_pos()[0]

    def init_env(self):
        wall1 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(- self.r * 2, 0), (- self.r * 2, self.r * 2.5)])
        )
        self.bodies.append(wall1)
        small_step = self.r
        large_step = small_step * 2.5
        start = - self.r * 2
        roof1 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2.5), (start + large_step, self.r * 2.5)])
        )
        self.bodies.append(roof1)
        start += large_step
        roof2 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2.5), (start, self.r * 2)])
        )
        self.bodies.append(roof2)
        roof3 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2), (start + small_step, self.r * 2)])
        )
        self.bodies.append(roof3)
        start += small_step
        roof4 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2), (start, self.r * 1.25)])
        )
        self.bodies.append(roof4)
        roof5 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 1.25), (start + large_step / 2, self.r * 1.25)])
        )
        self.bodies.append(roof5)
        start += large_step / 2
        roof6 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 1.25), (start, self.r * 2)])
        )
        self.bodies.append(roof6)
        roof7 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2), (start + small_step, self.r * 2)])
        )
        self.bodies.append(roof7)
        start += small_step
        roof8 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2), (start, self.r * 2.5)])
        )
        self.bodies.append(roof8)
        roof9 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2.5), (start + small_step, self.r * 2.5)])
        )
        self.bodies.append(roof9)
        start += small_step
        roof10 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2.5), (start, self.r * 2)])
        )
        self.bodies.append(roof10)
        roof11 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2), (start + small_step, self.r * 2)])
        )
        self.bodies.append(roof11)
        start += small_step
        roof12 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2), (start, self.r * 1.25)])
        )
        self.bodies.append(roof12)
        roof13 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 1.25), (start + large_step / 2, self.r * 1.25)])
        )
        self.bodies.append(roof13)
        start += large_step / 2
        roof14 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 1.25), (start, self.r * 2)])
        )
        self.bodies.append(roof14)
        roof15 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2), (start + small_step, self.r * 2)])
        )
        start += small_step
        self.bodies.append(roof15)
        roof16 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2), (start, self.r * 2.5)])
        )
        self.bodies.append(roof16)
        roof17 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2.5), (start + large_step / 2, self.r * 2.5)])
        )
        self.bodies.append(roof17)
        start += large_step / 2
        roof18 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2.5), (start, self.r * 1.25)])
        )
        self.bodies.append(roof18)
        roof19 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 1.25), (start + small_step, self.r)])
        )
        self.bodies.append(roof19)
        start += small_step
        roof20 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r), (start, self.r * 2.5)])
        )
        self.bodies.append(roof20)
        roof21 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 2.5), (start + large_step * 0.75, self.r * 2.5)])
        )
        self.bodies.append(roof21)

        start = - self.r * 2
        ground1 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, 0), (start + large_step * 2, 0)])
        )
        self.bodies.append(ground1)
        start += large_step * 2
        ground2 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, 0), (start, self.r * 0.25)])
        )
        self.bodies.append(ground2)
        ground3 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 0.25), (start + large_step / 2, self.r * 0.25)])
        )
        self.bodies.append(ground3)
        start += large_step / 2
        ground4 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, self.r * 0.25), (start, 0)])
        )
        self.bodies.append(ground4)
        ground5 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, 0), (start + large_step * 2.25, 0)])
        )
        self.bodies.append(ground5)
        start += large_step * 2.25
        wall2 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, 0), (start, self.r * 2.5)])
        )
        self.bodies.append(wall2)

    def get_initial_pos(self):
        return 0, self.r + 1

    def get_reward(self, morphology, t):
        pos = morphology.get_center_of_mass()[0]
        r = pos - self.prev_pos
        self.prev_pos = pos
        return r

    def get_fitness(self, morphology, t):
        return (morphology.get_center_of_mass()[0] - self.get_initial_pos()[0]) / (t / 60.0)

    def draw_env(self, w, h, center, screen, magnify):
        center_x, center_y = center
        for body in self.bodies:
            vertices = body.fixtures[0].shape.vertices
            l_x, l_y, r_x, r_y = (vertices[0][0] - center_x) * magnify + w / 2, \
                                 (vertices[0][1] - center_y) * magnify + h / 2, \
                                 (vertices[1][0] - center_x) * magnify + w / 2, \
                                 (vertices[1][1] - center_y) * magnify + h / 2
            pygame.draw.lines(screen, (0, 0, 255), False, [(l_x, h - l_y), (r_x, h - r_y)], 5)


class Carrier(BaseEnv):

    def __init__(self, world, config):
        BaseEnv.__init__(self, world)
        self.r = config["r"]
        self.prev_pos = self.get_initial_pos()[0]

    def should_step(self, morphology):
        return all([contact.other != self.bodies[0] for contact in self.bodies[1].contacts])

    def init_env(self):
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-500, 0), (500, 0)]),
        )
        self.bodies.append(ground)
        obj = self.world.CreateDynamicBody(position=(self.r, self.r * 2.5 + 1),
                                           fixtures=b2FixtureDef(shape=b2PolygonShape(box=(self.r / 3, self.r / 3)),
                                                                 density=500, friction=0.0))
        obj.fixedRotation = False
        self.bodies.append(obj)

    def get_initial_pos(self):
        return self.r, self.r + 1

    def get_reward(self, morphology, t):
        pos = morphology.get_center_of_mass()[0]
        r = pos - self.prev_pos
        self.prev_pos = pos
        return r

    def get_fitness(self, morphology, t):
        return (morphology.get_center_of_mass()[0] - self.get_initial_pos()[0]) / (t / 60.0)

    def draw_env(self, w, h, center, screen, magnify):
        center_x, center_y = center
        vertices = self.bodies[0].fixtures[0].shape.vertices
        l_x, l_y, r_x, r_y = (vertices[0][0] - center_x) * magnify + w / 2, \
                             (vertices[0][1] - center_y) * magnify + h / 2, \
                             (vertices[1][0] - center_x) * magnify + w / 2, \
                             (vertices[1][1] - center_y) * magnify + h / 2
        pygame.draw.lines(screen, (0, 0, 255), False, [(l_x, h - l_y), (r_x, h - r_y)], 5)

        shape = self.bodies[1].fixtures[0].shape
        half_width = abs(shape.vertices[0][0] - shape.vertices[1][0]) / 2
        half_height = abs(shape.vertices[0][1] - shape.vertices[2][1]) / 2
        cx, cy = self.bodies[1].position.x, self.bodies[1].position.y
        vertices = [(cx - half_width, cy - half_height), (cx + half_width, cy - half_height),
                    (cx + half_width, cy + half_height), (cx - half_width, cy + half_height)]
        new_vertices = [
            ((x - center_x) * magnify + w / 2, h - ((y - center_y) * magnify + h / 2)) for
            x, y in vertices]
        pygame.draw.rect(screen, (255, 0, 0),
                         (new_vertices[3][0], new_vertices[3][1],
                          half_width * 2 * magnify, half_height * 2 * magnify), 0)
        pygame.draw.rect(screen, (219, 112, 147),
                         (new_vertices[3][0], new_vertices[3][1],
                          half_width * 2 * magnify, half_height * 2 * magnify), 2)

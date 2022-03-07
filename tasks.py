import abc
import math
import os
import random

import numpy as np
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
    def get_fitness(self, morphology, t):
        pass

    @classmethod
    def create_env(cls, config, world):
        name = config["task"]
        if name == "obstacles":
            env = Obstacles(world)
        elif name == "flat":
            env = FlatLocomotion(world, config)
        elif name.startswith("hilly"):
            env = HillyLocomotion(world, config)
        elif name == "escape":
            env = Escape(world, config)
        elif name == "climber":
            env = Climber(world, config)
        elif name == "cave":
            env = CaveCrawler(world, config)
        else:
            raise ValueError("Invalid task name: {}".format(config["task"]))
        env.init_env()
        return env


class Obstacles(BaseEnv):

    def init_env(self):
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-100, 0), (100, 0)])
        )
        self.bodies.append(ground)
        box1 = self.world.CreateStaticBody(
            position=(0, 75),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(25.0, 2.5)),
                                  ))
        box1.fixedRotation = True
        box1.angle = -25 * math.pi / 180.0
        self.bodies.append(box1)
        box2 = self.world.CreateStaticBody(
            position=(55, 55),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(20.0, 2.5)),
                                  ))
        box2.fixedRotation = True
        box2.angle = 45 * math.pi / 180.0
        self.bodies.append(box2)
        box3 = self.world.CreateStaticBody(
            position=(0, 0),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(vertices=[(-50, 0.0),
                                                                 (0, 0.0),
                                                                 (-45, 20),
                                                                 ]
                                                       )))
        box3.fixedRotation = True
        self.bodies.append(box3)

    def get_initial_pos(self):
        return 0, 100

    def get_fitness(self, morphology, t):
        return np.nan


class FlatLocomotion(BaseEnv):

    def __init__(self, world, config):
        BaseEnv.__init__(self, world)
        self.r = config["r"]

    def init_env(self):
        ground = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-500, 0), (500, 0)]),
        )
        self.bodies.append(ground)
        # wall = self.world.CreateBody(
        #     shapes=b2EdgeShape(vertices=[(-7.5, 100), (-7.5, -100)])
        # )
        # self.bodies.append(wall)

    def get_initial_pos(self):
        return self.r, self.r + 1

    def get_fitness(self, morphology, t):
        return (morphology.get_center_of_mass()[0] - self.get_initial_pos()[0]) / (t / 60.0)


class HillyLocomotion(BaseEnv):

    def __init__(self, world, config):
        BaseEnv.__init__(self, world)
        self.h = int(config["task"].split("-")[1])
        self.w = int(config["task"].split("-")[2])
        self.r = config["r"]
        self.file_name = os.path.join(os.getcwd(), "terrains", ".".join(["hilly", str(config["seed"]), "txt"]))

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

    def get_fitness(self, morphology, t):
        return (morphology.get_center_of_mass()[0] - self.get_initial_pos()[0]) / (t / 60.0)


class Escape(BaseEnv):

    def __init__(self, world, config):
        BaseEnv.__init__(self, world)
        self.side = config["r"] * 3

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

    def get_fitness(self, morphology, t):
        return abs(morphology.get_center_of_mass()[0]) - self.get_initial_pos()[0]


class Climber(BaseEnv):

    def __init__(self, world, config):
        BaseEnv.__init__(self, world)
        self.r = config["r"]

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

    def get_fitness(self, morphology, t):
        return abs(morphology.get_center_of_mass()[1]) - self.get_initial_pos()[1]


class CaveCrawler(BaseEnv):

    def __init__(self, world, config):
        BaseEnv.__init__(self, world)
        self.r = config["r"]

    def init_env(self):
        wall1 = self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(- self.r * 2,  0), (- self.r * 2, self.r * 2.5)])
        )
        self.bodies.append(wall1)
        small_step = self.r
        large_step = small_step * 4
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

    def get_fitness(self, morphology, t):
        return (morphology.get_center_of_mass()[0] - self.get_initial_pos()[0]) / (t / 60.0)

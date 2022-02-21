import abc
import math
import os
import random

import numpy as np
from Box2D import b2EdgeShape, b2FixtureDef, b2PolygonShape


class BaseEnv(abc.ABC):

    def __init__(self, world):
        self.world = world

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
    def create_env(cls, args, world):
        if args.task == "obstacles":
            env = Obstacles(world)
        elif args.task == "flat":
            env = FlatLocomotion(world)
        elif args.task.startswith("hilly"):
            env = HillyLocomotion(args, world)
        elif args.task == "escape":
            env = Escape(world)
        elif args.task == "climber":
            env = Climber(world)
        else:
            raise ValueError("Invalid task name: {}".format(args.task))
        env.init_env()
        return env


class Obstacles(BaseEnv):

    def init_env(self):
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-100, 0), (100, 0)])
        )
        box1 = self.world.CreateStaticBody(
            position=(0, 75),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(25.0, 2.5)),
                                  ))
        box1.fixedRotation = True
        box1.angle = -25 * math.pi / 180.0

        box2 = self.world.CreateStaticBody(
            position=(55, 55),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(20.0, 2.5)),
                                  ))
        box2.fixedRotation = True
        box2.angle = 45 * math.pi / 180.0

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

    def get_initial_pos(self):
        return 0, 100

    def get_fitness(self, morphology, t):
        return np.nan


class FlatLocomotion(BaseEnv):

    def init_env(self):
        ground = self.world.CreateStaticBody(
            position=(492.5, 0),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=10.0,
                                  shape=b2PolygonShape(box=(500, 10))),
        )
        ground.angle = 1 * math.pi / 180
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-7.5, 100), (-7.5, -100)])
        )

    def get_initial_pos(self):
        return 0, 6

    def get_fitness(self, morphology, t):
        return (morphology.get_center_of_mass()[0] - self.get_initial_pos()[0]) / (t / 60.0)


class HillyLocomotion(BaseEnv):

    def __init__(self, args, world):
        BaseEnv.__init__(self, world)
        self.h = int(args.task.split("-")[1])
        self.w = int(args.task.split("-")[2])
        self.file_name = os.path.join(os.getcwd(), "terrains", ".".join(["hilly", str(args.seed), "txt"]))

    def init_env(self):
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, 100), (-20, -100)])
        )
        if not os.path.isfile(self.file_name):
            self._write_terrain()
        start = self._read_terrain()
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, 100), (start, -100)])
        )

    def _read_terrain(self):
        with open(self.file_name, "r") as file:
            file.readline()
            for line in file:
                start, end, height, prev_height = tuple(map(lambda x: float(x), line.split(";")))
                half_x, y = (end - start) / 2, 0
                self.world.CreateStaticBody(
                    position=(half_x + start, y),
                    allowSleep=True,
                    fixtures=b2FixtureDef(friction=0.8,
                                          shape=b2PolygonShape(vertices=[(half_x, height), (- half_x, prev_height),
                                                                         (- half_x, -100), (half_x, -100)])),
                )
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
        return 0, 10

    def get_fitness(self, morphology, t):
        return (morphology.get_center_of_mass()[0] - self.get_initial_pos()[0]) / (t / 60.0)

    def get_reward(self, morphology, t):
        return morphology.get_center_of_mass()[0] - morphology.sensor.prev_pos[0]


class Escape(BaseEnv):

    def init_env(self):
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-100, 0), (100, 0)])
        )
        roof = 12
        self.world.CreateStaticBody(
            position=(0, roof),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(roof * 0.75, 1)))
        )
        self.world.CreateStaticBody(
            position=(roof / 1.5, roof / 1.5),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(1, roof / 3)))
        )
        self.world.CreateStaticBody(
            position=(- roof / 1.5, roof / 1.5),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(1, roof / 3)))
        )

    def get_initial_pos(self):
        return 0, 5

    def get_fitness(self, morphology, t):
        return abs(morphology.get_center_of_mass()[0]) - self.get_initial_pos()[0]


class Climber(BaseEnv):

    def init_env(self):
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-100, 0), (100, 0)])
        )
        side = 6
        height = 100
        self.world.CreateStaticBody(
            position=(side, height / 2),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(1, height / 2)))
        )
        self.world.CreateStaticBody(
            position=(- side, height / 2),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(1, height / 2)))
        )

    def get_initial_pos(self):
        return 0, 5

    def get_fitness(self, morphology, t):
        return abs(morphology.get_center_of_mass()[1]) - self.get_initial_pos()[1]

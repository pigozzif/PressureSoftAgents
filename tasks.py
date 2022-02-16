import abc
import math
import os
import random

import numpy as np
from Box2D import b2EdgeShape, b2FixtureDef, b2PolygonShape

from pressure import PressureSoftBody


class BaseEnv(abc.ABC):

    def __init__(self, world):
        self.world = world
        self._init_env()

    @abc.abstractmethod
    def _init_env(self):
        pass

    @abc.abstractmethod
    def get_reward(self, morphology, t):
        pass

    @classmethod
    def create_env(cls, args, world):
        if args.task == "obstacles":
            return Obstacles(world)
        elif args.task == "flat":
            return FlatLocomotion(world)
        elif args.task == "hilly":
            return HillyLocomotion(world, world)
        elif args.task == "escape":
            return Escape(world)
        raise ValueError("Invalid task name: {}".format(args.task))


class Obstacles(BaseEnv):

    def _init_env(self):
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

    def get_reward(self, morphology, t):
        return np.nan


class FlatLocomotion(BaseEnv):

    def _init_env(self):
        ground = self.world.CreateStaticBody(
            position=(492.5, -6),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=10.0,
                                  shape=b2PolygonShape(box=(500, 10))),
        )
        ground.angle = 1 * math.pi / 180
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-7.5, 100), (-7.5, -100)])
        )

    def get_reward(self, morphology, t):
        return morphology.get_center_of_mass()[0] / (t / 60.0)


class HillyLocomotion(BaseEnv):

    def __init__(self, args, world):
        BaseEnv.__init__(self, world)
        self.h = int(args.task.split("-")[1])
        self.w = int(args.task.split("-")[2])

    def _init_env(self):
        width = 400
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, 100), (-20, -100)])
        )
        if os.path.isfile("./terrains/hilly.txt"):
            with open("./terrains/hilly.txt", "r") as file:
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
        else:
            with open("./terrains/hilly.txt", "w") as file:
                file.write(";".join(["start", "end", "height", "prev_height"]) + "\n")
            start = -20
            end = start + max(random.gauss(1, 0.25) * self.w, 1.0)
            prev_height = abs(random.gauss(0, self.h))
            height = abs(random.gauss(0, self.h))
            while end < width:
                half_x, y = (end - start) / 2, 0
                self.world.CreateStaticBody(
                    position=(half_x + start, y),
                    allowSleep=True,
                    fixtures=b2FixtureDef(friction=0.8,
                                          shape=b2PolygonShape(vertices=[(half_x, height), (- half_x, prev_height),
                                                                         (- half_x, -100), (half_x, -100)])),
                )
                with open("./terrains/hilly.txt", "a") as file:
                    file.write(";".join([str(start), str(end), str(height), str(prev_height)]) + "\n")
                start = end
                prev_height = height
                end += max(random.gauss(1, 0.25) * self.w, 1.0)
                height = abs(random.gauss(0, self.h))
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, 100), (start, -100)])
        )

    def get_reward(self, morphology, t):
        return morphology.get_center_of_mass()[0] / (t / 60.0)


class Escape(BaseEnv):

    def _init_env(self):
        ground = PressureSoftBody.r * 2
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-100, ground / 2), (100, ground / 2)])
        )
        self.world.CreateStaticBody(
            position=(0, ground * 1.5),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(ground - 1, 1)))
        )
        self.world.CreateStaticBody(
            position=(ground - 2, ground / 1.5),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(1, ground / 6)))
        )
        self.world.CreateStaticBody(
            position=(- ground + 2, ground / 1.5),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(1, ground / 6)))
        )
        self.world.CreateStaticBody(
            position=(ground - 2, ground / 0.75),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(1, ground / 6)))
        )
        self.world.CreateStaticBody(
            position=(- ground + 2, ground / 0.75),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(1, ground / 6)))
        )

    def get_reward(self, morphology, t):
        return abs(morphology.get_center_of_mass()[0]) / (t / 60.0)

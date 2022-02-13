import math
import random
import os

import numpy as np
import torch
from Box2D import b2FixtureDef, b2PolygonShape, b2EdgeShape

from es import OpenES, SimpleGA, CMAES, PEPG
from pressure import PressureSoftBody
from tensegrity import TensegritySoftBody
from voxel import VoxelSoftBody


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_solver(args, n_params):
    name = args.solver
    if name == "es":
        return OpenES(n_params, forget_best=False)
    elif name == "ga":
        return SimpleGA(n_params, popsize=100)
    elif name == "cmaes":
        pop_size = 4 + math.floor(3 * math.log(n_params))
        return CMAES(n_params, sigma_init=0.5, popsize=pop_size + (args.np - pop_size % args.np))
    elif name == "pepg":
        return PEPG(n_params, forget_best=False)
    raise ValueError("Invalid solver name: {}".format(name))


def create_soft_body(name, world):
    if name == "tensegrity":
        return TensegritySoftBody(world)
    elif name == "pressure":
        return PressureSoftBody(world)
    elif name == "voxel":
        return VoxelSoftBody(world)
    raise ValueError("Invalid soft body name: {}".format(name))


def create_task(world, task_name):
    if task_name == "flat":
        ground = world.CreateStaticBody(
            position=(492.5, -6),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=10.0,
                                  shape=b2PolygonShape(box=(500, 10))),
        )
        ground.angle = 1 * math.pi / 180
        world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-7.5, 100), (-7.5, -100)])
        )
    elif task_name.startswith("hilly"):
        h, w = int(task_name.split("-")[1]), int(task_name.split("-")[2])
        width = 400
        world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, 100), (-20, -100)])
        )
        if os.path.isfile("./hilly.txt"):
            with open("./hilly.txt", "r") as file:
                file.readline()
                for line in file:
                    start, end, height, prev_height = tuple(map(lambda x: float(x), line.split(";")))
                    half_x, y = (end - start) / 2, 0
                    world.CreateStaticBody(
                        position=(half_x + start, y),
                        allowSleep=True,
                        fixtures=b2FixtureDef(friction=0.8,
                                              shape=b2PolygonShape(vertices=[(half_x, height), (- half_x, prev_height),
                                                                             (- half_x, 0), (half_x, 0)])),
                    )
        else:
            with open("./hilly.txt", "w") as file:
                file.write(";".join(["start", "end", "height", "prev_height"]) + "\n")
            start = -20
            end = start + max(random.gauss(1, 0.25) * w, 1.0)
            prev_height = abs(random.gauss(0, h))
            height = abs(random.gauss(0, h))
            while end < width:
                half_x, y = (end - start) / 2, 0
                world.CreateStaticBody(
                    position=(half_x + start, y),
                    allowSleep=True,
                    fixtures=b2FixtureDef(friction=0.8,
                                          shape=b2PolygonShape(vertices=[(half_x, height), (- half_x, prev_height),
                                                                         (- half_x, 0), (half_x, 0)])),
                )
                with open("./hilly.txt", "a") as file:
                    file.write(";".join([str(start), str(end), str(height), str(prev_height)]) + "\n")
                start = end
                prev_height = height
                end += max(random.gauss(1, 0.25) * w, 1.0)
                height = abs(random.gauss(0, h))
        world.CreateBody(
            shapes=b2EdgeShape(vertices=[(start, 100), (start, -100)])
        )
    elif task_name == "obstacles":
        world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-100, -100), (100, -100)])
        )
        box1 = world.CreateStaticBody(
            position=(0, -15),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(25.0, 2.5)),
                                  ))
        box1.fixedRotation = True
        box1.angle = -25 * math.pi / 180.0

        box2 = world.CreateStaticBody(
            position=(55, -45),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(box=(20.0, 2.5)),
                                  ))
        box2.fixedRotation = True
        box2.angle = 45 * math.pi / 180.0

        box3 = world.CreateStaticBody(
            position=(0, -100),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(vertices=[(-50, 0.0),
                                                                 (0, 0.0),
                                                                 (-45, 20),
                                                                 ]
                                                       )))
        box3.fixedRotation = True
    else:
        raise ValueError("Invalid task name: {}".format(task_name))

import math

from Box2D import b2FixtureDef, b2PolygonShape, b2EdgeShape

from src.pressure import PressureSoftBody
from src.tensegrity import TensegritySoftBody
from src.voxel import VoxelSoftBody


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
        world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, -10), (1000, -10)])
        )
        world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-20, 100), (-20, -100)])
        )
        return
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
        return
    raise ValueError("Invalid task name: {}".format(task_name))

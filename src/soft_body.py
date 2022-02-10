import abc
import math

from Box2D import b2EdgeShape, b2FixtureDef, b2PolygonShape


class SoftBody(abc.ABC):

    def __init__(self, world, min_x, max_x):
        self.world = world
        # The ground
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(min_x, -100), (max_x, -100)])
        )

    @abc.abstractmethod
    def physics_step(self):
        pass

    @abc.abstractmethod
    def sense(self):
        pass

    def create_test_obstacles(self):
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

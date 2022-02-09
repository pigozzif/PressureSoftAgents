import math

from Box2D import b2FixtureDef, b2PolygonShape, b2EdgeShape, b2DistanceJointDef, b2CircleShape
from Box2D.examples.framework import Framework, main
from enum import Enum


class Direction(Enum):
    N = (0, 1)
    NE = (1, 1)
    SE = (1, -1)
    S = (0, -1)
    SW = (-1, -1)
    NW = (-1, 1)

    def __init__(self, x, y):
        self.x = x
        self.y = y


class TensegrityModule(object):

    def __init__(self, x, y, world, fixture, half_width, modules):
        self.x, self.y = x / (half_width * 2), y / (math.sqrt(3) * half_width)
        self.world = world
        self.masses = self._create_masses(x, y, fixture, half_width, modules)
        self.cables = self._create_cables()
        self.rods = self._create_rods()

    def _create_masses(self, x, y, fixture, half_width, modules):
        masses = [self.world.CreateDynamicBody(position=(x + half_width, y), fixtures=fixture)]
        if (self.x, self.y - 1) in modules:
            masses.append(modules[(self.x, self.y - 1)].masses[5])
            masses.append(modules[(self.x, self.y - 1)].masses[4])
        else:
            masses.append(self.world.CreateDynamicBody(position=(x + half_width / 2, y -
                                                                 (math.sqrt(3) * half_width) / 2), fixtures=fixture))
            masses.append(self.world.CreateDynamicBody(position=(x - half_width / 2, y -
                                                                 (math.sqrt(3) * half_width) / 2),
                                                       fixtures=fixture))
        if (self.x - 1, self.y) in modules:
            masses.append(modules[(self.x - 1, self.y)].masses[0])
        else:
            masses.append(self.world.CreateDynamicBody(position=(x - half_width, y), fixtures=fixture))
        masses.append(self.world.CreateDynamicBody(position=(x - half_width / 2, y +
                                                             (math.sqrt(3) * half_width) / 2), fixtures=fixture))
        masses.append(self.world.CreateDynamicBody(position=(x + half_width / 2, y +
                                                             (math.sqrt(3) * half_width) / 2), fixtures=fixture))
        return masses

    def _create_cables(self):
        cables = []
        for i in range(len(self.masses)):
            cables.append(self._create_cable(self.masses[i], self.masses[(i + 1) % len(self.masses)]))
        return cables

    def _create_cable(self, mass_a, mass_b):
        dfn = b2DistanceJointDef(
            bodyA=mass_a,
            bodyB=mass_b,
            anchorA=mass_a.position,
            anchorB=mass_b.position,
            frequencyHz=4.0,
            dampingRatio=0.5,
            collideConnected=True
        )
        return self.world.CreateJoint(dfn)

    def _create_rods(self):
        rods = []
        for i in range(len(self.masses)):
            rods.append(self._create_rod(self.masses[i], self.masses[(i + 2) % len(self.masses)]))
            rods.append(self._create_rod(self.masses[i], self.masses[(i + 4) % len(self.masses)]))
        return rods

    def _create_rod(self, mass_a, mass_b):
        dfn = b2DistanceJointDef(
            bodyA=mass_a,
            bodyB=mass_b,
            anchorA=mass_a.position,
            anchorB=mass_b.position,
            frequencyHz=4.0,
            dampingRatio=0,
            collideConnected=True
        )
        return self.world.CreateJoint(dfn)


class TensegrityBased(Framework):
    name = "Tensegrity-based soft body"
    description = "Demonstration of a tensegrity-based soft body simulation."
    n_modules = 2
    n_bodies_x = 4
    n_bodies_y = 2
    module_size = 2.5

    def __init__(self):
        super(TensegrityBased, self).__init__()

        # The ground
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-100, -100), (100, -100)])
        )
        self._create_obstacles()
        fixture = b2FixtureDef(shape=b2CircleShape(),
                               density=5, friction=0.2)
        self.modules = {}
        for x in range(self.n_bodies_x + 1):
            for y in range(self.n_bodies_y + 1):
                pos_x, pos_y = x * self.module_size * 2, y * math.sqrt(3) * self.module_size
                self.modules[(x, y)] = TensegrityModule(pos_x, pos_y, self.world, fixture, self.module_size,
                                                        self.modules)

    def _create_obstacles(self):
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


if __name__ == "__main__":
    main(TensegrityBased)

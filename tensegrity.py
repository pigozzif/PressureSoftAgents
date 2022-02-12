import math

import numpy as np
from Box2D import b2FixtureDef, b2DistanceJointDef, b2CircleShape

from soft_body import BaseSoftBody


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
            collideConnected=False
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


class TensegritySoftBody(BaseSoftBody):
    name = "Tensegrity-based soft body"
    description = "Demonstration of a tensegrity-based soft body simulation."
    n_modules = 2
    n_bodies_x = 4
    n_bodies_y = 2
    module_size = 2.5

    def __init__(self, world):
        super(TensegritySoftBody, self).__init__(world)
        fixture = b2FixtureDef(shape=b2CircleShape(),
                               density=5, friction=0.2)
        self.modules = {}
        for x in range(self.n_bodies_x + 1):
            for y in range(self.n_bodies_y + 1):
                pos_x, pos_y = x * self.module_size * 2, y * math.sqrt(3) * self.module_size
                self.modules[(x, y)] = TensegrityModule(pos_x, pos_y, self.world, fixture, self.module_size,
                                                        self.modules)

    def physics_step(self):
        pass

    def sense(self):
        return np.array(
            [[min(len(mass.contacts), 1) for mass in module.masses] for module in self.modules.values()]).flatten()

    def apply_control(self, control):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError

    def get_center_of_mass(self):
        return np.mean([np.mean([mass.position for mass in module.masses], axis=0) for module in self.modules.values()],
                       axis=0)

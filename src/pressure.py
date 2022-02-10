import math

import numpy as np
from Box2D import b2FixtureDef, b2CircleShape, b2DistanceJointDef, b2Vec2
import matplotlib.path as path

from soft_body import SoftBody


class PressureSoftBody(SoftBody):
    name = "Pressure-based Soft Body"
    description = "Demonstration of a pressure-based soft body simulation."
    n_masses = 25
    r = 10
    n = 28.0134 * 500
    R = 8.31446261815324
    T = 298
    nRT = n * R * T

    def __init__(self, world):
        super(PressureSoftBody, self).__init__(world)
        fixture = b2FixtureDef(shape=b2CircleShape(radius=1),
                               density=5000, friction=0.2)
        self.masses = []
        self.joints = []
        self._add_masses(fixture)

    def _add_masses(self, fixture):
        delta_theta = (360 * math.pi / 180) / self.n_masses
        theta = 0
        prev_mass = None
        for i in range(self.n_masses):
            x = self.r * math.cos(theta)
            y = self.r * math.sin(theta)
            mass = self.world.CreateDynamicBody(position=(x, y), fixtures=fixture)
            mass.angle = theta
            if prev_mass is not None:
                self._add_joint(prev_mass, mass)
            theta += delta_theta
            self.masses.append(mass)
            prev_mass = mass
        self._add_joint(prev_mass, self.masses[0])

    def _add_joint(self, mass, prev_mass):
        dfn = b2DistanceJointDef(
            bodyA=prev_mass,
            bodyB=mass,
            anchorA=prev_mass.position,
            anchorB=mass.position,
            dampingRatio=0.5,
            collideConnected=False
        )
        self.joints.append(self.world.CreateJoint(dfn))

    @staticmethod
    def _get_normalized_normal(mass_a, mass_b, polygon):
        normal1 = b2Vec2(- mass_b.position.y + mass_a.position.y, mass_b.position.x - mass_a.position.x)
        normal1.Normalize()
        midpoint_x, midpoint_y = (mass_a.position.x + mass_b.position.x) / 2, (mass_a.position.y + mass_b.position.y) / 2
        point1 = np.array([midpoint_x + normal1.x, midpoint_y + normal1.y])
        normal2 = b2Vec2(mass_b.position.y - mass_a.position.y, - mass_b.position.x + mass_a.position.x)
        normal2.Normalize()
        a = (polygon.contains_point(point1, radius=0.001) or polygon.contains_point(point1, radius=-0.001))
        return normal2 if a else normal1

    @staticmethod
    def _get_area(positions):
        return 0.5 * abs(sum(x0 * y1 - x1 * y0
                             for ((x0, y0), (x1, y1)) in zip(positions, positions[1:] + [positions[0]])))

    def physics_step(self):
        positions = [mass.position for mass in self.masses]
        area = self._get_area(positions)
        polygon = path.Path(np.array(positions))
        for joint in self.joints:
            mass_a = joint.bodyA
            mass_b = joint.bodyB
            normal = self._get_normalized_normal(mass_a, mass_b, polygon)
            pressure = (self.nRT / area) * joint.length
            pressure /= 2
            pressure_force = normal * pressure
            mass_a.ApplyForceToCenter(pressure_force, True)
            mass_b.ApplyForceToCenter(pressure_force, True)

    def sense(self):
        return np.array([min(len(mass.contacts), 1) for mass in self.masses])

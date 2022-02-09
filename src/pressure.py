import math

import numpy as np
from Box2D import b2EdgeShape, b2FixtureDef, b2PolygonShape, b2CircleShape, b2DistanceJointDef, \
    b2Vec2
from Box2D.examples.framework import Framework, main
import matplotlib.path as path


class PressureBased(Framework):
    name = "Pressure-based Soft Body"
    description = "Demonstration of a pressure-based soft body simulation."
    n_masses = 25
    r = 10
    n = 28.0134 * 500
    R = 8.31446261815324
    T = 298
    nRT = n * R * T

    def __init__(self):
        super(PressureBased, self).__init__()

        # The ground
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-100, -100), (100, -100)]),
        )
        self._create_obstacles()

        fixture = b2FixtureDef(shape=b2CircleShape(radius=1),
                               density=5000, friction=0.2)
        self.masses = []
        self.joints = []
        self._add_masses(fixture)

    def _add_masses(self, fixture):
        delta_theta = (360 * math.pi / 180) / self.n_masses
        theta = 0
        for i in range(self.n_masses):
            x = self.r * math.cos(theta)
            y = self.r * math.sin(theta)
            mass = self.world.CreateDynamicBody(position=(x, y), fixtures=fixture)
            mass.angle = theta
            self._add_joint(self.masses[i], self.masses[(i + 1) % len(self.masses)])
            theta += delta_theta
            self.masses.append(mass)

    def _add_joint(self, mass, prev_mass):
        dfn = b2DistanceJointDef(
            bodyA=prev_mass,
            bodyB=mass,
            anchorA=prev_mass.position,
            anchorB=mass.position,
            dampingRatio=0.5,
            collideConnected=True
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

    def apply_pressure(self):
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

    def Step(self, settings):
        Framework.Step(self, settings)
        self.apply_pressure()


if __name__ == "__main__":
    main(PressureBased)

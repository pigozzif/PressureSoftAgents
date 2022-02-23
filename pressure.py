import math

import numpy as np
from Box2D import b2FixtureDef, b2DistanceJointDef, b2Vec2, b2PolygonShape
import matplotlib.path as path
from dataclasses import dataclass

from soft_body import BaseSoftBody, SpringData, Sensor


@dataclass
class PressureData(object):
    current: float
    min: float
    mid: float
    max: float


class PressureSoftBody(BaseSoftBody):
    mol = 28.0134
    R = 8.31446261815324

    def __init__(self, config, world, start_x, start_y, control_pressure=False):
        super(PressureSoftBody, self).__init__(world, start_x, start_y)
        self.n_masses = config["n_masses"]
        self.r = config["r"]
        self.mass = config["mass"]
        self.n = self.mass * self.mol
        self.T = config["T"]
        self.nRT = self.n * self.R * self.T
        fixture = b2FixtureDef(shape=b2PolygonShape(box=(0.5, 0.5)),
                               density=2500, friction=10.0)
        self.masses = []
        self.joints = []
        self._add_masses(fixture)
        self.sensor = Sensor(self.n_masses * 3 + 2 + 1, 0.25 * 60, self)
        self.control_pressure = control_pressure
        max_p = self.nRT / (math.pi * self.r * 2)
        min_p = max_p * 0.2
        self.pressure = PressureData(self._compute_pressure([mass.position for mass in self.masses]), min_p,
                                     (max_p - min_p) / 2 + min_p, max_p)

    def _add_masses(self, fixture):
        delta_theta = (360 * math.pi / 180) / self.n_masses
        theta = 0
        prev_mass = None
        for i in range(self.n_masses):
            x = self.r * math.cos(theta) + self.start_x
            y = self.r * math.sin(theta) + self.start_y
            mass = self.world.CreateDynamicBody(position=(x, y), fixtures=fixture)
            mass.angle = theta
            if prev_mass is not None:
                self._add_joint(prev_mass, mass)
            theta += delta_theta
            self.masses.append(mass)
            prev_mass = mass
        self._add_joint(prev_mass, self.masses[0])

    def _add_joint(self, mass, prev_mass):
        distance = math.sqrt((prev_mass.position[0] - mass.position[0]) ** 2 +
                             (prev_mass.position[1] - mass.position[1]) ** 2)
        dfn = b2DistanceJointDef(
            bodyA=prev_mass,
            bodyB=mass,
            anchorA=prev_mass.position,
            anchorB=mass.position,
            dampingRatio=0.3,
            frequencyHz=8,
            collideConnected=False,
            userData=SpringData(distance, distance * 1.25, distance * 0.75)
        )
        self.joints.append(self.world.CreateJoint(dfn))

    def size(self):
        return self.r * 2, self.r * 2

    @staticmethod
    def _get_normalized_normal(mass_a, mass_b, polygon):
        normal1 = b2Vec2(- mass_b.position.y + mass_a.position.y, mass_b.position.x - mass_a.position.x)
        normal1.Normalize()
        midpoint_x, midpoint_y = (mass_a.position.x + mass_b.position.x) / 2,\
                                 (mass_a.position.y + mass_b.position.y) / 2
        point1 = np.array([midpoint_x + normal1.x, midpoint_y + normal1.y])
        normal2 = b2Vec2(mass_b.position.y - mass_a.position.y, - mass_b.position.x + mass_a.position.x)
        normal2.Normalize()
        a = (polygon.contains_point(point1, radius=0.001) or polygon.contains_point(point1, radius=-0.001))
        return normal2 if a else normal1

    @staticmethod
    def _get_area(positions):
        return 0.5 * abs(sum(x0 * y1 - x1 * y0
                             for ((x0, y0), (x1, y1)) in zip(positions, positions[1:] + [positions[0]])))

    def _compute_pressure(self, positions):
        return self.nRT / PressureSoftBody._get_area(positions)

    def physics_step(self):
        positions = [mass.position for mass in self.masses]
        if not self.control_pressure:
            self.pressure.current = self._compute_pressure(positions)
        polygon = path.Path(np.array(positions))
        for joint in self.joints:
            mass_a = joint.bodyA
            mass_b = joint.bodyB
            normal = self._get_normalized_normal(mass_a, mass_b, polygon)
            pressure = self.pressure.current * joint.length
            pressure /= 2
            pressure_force = normal * pressure
            mass_a.ApplyForceToCenter(pressure_force, True)
            mass_b.ApplyForceToCenter(pressure_force, True)

    def get_obs(self):
        return self.sensor.sense(self)

    def apply_control(self, control):
        if self.control_pressure:
            force = control[-1]
            # if force >= 0:
            #     self.pressure.current = self.pressure.mid - 0.99 * ((self.pressure.mid - self.pressure.min) * force)
            # else:
            #     self.pressure.current = self.pressure.mid + 0.99 * ((self.pressure.max - self.pressure.mid) * (- force))
            self.pressure.current = min(max(force, self.pressure.min), self.pressure.max)  # min(max(self.pressure.current + force * 100, self.pressure.min), self.pressure.max)
        for force, joint in zip(control[:-1 if self.control_pressure else 0], self.joints):
            data = joint.userData
            if force >= 0:
                joint.length = data.rest_length - (data.rest_length - data.min) * force
            else:
                joint.length = data.rest_length + (data.max - data.rest_length) * (- force)

    def get_output_dim(self):
        return len(self.joints) + (1 if self.control_pressure else 0)

    def get_center_of_mass(self):
        return np.mean([mass.position for mass in self.masses], axis=0)

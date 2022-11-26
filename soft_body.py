import abc
import math

import numpy as np
from Box2D import b2DistanceJointDef, b2FixtureDef, b2CircleShape, b2Vec2, b2PolygonShape
from dataclasses import dataclass
from matplotlib import path


class Sensor(object):

    def __init__(self, dim, window_size, morphology):
        self.dim = dim
        self.window_size = window_size
        self._memory = np.empty((0, dim))
        self.prev_pos = morphology.get_center_of_mass()

    def sense(self, morphology):
        curr_pos = morphology.get_center_of_mass()
        obs = np.concatenate([np.array([min(len(mass.contacts), 1) for mass in morphology.masses]),
                              np.ravel([mass.position - curr_pos for mass in morphology.masses]),
                              np.ravel([curr_pos - self.prev_pos]),
                              np.array([morphology.pressure.current])], axis=0)
        self.prev_pos = curr_pos
        return self.normalize_obs(obs, morphology)

    def normalize_obs(self, obs, morphology):
        for i, o in enumerate(obs):
            if len(morphology.masses) <= i < len(obs) - 2:
                obs[i] /= 5 * 1.75
            elif len(obs) - 3 <= i < len(obs) - 1:
                obs[i] /= 8.0
            elif i == len(obs) - 1:
                obs[i] /= morphology.pressure.max
        self._realloc_memory(obs)
        obs = np.mean(self._memory, axis=0)
        return obs

    def _realloc_memory(self, obs):
        if len(self._memory) >= self.window_size:
            self._memory = np.delete(self._memory, 0, axis=0).reshape(-1, self.dim)
        self._memory = np.append(self._memory, obs.reshape(1, -1), axis=0).reshape(-1, self.dim)


class BaseSoftBody(abc.ABC):

    def __init__(self, world, start_x, start_y):
        self.world = world
        self.start_x = start_x
        self.start_y = start_y

    @abc.abstractmethod
    def size(self):
        pass

    @abc.abstractmethod
    def physics_step(self):
        pass

    @abc.abstractmethod
    def get_obs(self):
        pass

    @abc.abstractmethod
    def apply_control(self, control):
        pass

    def get_input_dim(self):
        return len(self.get_obs())

    @abc.abstractmethod
    def get_output_dim(self):
        pass

    @abc.abstractmethod
    def get_center_of_mass(self):
        pass

    @classmethod
    def create_soft_body(cls, config, pos, world):
        name = config["body"]
        if name == "tensegrity":
            return TensegritySoftBody(world, pos[0], pos[1])
        elif name == "pressure":
            return PressureSoftBody(config, world, pos[0], pos[1],
                                    control_pressure=bool(int(config["control_pressure"])))
        elif name == "voxel":
            return VoxelSoftBody(world, pos[0], pos[1])
        raise ValueError("Invalid soft body name: {}".format(config["body"]))


@dataclass
class SpringData(object):
    min: float
    max: float

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
        fixture = b2FixtureDef(shape=b2PolygonShape(box=(0.5, 0.5)), density=2500, friction=10.0)
        self.masses = []
        self.joints = []
        self._add_masses(fixture)
        self.sensor = Sensor(self.n_masses * 3 + 2 + 1, 0.25 * 60, self)
        self.control_pressure = control_pressure
        max_p = self.get_maximum_pressure(self.T, self.mass, self.r)
        min_p = max_p * 0.2
        self.pressure = PressureData(self._compute_pressure(), min_p, (max_p - min_p) / 2 + min_p, max_p)
        self.mass_marker = {}

    @staticmethod
    def get_maximum_pressure(T, mass, r):
        return ((PressureSoftBody.R * T * mass * PressureSoftBody.mol) / (r ** 2 * math.pi)) * 1.25

    def _add_masses(self, fixture):
        delta_theta = (360 * math.pi / 180) / self.n_masses
        theta = 0
        prev_mass = None
        for i in range(self.n_masses):
            x = self.r * math.cos(theta) + self.start_x
            y = self.r * math.sin(theta) + self.start_y
            mass = self.world.CreateDynamicBody(position=(x, y), fixtures=fixture)
            mass.angle = theta
            mass.fixedRotation = True
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
            dampingRatio=0.3,
            frequencyHz=8,
            collideConnected=False,
            userData=SpringData(-10, 10)
        )
        self.joints.append(self.world.CreateJoint(dfn))

    def size(self):
        return self.r * 2, self.r * 2

    @staticmethod
    def _get_normalized_normal(mass_a, mass_b, polygon):
        normal1 = b2Vec2(- mass_b.position.y + mass_a.position.y, mass_b.position.x - mass_a.position.x)
        normal1.Normalize()
        midpoint_x, midpoint_y = PressureSoftBody._get_midpoint(mass_a, mass_b)
        point1 = np.array([midpoint_x + normal1.x, midpoint_y + normal1.y])
        normal2 = b2Vec2(mass_b.position.y - mass_a.position.y, - mass_b.position.x + mass_a.position.x)
        normal2.Normalize()
        a = (polygon.contains_point(point1, radius=0.001) or polygon.contains_point(point1, radius=-0.001))
        return normal2 if a else normal1

    @staticmethod
    def _get_midpoint(mass_a, mass_b):
        return (mass_a.position.x + mass_b.position.x) / 2, (mass_a.position.y + mass_b.position.y) / 2

    @staticmethod
    def get_polygon_area(positions):
        return 0.5 * abs(sum(x0 * y1 - x1 * y0
                             for ((x0, y0), (x1, y1)) in zip(positions, positions[1:] + [positions[0]])))

    def _compute_pressure(self):
        return self.nRT / self.get_polygon_area([mass.position for mass in self.masses])

    def physics_step(self):
        positions = [mass.position for mass in self.masses]
        if not self.control_pressure:
            self.pressure.current = self._compute_pressure()
        polygon = path.Path(np.array(positions))
        self.mass_marker.clear()
        center_of_mass = self.get_center_of_mass()
        for i, mass in enumerate(self.masses):
            prev_mass = self.masses[i - 1 if i != 0 else len(self.masses) - 1]
            next_mass = self.masses[i + 1 if i != len(self.masses) - 1 else 0]
            midpoint = np.array([*self._get_midpoint(prev_mass, next_mass)])
            if np.linalg.norm([center_of_mass - midpoint]) > np.linalg.norm([center_of_mass - mass.position]):
                mass.linearVelocity = - mass.linearVelocity
        for joint in self.joints:
            mass_a = joint.bodyA
            mass_b = joint.bodyB
            normal = self._get_normalized_normal(mass_a, mass_b, polygon)
            pressure = self.pressure.current * joint.length
            pressure /= 2
            pressure_force = normal * pressure
            if mass_a not in self.mass_marker:
                mass_a.ApplyForceToCenter(pressure_force, True)
            else:
                new_pressure_force = self.mass_marker[mass_a] * pressure_force
                mass_a.ApplyForceToCenter(b2Vec2(new_pressure_force[0], new_pressure_force[1]), True)
            if mass_b not in self.mass_marker:
                mass_b.ApplyForceToCenter(pressure_force, True)
            else:
                new_pressure_force = self.mass_marker[mass_b] * pressure_force
                mass_b.ApplyForceToCenter(b2Vec2(new_pressure_force[0], new_pressure_force[1]), True)

    def get_obs(self):
        return self.sensor.sense(self)

    def apply_control(self, control):
        if self.control_pressure:
            self.pressure.current = min(max(self.pressure.current + control[-1], self.pressure.min), self.pressure.max)
        for force, joint in zip(control[:-1 if self.control_pressure else 0], self.joints):
            joint.frequency = min(max(int(joint.frequency + force), joint.userData.min), joint.userData.max)

    def get_output_dim(self):
        return len(self.joints) + (1 if self.control_pressure else 0)

    def get_center_of_mass(self):
        return np.mean([mass.position for mass in self.masses], axis=0)


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
    n_modules = 2
    n_bodies_x = 4
    n_bodies_y = 2
    module_size = 2.5

    def __init__(self, world, start_x, start_y):
        super(TensegritySoftBody, self).__init__(world, start_x, start_y)
        fixture = b2FixtureDef(shape=b2CircleShape(),
                               density=5, friction=0.2)
        self.modules = {}
        for x in range(self.n_bodies_x + 1):
            for y in range(self.n_bodies_y + 1):
                pos_x, pos_y = x * self.module_size * 2 + self.start_x,\
                               y * math.sqrt(3) * self.module_size + self.start_y
                self.modules[(x, y)] = TensegrityModule(pos_x, pos_y, self.world, fixture, self.module_size,
                                                        self.modules)

    def size(self):
        return self.n_bodies_x * self.module_size, self.n_bodies_y * self.module_size

    def physics_step(self):
        pass

    def get_obs(self):
        return np.array(
            [[min(len(mass.contacts), 1) for mass in module.masses] for module in self.modules.values()]).flatten()

    def apply_control(self, control):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError

    def get_center_of_mass(self):
        return np.mean([np.mean([mass.position for mass in module.masses], axis=0) for module in self.modules.values()],
                       axis=0)


class Voxel(object):

    def __init__(self, x, y, voxel_size, world, fixture, voxels, max_x, max_y):
        self.x, self.y = x / voxel_size, (y - 1) / voxel_size
        self.world = world
        # initialize masses
        self.masses = self._create_masses(x, y, voxel_size, fixture, voxels)
        # initialize joints
        self.joints = self._create_joints(voxels, max_x, max_y)

    def _create_masses(self, x, y, voxel_size, fixture, voxels):
        masses = {}

        if (self.x, self.y - 1) in voxels:
            masses[0] = voxels[(self.x, self.y - 1)].masses[1]
        elif (self.x - 1, self.y) in voxels:
            masses[0] = voxels[(self.x - 1, self.y)].masses[2]
        else:
            masses[0] = self.world.CreateDynamicBody(position=(x, y), fixtures=fixture)

        if (self.x, self.y - 1) in voxels:
            masses[2] = voxels[self.x, self.y - 1].masses[3]
        else:
            masses[2] = self.world.CreateDynamicBody(position=(x + voxel_size, y), fixtures=fixture)

        if (self.x - 1, self.y) in voxels:
            masses[1] = voxels[(self.x - 1, self.y)].masses[3]
        else:
            masses[1] = self.world.CreateDynamicBody(position=(x, y + voxel_size), fixtures=fixture)

        masses[3] = self.world.CreateDynamicBody(position=(x + voxel_size, y + voxel_size), fixtures=fixture)
        return masses

    def _create_joints(self, voxels, max_x, max_y):
        joints = []
        sets = [(self.masses[0], self.masses[3], (0, 0), (-0, -0)),
                (self.masses[1], self.masses[2], (0, -0), (-0, 0))
                ]

        if self.y + 1 == max_y:
            sets.append((self.masses[1], self.masses[3], (0, 0), (-0, -0)))
        else:
            sets.append((self.masses[1], self.masses[3], (0, 0), (-0, 0)))

        if self.x + 1 == max_x:
            sets.append((self.masses[3], self.masses[2], (0, 0), (0, -0)))
        else:
            sets.append((self.masses[3], self.masses[2], (0, 0), (0, -0)))

        if (self.x, self.y - 1) not in voxels:
            if self.y == 0:
                sets.append((self.masses[2], self.masses[0], (-0, 0), (0, 0)))
            else:
                sets.append((self.masses[2], self.masses[0], (-0, 0), (0, 0)))
        if (self.x - 1, self.y) not in voxels:
            if self.x == 0:
                sets.append((self.masses[0], self.masses[1], (-0, -0), (-0, 0)))
            else:
                sets.append((self.masses[0], self.masses[1], (0, -0), (0, 0)))

        for bodyA, bodyB, localAnchorA, localAnchorB in sets:
            dfn = b2DistanceJointDef(
                frequencyHz=4.0,
                dampingRatio=0.5,
                bodyA=bodyA,
                bodyB=bodyB,
                localAnchorA=localAnchorA,
                localAnchorB=localAnchorB,
            )
            joints.append(self.world.CreateJoint(dfn))
        return joints


class VoxelSoftBody(BaseSoftBody):
    voxels = []
    joints = []
    n_bodies_x = 2
    n_bodies_y = 5
    voxel_size = 5

    def __init__(self, world, start_x, start_y):
        super(VoxelSoftBody, self).__init__(world, start_x, start_y)
        fixture = b2FixtureDef(shape=b2CircleShape(),
                               density=5, friction=0.2)
        # initialize voxels
        self.voxels = {}
        for x in range(self.n_bodies_x + 1):
            for y in range(self.n_bodies_y + 1):
                pos_x, pos_y = x * self.voxel_size + self.start_x, y * self.voxel_size + self.start_y
                self.voxels[(x, y)] = Voxel(pos_x, pos_y + 1, self.voxel_size, self.world, fixture, self.voxels,
                                            self.n_bodies_x, self.n_bodies_y)

    def size(self):
        return self.n_bodies_x * self.voxel_size, self.n_bodies_y * self.voxel_size

    def physics_step(self):
        pass

    def get_obs(self):
        return np.array([[min(len(mass.contacts), 1) for mass in voxel.masses.values()]
                         for voxel in self.voxels.values()]).flatten()

    def apply_control(self, control):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError

    def get_center_of_mass(self):
        return np.mean([np.mean([mass.position for mass in voxel.masses.values()], axis=0)
                        for voxel in self.voxels.values()], axis=0)

import numpy as np
from Box2D import (b2DistanceJointDef, b2FixtureDef, b2CircleShape)

from soft_body import BaseSoftBody


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
        return np.array([[min(len(mass.contacts), 1) for mass in voxel.masses.values()] for voxel in self.voxels.values()]).flatten()

    def apply_control(self, control):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError

    def get_center_of_mass(self):
        return np.mean([np.mean([mass.position for mass in voxel.masses.values()], axis=0) for voxel in self.voxels.values()], axis=0)

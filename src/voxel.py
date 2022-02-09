import math

from Box2D.examples.framework import (Framework, Keys, main)
from Box2D import (b2DistanceJointDef, b2EdgeShape, b2FixtureDef,
                   b2PolygonShape, b2CircleShape)


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


class VoxelBased(Framework):
    name = "Voxel-based soft body"
    description = "Demonstration of a voxel-based soft body simulation."
    voxels = []
    joints = []
    n_bodies_x = 2
    n_bodies_y = 5
    voxel_size = 5

    def __init__(self):
        super(VoxelBased, self).__init__()

        # The ground
        self.world.CreateBody(
            shapes=b2EdgeShape(vertices=[(-100, -100), (100, -100)])
        )
        self._create_obstacles()

        fixture = b2FixtureDef(shape=b2CircleShape(),
                               density=5, friction=0.2)

        # initialize voxels
        self.voxels = {}
        for x in range(self.n_bodies_x + 1):
            for y in range(self.n_bodies_y + 1):
                pos_x, pos_y = x * self.voxel_size, y * self.voxel_size
                self.voxels[(x, y)] = Voxel(pos_x, pos_y + 1, self.voxel_size, self.world, fixture, self.voxels,
                                            self.n_bodies_x, self.n_bodies_y)

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
            position=(25, -100),
            allowSleep=True,
            fixtures=b2FixtureDef(friction=0.8,
                                  shape=b2PolygonShape(vertices=[(-25, 0.0),
                                                                 (25, 0.0),
                                                                 (-20, 20),
                                                                 ]
                                                       )))
        box3.fixedRotation = True

    def Keyboard(self, key):
        if key == Keys.K_b:
            for voxel in self.voxels.values():
                for body in voxel.masses.values():
                    # Gets both FixtureDestroyed and JointDestroyed callbacks.
                    self.world.DestroyBody(body)
                    break

        elif key == Keys.K_j:
            for joint in self.joints:
                # Does not get a JointDestroyed callback!
                self.world.DestroyJoint(joint)
                self.joints.remove(joint)
                break

    def FixtureDestroyed(self, fixture):
        super(VoxelBased, self).FixtureDestroyed(fixture)
        body = fixture.body
        for voxel in self.voxels.values():
            for k, v in voxel.masses.items():
                if body is v:
                    del voxel.masses[k]

    def JointDestroyed(self, joint):
        if joint in self.joints:
            self.joints.remove(joint)


if __name__ == "__main__":
    main(VoxelBased)

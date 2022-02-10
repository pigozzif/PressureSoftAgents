from Box2D.examples.framework import Framework

from src.utils import create_soft_body


def main(name, *args):
    framework = SoftBodyFramework(name, *args)
    framework.run()


class SoftBodyFramework(Framework):

    def __init__(self, soft_body_name, min_x, max_x):
        super(SoftBodyFramework, self).__init__()
        self.soft_body = create_soft_body(soft_body_name, self.world, min_x, max_x)
        self.soft_body.create_test_obstacles()
        self.name = self.soft_body.name
        self.description = self.soft_body.description
        help(self.world.contactListener)

    def Step(self, settings):
        Framework.Step(self, settings)
        self.soft_body.physics_step()


if __name__ == "__main__":
    main("voxel", -100, 100)

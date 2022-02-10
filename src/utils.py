from src.pressure import PressureSoftBody
from src.tensegrity import TensegritySoftBody
from src.voxel import VoxelSoftBody


def create_soft_body(name, world, min_x, max_x):
    if name == "tensegrity":
        return TensegritySoftBody(world, min_x, max_x)
    elif name == "pressure":
        return PressureSoftBody(world, min_x, max_x)
    elif name == "voxel":
        return VoxelSoftBody(world, min_x, max_x)
    raise ValueError("Invalid soft body name: {}".format(name))

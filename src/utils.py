from src.pressure import PressureSoftBody
from src.tensegrity import TensegritySoftBody
from src.voxel import VoxelSoftBody


def create_soft_body(name, world):
    if name == "tensegrity":
        return TensegritySoftBody(world)
    elif name == "pressure":
        return PressureSoftBody(world)
    elif name == "voxel":
        return VoxelSoftBody(world)
    raise ValueError("Invalid soft body name: {}".format(name))

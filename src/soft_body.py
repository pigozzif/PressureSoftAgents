import abc

from dataclasses import dataclass


@dataclass
class SpringData(object):
    rest_length: float
    min: float
    max: float


class BaseSoftBody(abc.ABC):

    def __init__(self, world):
        self.world = world

    @abc.abstractmethod
    def physics_step(self):
        pass

    @abc.abstractmethod
    def sense(self):
        pass

    @abc.abstractmethod
    def apply_control(self, control):
        pass

    def get_input_dim(self):
        return len(self.sense())

    @abc.abstractmethod
    def get_output_dim(self):
        pass

    @abc.abstractmethod
    def get_center_of_mass(self):
        pass

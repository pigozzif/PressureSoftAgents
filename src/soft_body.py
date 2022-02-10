import abc


class SoftBody(abc.ABC):

    def __init__(self, world):
        self.world = world

    @abc.abstractmethod
    def physics_step(self):
        pass

    @abc.abstractmethod
    def sense(self):
        pass

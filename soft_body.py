import abc

import numpy as np
from dataclasses import dataclass


@dataclass
class SpringData(object):
    rest_length: float
    min: float
    max: float


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

import abc
import math
import random

import numpy as np
import torch

from soft_body import PressureSoftBody


class BaseController(abc.ABC):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __str__(self):
        return "BaseController[input_dim={},output_dim={}]".format(self.input_dim, self.output_dim)

    @abc.abstractmethod
    def get_params(self):
        pass

    @abc.abstractmethod
    def set_params(self, params):
        pass

    @abc.abstractmethod
    def control(self, t, obs):
        pass

    @abc.abstractmethod
    def get_number_of_params(self):
        pass

    @staticmethod
    def get_number_of_params_for_controller(config):
        if config["brain"] == "random":
            return 0
        elif config["brain"] == "phase":
            return config["n_masses"] + 1 + 2
        elif config["brain"] == "inflate":
            return 0
        elif config["brain"] == "mlp":
            return (config["n_masses"] * 3 + 3) * (config["n_masses"] + 1) + config["n_masses"] + 1
        raise ValueError("Invalid controller name: {}".format(config.brain))

    @classmethod
    def create_controller(cls, config, input_dim, output_dim, brain, solution):
        if brain == "random":
            controller = RandomController(input_dim, output_dim)
        elif brain == "phase":
            controller = PhaseController(input_dim, output_dim)
        elif brain == "inflate":
            controller = InflateController(input_dim, output_dim,
                                           PressureSoftBody.get_pressure_at_rest(config["T"],
                                                                                 config["mass"], config["r"]) / 100)
        elif brain == "mlp":
            controller = MLPController(input_dim, output_dim, config["control_pressure"])
        else:
            raise ValueError("Invalid controller name: {}".format(brain))
        controller.set_params(solution)
        return controller


class RandomController(BaseController):

    def __init__(self, input_dim, output_dim):
        BaseController.__init__(self, input_dim, output_dim)

    def __str__(self):
        return super(RandomController, self).__str__().replace("Base", "Random")

    def get_params(self):
        return np.empty(0)

    def set_params(self, params):
        pass

    def control(self, t, obs):
        return np.random.random(self.output_dim) * 2.0 - 1.0

    def get_number_of_params(self):
        return 0


class PhaseController(BaseController):

    def __init__(self, input_dim, output_dim):
        BaseController.__init__(self, input_dim, output_dim)
        self.freq = random.random()
        self.ampl = random.random()
        self.phases = []

    def get_params(self):
        return np.concatenate([self.freq, self.ampl, self.phases])

    def set_params(self, params):
        self.freq = params[0]
        self.ampl = params[1]
        self.phases = params[2:]

    def control(self, t, obs):
        return np.sin([2 * math.pi * self.freq * t * self.phases[i] * self.ampl for i in range(self.output_dim)])

    def get_number_of_params(self):
        return self.output_dim + 2


class InflateController(BaseController):

    def __init__(self, input_dim, output_dim, delta_p):
        BaseController.__init__(self, input_dim, output_dim)
        self.delta_p = delta_p

    def get_params(self):
        return np.empty(0)

    def set_params(self, params):
        pass

    def control(self, t, obs):
        if t < 360:
            return np.zeros(self.output_dim)
        return np.array([0 if i < self.output_dim - 1 else self.delta_p for i in range(self.output_dim)])

    def get_number_of_params(self):
        return 0


class MLPController(BaseController):

    def __init__(self, input_dim, output_dim, control_pressure):
        BaseController.__init__(self, input_dim, output_dim)
        self.joint_nn = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_dim, out_features=self.output_dim - 1),
            torch.nn.Identity()
            )
        if control_pressure:
            self.pressure_nn = torch.nn.Sequential(torch.nn.Linear(in_features=self.input_dim, out_features=1),
                                                   torch.nn.Identity()
                                                   )
        self.control_pressure = control_pressure

    def __str__(self):
        return super(MLPController, self).__str__().replace("Base", "MLP")

    def get_params(self):
        params = np.empty(0)
        for _, p in self.joint_nn.parameters():
            params = np.append(params, p.detach().numpy())
        if not self.control_pressure:
            return params
        for _, p in self.pressure_nn.parameters():
            params = np.append(params, p.detach().numpy())
        return params

    def set_params(self, params):
        state_dict = self.joint_nn.state_dict()
        start = 0
        for key, coeffs in state_dict.items():
            num = coeffs.numel()
            state_dict[key] = torch.tensor(np.array(params[start:start + num]).reshape(state_dict[key].shape))
            start += num
        self.joint_nn.load_state_dict(state_dict)
        if not self.control_pressure:
            return
        state_dict = self.pressure_nn.state_dict()
        for key, coeffs in state_dict.items():
            num = coeffs.numel()
            state_dict[key] = torch.tensor(np.array(params[start:start + num]).reshape(state_dict[key].shape))
            start += num
        self.pressure_nn.load_state_dict(state_dict)

    def control(self, t, obs):
        obs = torch.from_numpy(obs).float()
        if not self.control_pressure:
            return self.joint_nn(obs).detach().numpy()
        return np.concatenate([self.joint_nn(obs).detach().numpy(), self.pressure_nn(obs).detach().numpy()])

    def get_number_of_params(self):
        raise self.input_dim * self.output_dim + self.output_dim
